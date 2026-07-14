"""Self-play machinery for training a Monopoly agent against itself.

The pieces here turn the single-agent ``MonopolyEnv`` into a self-play setup:

* :func:`policy_from_model` wraps a trained MaskablePPO model as the
  ``(observation, action_mask) -> action`` callable the env's opponent seats
  expect.
* :class:`OpponentPool` is an env ``opponent_provider``: at each episode it
  samples an opponent from a directory of policy snapshots (or the baseline with
  some probability), loading and caching models lazily.
* :class:`SelfPlayCallback` periodically snapshots the learner into that pool
  directory, so the opponents track the improving agent. Envs pick up new
  snapshots at their next ``reset`` simply by re-scanning the directory -- no
  cross-process messaging required, which is what makes this work cleanly under
  ``SubprocVecEnv``.
* :func:`make_selfplay_env` builds a picklable env factory for the vec env.

Because opponents are sampled per episode and the pool starts empty, training
naturally follows a curriculum: baseline opponents first, then a growing mix of
past snapshots of the agent.
"""

import glob
import os
import random
from collections import OrderedDict

from stable_baselines3.common.callbacks import BaseCallback

from engine.rl_env import MonopolyEnv
from training.baselines import make_baseline_trio


def policy_from_model(model, deterministic=False):
    """Wraps a MaskablePPO ``model`` (or a bare policy) as an opponent callable.

    Returns ``(observation, action_mask_bool) -> action_index``. Opponents
    default to sampling (``deterministic=False``) for behavioural variety.
    """

    def policy(observation, action_mask):
        action, _ = model.predict(
            observation, action_masks=action_mask, deterministic=deterministic)
        return int(action)

    return policy


def load_opponent_policy(path):
    """Loads a snapshot as an inference-only policy, and *nothing else*.

    An opponent never learns, so everything ``MaskablePPO.load`` builds around the
    network -- above all the Adam optimizer's state -- is dead weight that every
    one of the (up to 64) env worker processes would otherwise hold in memory for
    every cached snapshot. Training runs under a **32 GB cgroup cap** here, and
    the torch runtime alone costs a few hundred MB per worker, so this is not a
    micro-optimisation: keeping the whole algorithm object is what pushed the
    workers over the cap and got them OOM-killed mid-run.

    Returns the policy (it has the ``.predict(obs, action_masks=...)`` that
    :func:`policy_from_model` needs), or raises if the zip can't be read.
    """
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(path, device="cpu")
    policy = model.policy
    policy.set_training_mode(False)
    policy.optimizer = None      # inference only: drop the optimizer state
    model.policy = None          # break the cycle so the algorithm object is freed
    del model
    return policy


class OpponentPool:
    """Env ``opponent_provider`` sampling opponents from a snapshot directory.

    Args:
        pool_dir (str): Directory holding ``*.zip`` policy snapshots.
        baseline_prob (float): Probability of returning the trivial engine
            baseline (``None``) instead of a snapshot.
        fp_prob (float): Probability of returning the hand-crafted FP-A/B/C trio
            (:func:`training.baselines.make_baseline_trio`) -- real strategy
            opponents available from the first step, before the snapshot pool
            fills. Partitions the non-snapshot mass together with
            ``baseline_prob``.
        cache_size (int): Max number of loaded models kept in memory (LRU).
        deterministic (bool): Whether sampled opponents act greedily.
        seed (int | None): Seed for the sampling RNG.
    """

    def __init__(self, pool_dir, baseline_prob=0.2, fp_prob=0.0, cache_size=8,
                 deterministic=False, seed=None, expected_obs_shape=None,
                 expected_action_n=None, num_opponents=3):
        self.pool_dir = pool_dir
        self.baseline_prob = baseline_prob
        self.fp_prob = fp_prob
        self.cache_size = cache_size
        self.deterministic = deterministic
        self.num_opponents = num_opponents
        # Snapshots trained against an older obs *or* action space would crash at
        # predict time; skip any whose observation or action space doesn't match.
        # Both must be checked -- e.g. a snapshot from after an obs change but
        # before an action-space change has a matching obs shape yet a stale
        # action count, and would only blow up when predict is called.
        self.expected_obs_shape = (tuple(expected_obs_shape)
                                   if expected_obs_shape is not None else None)
        self.expected_action_n = expected_action_n
        self._rng = random.Random(seed)
        self._cache = OrderedDict()

    def __call__(self):
        """Samples an opponent roster for one episode.

        With probability ``fp_prob`` the roster is the **whole FP-A/B/C trio**,
        and with ``baseline_prob`` it is all trivial baselines. Keeping these
        rosters coherent matters: the FP trio is the benchmark the agent is
        measured against, and sprinkling FP bots in seat-by-seat would make a
        full trio vanishingly rare (0.3 per seat is a 2.7% chance of all three).

        Otherwise the roster is drawn from the snapshot pool **independently per
        seat**, so the agent faces an old self and a recent self in the same
        game rather than three copies of one snapshot -- much more varied play
        out of the same pool, and the direct counter to the strategic cycling
        that a single-opponent roster invites.

        Returns a list the env deals across the opponent seats; a ``None`` entry
        leaves that seat on the engine baseline.
        """
        r = self._rng.random()
        if r < self.fp_prob:
            return make_baseline_trio()
        if r < self.fp_prob + self.baseline_prob:
            return None
        return [self._sample_snapshot() for _ in range(self.num_opponents)]

    def _sample_snapshot(self):
        """A policy from a uniformly-drawn pool snapshot, or ``None`` (baseline)
        when the pool is empty -- as it is early in training."""
        paths = glob.glob(os.path.join(self.pool_dir, "*.zip"))
        if not paths:
            return None
        model = self._load(self._rng.choice(paths))
        if model is None:
            return None
        return policy_from_model(model, self.deterministic)

    def _load(self, path):
        """Loads (and LRU-caches) a snapshot policy; ``None`` if it can't be read.

        A snapshot may be pruned by the callback between the directory scan and
        the load, so failures fall back to the baseline rather than crash.
        """
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        try:
            policy = load_opponent_policy(path)
        except Exception:
            return None
        obs_mismatch = (self.expected_obs_shape is not None
                        and tuple(policy.observation_space.shape)
                        != self.expected_obs_shape)
        act_mismatch = (self.expected_action_n is not None
                        and policy.action_space.n != self.expected_action_n)
        if obs_mismatch or act_mismatch:
            # Stale snapshot from an older obs/action space: cache the rejection
            # (as None) so we don't reload it every episode, and fall back to
            # baseline this time.
            policy = None
        self._cache[path] = policy
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return policy


class SelfPlayCallback(BaseCallback):
    """Snapshots the learner into the opponent pool at a fixed step interval.

    The pool is capped at ``pool_size``, but *which* snapshots it keeps matters
    as much as how many. Pruning to the newest N (the old behaviour) leaves a
    sliding window of near-copies of the agent's recent self -- at a 100k
    snapshot interval and a cap of 10, it never faces anything older than its
    last million steps. That is the classic setup for strategic cycling: the
    agent learns to beat its current self, drifts, forgets the counter to what it
    used to do, and can cycle indefinitely without getting stronger.

    So eviction keeps a *spread over training history* instead:

    * the **earliest** snapshot is an anchor and is never evicted -- a permanent
      reference point that the agent must keep beating;
    * the newest is always kept (it is the strongest);
    * otherwise the snapshot whose two neighbours in time are closest is dropped,
      which thins the most crowded stretch of history and leaves the survivors
      spread across the whole run.
    """

    def __init__(self, pool_dir, snapshot_freq, pool_size=10, verbose=0):
        super().__init__(verbose)
        self.pool_dir = pool_dir
        self.snapshot_freq = snapshot_freq
        self.pool_size = pool_size
        self._last_snapshot = 0

    def _on_training_start(self):
        os.makedirs(self.pool_dir, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps - self._last_snapshot >= self.snapshot_freq:
            self._snapshot()
            self._last_snapshot = self.num_timesteps
        return True

    def _snapshot(self):
        path = os.path.join(
            self.pool_dir, f"snapshot_{self.num_timesteps:09d}.zip")
        self.model.save(path)
        # Filenames are zero-padded step counts, so lexical order is time order.
        snaps = sorted(glob.glob(os.path.join(self.pool_dir, "snapshot_*.zip")))
        for old in self._evict(snaps):
            try:
                os.remove(old)
            except OSError:
                pass
        if self.verbose:
            print(f"[self-play] snapshot at {self.num_timesteps} steps "
                  f"(pool size {min(len(snaps), self.pool_size)})")

    def _evict(self, snaps):
        """Which snapshots to delete, keeping a spread across training history.

        Returns the paths to remove (never the first or the last).
        """
        drop = []
        keep = list(snaps)
        while len(keep) > self.pool_size:
            steps = [_snapshot_steps(p) for p in keep]
            # Interior snapshot with the smallest gap to its neighbours: the most
            # redundant point in the timeline.
            gaps = [(steps[i + 1] - steps[i - 1], i)
                    for i in range(1, len(keep) - 1)]
            if not gaps:
                break
            _, victim = min(gaps)
            drop.append(keep.pop(victim))
        return drop


def _snapshot_steps(path):
    """The training step a snapshot filename encodes (0 if unparseable)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def make_selfplay_env(rank, seed, seat, reward_mode, max_turns, pool_dir,
                      baseline_prob, opp_deterministic, fp_prob=0.0,
                      gamma=0.999, cfg=None, cache_size=3):
    """Returns a picklable factory for one self-play ``MonopolyEnv`` worker.

    ``gamma`` is handed to the env so its potential-based shaping discounts with
    the *same* factor the learner does -- otherwise the shaping is not the
    policy-invariant transform it is meant to be. ``cfg`` is the run's
    :class:`~engine.config.RewardConfig` (a frozen dataclass, so it pickles
    across to the worker processes), which is what makes a coefficient sweep
    possible without monkeypatching module globals.

    ``cache_size`` is the per-worker LRU of loaded snapshot policies, and it is
    **the memory knob**: see :func:`memory_budget_gb`.
    """

    def _init():
        env = MonopolyEnv(seat=seat, reward_mode=reward_mode,
                          max_turns=max_turns, seed=seed + rank, gamma=gamma,
                          cfg=cfg)
        # Per-worker LRU cache of loaded snapshot policies.
        #
        # MEMORY IS THE BINDING CONSTRAINT HERE, not disk: there is one of these
        # caches per env *worker process*, so the cost is (n_envs x cache_size)
        # resident policies. The default of 3 is one per opponent seat, so an
        # episode that draws three distinct snapshots still never reloads within
        # itself; going above that buys reuse *across* episodes and little else.
        # Raise it only alongside the job's --mem (see memory_budget_gb): it was
        # a cache of 6 x 64 workers that blew a 32 GB cgroup cap and got the
        # workers OOM-killed, which surfaces in the parent as an opaque
        # BrokenPipeError/EOFError with no traceback (the kernel SIGKILLs the
        # child, so there is nothing to raise).
        pool = OpponentPool(pool_dir, baseline_prob=baseline_prob,
                            fp_prob=fp_prob,
                            deterministic=opp_deterministic, seed=seed + rank,
                            cache_size=cache_size,
                            expected_obs_shape=env.observation_space.shape,
                            expected_action_n=env.action_space.n)
        # Read at each reset (see MonopolyEnv.reset); safe to set post-init.
        env._opponent_provider = pool
        return env

    return _init


# Fitted against two *measured peaks* on this cluster (obs 265, net 256x256,
# cache_size=3): 32 envs peaked at 14.7 GB and 64 envs at 27.0 GB resident. That
# is 0.384 GB per additional worker and a 2.4 GB intercept for the learner.
#
# Splitting the per-worker slope into the fixed part and the cache part: a
# forkserver child pays for its own torch runtime (~0.32 GB, the dominant term --
# it is why envs are expensive here at all) plus its resident opponent policies.
# An inference-only 265-dim 256x256 MaskablePPO policy is ~0.02 GB, so cache=3
# adds ~0.06 and 0.32 + 0.06 = 0.38, recovering the measured slope.
_GB_LEARNER = 2.4
_GB_PER_ENV = 0.32              # torch runtime etc., excluding the snapshot cache
_GB_PER_CACHED_POLICY = 0.02


def memory_budget_gb(n_envs, cache_size=3, headroom=1.15):
    """Rough RAM a self-play run needs, in GB -- what to pass to sbatch --mem.

    The constants are fit to observed *peaks*, so the headroom here is genuine
    slack rather than a second safety factor stacked on a conservative estimate:
    15% covers the transient of several workers deserialising a snapshot at once.
    Erring high is still the right bias -- overshooting --mem costs a little queue
    time, while undershooting costs the entire run, and it does so silently
    (see :func:`cgroup_memory_limit_gb`).
    """
    resident = (_GB_LEARNER
                + n_envs * (_GB_PER_ENV + cache_size * _GB_PER_CACHED_POLICY))
    return resident * headroom


def max_envs_for_cap(cap_gb, cache_size=3, headroom=1.15):
    """Largest ``--n-envs`` that fits in ``cap_gb`` -- the inverse of the above."""
    per_env = _GB_PER_ENV + cache_size * _GB_PER_CACHED_POLICY
    return max(1, int((cap_gb / headroom - _GB_LEARNER) / per_env))


def cgroup_memory_limit_gb():
    """This process's cgroup-v2 memory cap in GB, or ``None`` if uncapped.

    Under SLURM this is the real ceiling and ``free`` does **not** show it --
    ``free`` reports the whole physical host (251 GB here), while the job is held
    to its ``--mem`` (32 GB), so the host looks reassuringly empty right up until
    a worker is OOM-killed. The limit is set at submission and is not writable
    from inside the job. Walks up from the process's own cgroup because the cap
    may sit on any ancestor (SLURM sets it on the job and step scopes, not the
    leaf task).
    """
    try:
        with open("/proc/self/cgroup") as fh:
            rel = fh.readline().strip().split(":")[-1]
    except OSError:
        return None
    path = os.path.join("/sys/fs/cgroup", rel.lstrip("/"))
    limits = []
    while path.startswith("/sys/fs/cgroup"):
        try:
            with open(os.path.join(path, "memory.max")) as fh:
                raw = fh.read().strip()
            if raw != "max":
                limits.append(int(raw) / 1024 ** 3)
        except (OSError, ValueError):
            pass
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return min(limits) if limits else None
