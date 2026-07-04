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
    """Wraps a MaskablePPO ``model`` as an env opponent policy callable.

    Returns ``(observation, action_mask_bool) -> action_index``. Opponents
    default to sampling (``deterministic=False``) for behavioural variety.
    """

    def policy(observation, action_mask):
        action, _ = model.predict(
            observation, action_masks=action_mask, deterministic=deterministic)
        return int(action)

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
                 expected_action_n=None):
        self.pool_dir = pool_dir
        self.baseline_prob = baseline_prob
        self.fp_prob = fp_prob
        self.cache_size = cache_size
        self.deterministic = deterministic
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
        """Samples an opponent spec for one episode.

        Returns the FP trio (prob ``fp_prob``), the trivial engine baseline
        (``None``, prob ``baseline_prob``), or a sampled snapshot policy.
        """
        r = self._rng.random()
        if r < self.fp_prob:
            return make_baseline_trio()
        if r < self.fp_prob + self.baseline_prob:
            return None
        paths = glob.glob(os.path.join(self.pool_dir, "*.zip"))
        if not paths:
            return None  # empty pool early in training -> baseline opponents
        model = self._load(self._rng.choice(paths))
        if model is None:
            return None
        return policy_from_model(model, self.deterministic)

    def _load(self, path):
        """Loads (and LRU-caches) a snapshot; ``None`` if it can't be read.

        A snapshot may be pruned by the callback between the directory scan and
        the load, so failures fall back to the baseline rather than crash.
        """
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        try:
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(path, device="cpu")
        except Exception:
            return None
        obs_mismatch = (self.expected_obs_shape is not None
                        and tuple(model.observation_space.shape)
                        != self.expected_obs_shape)
        act_mismatch = (self.expected_action_n is not None
                        and model.action_space.n != self.expected_action_n)
        if obs_mismatch or act_mismatch:
            # Stale snapshot from an older obs/action space: cache the rejection
            # (as None) so we don't reload it every episode, and fall back to
            # baseline this time.
            model = None
        self._cache[path] = model
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return model


class SelfPlayCallback(BaseCallback):
    """Snapshots the learner into the opponent pool at a fixed step interval.

    Keeps only the most recent ``pool_size`` snapshots so the pool stays bounded
    and weighted toward recent (stronger) versions of the agent.
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
        snaps = sorted(glob.glob(os.path.join(self.pool_dir, "snapshot_*.zip")))
        for old in snaps[:-self.pool_size]:  # prune all but the newest pool_size
            try:
                os.remove(old)
            except OSError:
                pass
        if self.verbose:
            kept = min(len(snaps), self.pool_size)
            print(f"[self-play] snapshot at {self.num_timesteps} steps "
                  f"(pool size {kept})")


def make_selfplay_env(rank, seed, seat, reward_mode, max_turns, pool_dir,
                      baseline_prob, opp_deterministic, fp_prob=0.0):
    """Returns a picklable factory for one self-play ``MonopolyEnv`` worker."""

    def _init():
        env = MonopolyEnv(seat=seat, reward_mode=reward_mode,
                          max_turns=max_turns, seed=seed + rank)
        # Small per-worker cache: with many parallel envs the opponent models
        # are the dominant memory cost, so keep only a couple resident (they're
        # cheap to reload from disk) to stay well under tight memory caps.
        pool = OpponentPool(pool_dir, baseline_prob=baseline_prob,
                            fp_prob=fp_prob,
                            deterministic=opp_deterministic, seed=seed + rank,
                            cache_size=2,
                            expected_obs_shape=env.observation_space.shape,
                            expected_action_n=env.action_space.n)
        # Read at each reset (see MonopolyEnv.reset); safe to set post-init.
        env._opponent_provider = pool
        return env

    return _init
