"""Train a Monopoly agent with self-play (MaskablePPO + a snapshot pool).

Opponents start as the engine baseline and are progressively replaced by past
snapshots of the agent (sampled per episode from a pool directory), so the agent
learns against steadily stronger versions of itself. See ``selfplay.py`` for the
mechanism and the project roadmap (Phase 2) for the rationale.

Usage:
    python train_selfplay.py --timesteps 1000000 --n-envs 8
    tensorboard --logdir runs/sp_tb

Final evaluation is reported against the fixed baseline opponents, which is a
stationary yardstick for tracking progress across runs.
"""

import os

# Keep each process single-threaded for the math libraries (see train.py for
# why). Set before importing numpy/torch; ``setdefault`` lets an explicit
# environment variable override.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib.util

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from engine.config import RewardConfig, save_run_metadata
from engine.rl_env import MonopolyEnv
from training.selfplay import SelfPlayCallback, make_selfplay_env
from training.train import WinRateCallback

torch.set_num_threads(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seat", type=int, default=None,
                        help="seat the agent controls (default: random each episode)")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"],
                        default="shaped")
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    # Previously every one of these was left at an SB3 default and none was
    # reachable from the CLI, so a sweep could not touch them.
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256],
                        help="hidden layer sizes for both the policy and value "
                             "heads (SB3's default is a 64x64 tanh MLP, which is "
                             "very small for a 265-dim obs and 211 actions)")
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03,
                        help="early-stop an update that moves the policy too far "
                             "(SB3 default is None: no limit at all)")
    parser.add_argument("--lr-schedule", choices=["constant", "linear"],
                        default="linear",
                        help="decay the learning rate to 0 over the run")
    # Reward coefficients. Anything in RewardConfig can be swept this way; these
    # two are exposed because solvency_penalty_coef is the one uncalibrated knob.
    parser.add_argument("--solvency-penalty-coef", type=float, default=None,
                        help="per-turn drag for holding less cash than the "
                             "board's rent threat warrants (default: "
                             f"{RewardConfig().solvency_penalty_coef})")
    parser.add_argument("--solvency-cushion-turns", type=float, default=None,
                        help="rounds of expected rent the cash cushion should "
                             "cover (default: "
                             f"{RewardConfig().solvency_cushion_turns})")
    # Self-play knobs.
    parser.add_argument("--pool-dir", default="runs/sp_pool",
                        help="directory of opponent snapshots")
    parser.add_argument("--snapshot-freq", type=int, default=100_000,
                        help="env steps between adding a snapshot to the pool")
    parser.add_argument("--pool-size", type=int, default=10,
                        help="max snapshots kept (newest win)")
    parser.add_argument("--baseline-prob", type=float, default=0.2,
                        help="probability an episode uses the trivial engine "
                             "baseline opponents")
    parser.add_argument("--fp-prob", type=float, default=0.3,
                        help="probability an episode uses the hand-crafted "
                             "FP-A/B/C trio as opponents")
    parser.add_argument("--opp-deterministic", action="store_true",
                        help="sampled opponents act greedily")
    # I/O.
    parser.add_argument("--save-path", default="runs/monopoly_ppo")
    parser.add_argument("--checkpoint-dir", default="runs/sp_checkpoints")
    parser.add_argument("--save-freq", type=int, default=200_000)
    parser.add_argument("--logdir", default="runs/sp_tb")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    # One RewardConfig for the whole run: it reaches every worker env, and it is
    # written to the model's metadata sidecar so the checkpoint can be tied back
    # to the economics it learned under.
    overrides = {k: v for k, v in (
        ("solvency_penalty_coef", args.solvency_penalty_coef),
        ("solvency_cushion_turns", args.solvency_cushion_turns),
    ) if v is not None}
    cfg = RewardConfig(**overrides)

    venv = SubprocVecEnv([
        make_selfplay_env(i, args.seed, args.seat, args.reward_mode,
                          args.max_turns, args.pool_dir, args.baseline_prob,
                          args.opp_deterministic, fp_prob=args.fp_prob,
                          gamma=args.gamma, cfg=cfg)
        for i in range(args.n_envs)
    ])
    venv = VecMonitor(venv)

    tensorboard_log = args.logdir
    if importlib.util.find_spec("tensorboard") is None:
        print("tensorboard not installed; disabling TensorBoard logging "
              "(pip install tensorboard to enable).")
        tensorboard_log = None

    # Anneal the learning rate to 0 over the run: SB3 accepts a callable of
    # remaining progress (1 -> 0).
    if args.lr_schedule == "linear":
        base_lr = args.learning_rate

        def learning_rate(progress_remaining):
            return progress_remaining * base_lr
    else:
        learning_rate = args.learning_rate

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        venv,
        policy_kwargs=dict(
            net_arch=dict(pi=list(args.net_arch), vf=list(args.net_arch)),
            activation_fn=torch.nn.ReLU,
        ),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        learning_rate=learning_rate,
        seed=args.seed,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )

    callbacks = [
        WinRateCallback(),
        SelfPlayCallback(args.pool_dir, snapshot_freq=args.snapshot_freq,
                         pool_size=args.pool_size, verbose=1),
    ]
    if args.save_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(args.save_freq // args.n_envs, 1),
            save_path=args.checkpoint_dir,
            name_prefix="monopoly_ppo",
        ))

    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                progress_bar=args.progress)
    model.save(args.save_path)
    # Record the economics this model actually learned under, so evaluation and
    # the GUI don't silently apply today's defaults to it.
    meta_path = save_run_metadata(args.save_path, cfg, args)
    print(f"Saved model to {args.save_path}.zip (metadata: {meta_path})")
    venv.close()

    # Final evaluation against the hand-crafted FP-A/B/C trio -- a strong,
    # stationary yardstick (the trivial engine baseline is too weak to be
    # informative).
    from validation.evaluate import run_evaluation
    from training.baselines import make_baseline_trio

    eval_env = MonopolyEnv(seat=0, reward_mode=args.reward_mode,
                           seed=args.seed + 10_000, gamma=args.gamma, cfg=cfg,
                           opponent_policy=make_baseline_trio())
    stats = run_evaluation(model, eval_env, episodes=args.eval_episodes)
    eval_env.close()
    print(f"Final win rate vs FP trio: {stats['win_rate'] * 100:.1f}% "
          f"({stats['wins']}/{stats['episodes']}), "
          f"mean return {stats['mean_return']:+.2f}")


if __name__ == "__main__":
    main()
