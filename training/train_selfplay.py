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
    # Self-play knobs.
    parser.add_argument("--pool-dir", default="runs/sp_pool",
                        help="directory of opponent snapshots")
    parser.add_argument("--snapshot-freq", type=int, default=100_000,
                        help="env steps between adding a snapshot to the pool")
    parser.add_argument("--pool-size", type=int, default=10,
                        help="max snapshots kept (newest win)")
    parser.add_argument("--baseline-prob", type=float, default=0.2,
                        help="probability an episode uses baseline opponents")
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

    venv = SubprocVecEnv([
        make_selfplay_env(i, args.seed, args.seat, args.reward_mode,
                          args.max_turns, args.pool_dir, args.baseline_prob,
                          args.opp_deterministic)
        for i in range(args.n_envs)
    ])
    venv = VecMonitor(venv)

    tensorboard_log = args.logdir
    if importlib.util.find_spec("tensorboard") is None:
        print("tensorboard not installed; disabling TensorBoard logging "
              "(pip install tensorboard to enable).")
        tensorboard_log = None

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        venv,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
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
    print(f"Saved model to {args.save_path}.zip")
    venv.close()

    # Final evaluation against the fixed baseline (a stationary yardstick).
    from validation.evaluate import run_evaluation

    eval_env = MonopolyEnv(seat=0, reward_mode=args.reward_mode,
                           seed=args.seed + 10_000)
    stats = run_evaluation(model, eval_env, episodes=args.eval_episodes)
    eval_env.close()
    print(f"Final win rate vs baseline: {stats['win_rate'] * 100:.1f}% "
          f"({stats['wins']}/{stats['episodes']}), "
          f"mean return {stats['mean_return']:+.2f}")


if __name__ == "__main__":
    main()
