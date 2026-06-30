"""Train a MaskablePPO agent to play Monopoly via ``MonopolyEnv``.

Runs several environments in parallel with ``SubprocVecEnv`` (each
``MonopolyEnv`` owns a worker thread, so subprocesses are the correct way to
parallelize), trains ``sb3_contrib.MaskablePPO`` against the engine's baseline
opponents, logs to TensorBoard, checkpoints periodically, and saves the final
model. A short evaluation is printed at the end.

Usage:
    python train.py --timesteps 200000 --n-envs 8
    tensorboard --logdir runs/tb        # to watch progress

The agent controls one seat (``--seat``); the other players use the engine's
built-in baseline policy.
"""

import argparse
import importlib.util

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from engine.rl_env import MonopolyEnv


def make_env(rank, seed, seat, reward_mode, max_turns):
    """Returns a picklable factory for one seeded ``MonopolyEnv`` worker."""

    def _init():
        return MonopolyEnv(seat=seat, reward_mode=reward_mode,
                           max_turns=max_turns, seed=seed + rank)

    return _init


class WinRateCallback(BaseCallback):
    """Logs the rolling win rate over recent finished episodes to TensorBoard."""

    def __init__(self, window=200):
        super().__init__()
        self.window = window
        self._results = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "won" in info:
                self._results.append(1.0 if info["won"] else 0.0)
                if len(self._results) > self.window:
                    self._results.pop(0)
        if self._results:
            self.logger.record("rollout/win_rate", float(np.mean(self._results)))
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8,
                        help="parallel environments (SubprocVecEnv)")
    parser.add_argument("--seat", type=int, default=None,
                        help="seat the agent controls (default: random each episode)")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"],
                        default="shaped")
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=512,
                        help="rollout length per env before each update")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="discount; high because Monopoly games are long")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-path", default="runs/monopoly_ppo",
                        help="path for the saved model (.zip appended)")
    parser.add_argument("--checkpoint-dir", default="runs/checkpoints")
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="total env steps between checkpoints (0 disables)")
    parser.add_argument("--logdir", default="runs/tb",
                        help="TensorBoard log directory")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--progress", action="store_true",
                        help="show a progress bar (needs tqdm + rich)")
    args = parser.parse_args()

    venv = SubprocVecEnv([
        make_env(i, args.seed, args.seat, args.reward_mode, args.max_turns)
        for i in range(args.n_envs)
    ])
    venv = VecMonitor(venv)

    # TensorBoard logging needs the optional ``tensorboard`` package; skip it
    # gracefully (training still runs, logging to stdout) if it is absent.
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

    callbacks = [WinRateCallback()]
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

    # Final evaluation on a fresh single environment vs the baseline.
    from evaluate import run_evaluation

    eval_env = MonopolyEnv(seat=0, reward_mode=args.reward_mode,
                           seed=args.seed + 10_000)
    stats = run_evaluation(model, eval_env, episodes=args.eval_episodes)
    eval_env.close()
    print(f"Final win rate vs baseline: {stats['win_rate'] * 100:.1f}% "
          f"({stats['wins']}/{stats['episodes']}), "
          f"mean return {stats['mean_return']:+.2f}")


if __name__ == "__main__":
    main()
