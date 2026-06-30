"""Evaluate a trained MaskablePPO agent against the baseline opponents.

Loads a model saved by ``train.py`` and plays it through ``MonopolyEnv`` for a
number of episodes, reporting win rate, mean return, and mean episode length.

Usage:
    python evaluate.py runs/monopoly_ppo --episodes 200
    python evaluate.py runs/monopoly_ppo --seat 0 --stochastic
    python evaluate.py runs/monopoly_selfplay --opponent runs/monopoly_ppo
"""

import argparse

import numpy as np

from engine.rl_env import MonopolyEnv


def run_evaluation(model, env, episodes=100, deterministic=True):
    """Plays ``episodes`` games with ``model`` on ``env`` and returns stats.

    Args:
        model: A trained MaskablePPO model (anything with a ``predict`` taking
            ``action_masks``).
        env (MonopolyEnv): The environment to play in.
        episodes (int): Number of games to play.
        deterministic (bool): Greedy actions if True, else sample from the
            (masked) policy.

    Returns:
        dict: ``episodes``, ``wins``, ``win_rate``, ``mean_return``,
        ``mean_length``.
    """
    wins = 0
    returns = []
    lengths = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            steps += 1
            done = terminated or truncated
        wins += bool(info.get("won"))
        returns.append(total)
        lengths.append(steps)
    return {
        "episodes": episodes,
        "wins": wins,
        "win_rate": wins / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to a saved MaskablePPO model (.zip)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seat", type=int, default=0,
                        help="seat the agent controls")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"],
                        default="shaped")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of taking the argmax")
    parser.add_argument("--opponent", default=None,
                        help="path to a model that drives the opponent seats "
                             "(default: engine baseline opponents)")
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO  # imported lazily; heavy dependency

    opponent_policy = None
    if args.opponent is not None:
        from selfplay import policy_from_model
        opponent_policy = policy_from_model(MaskablePPO.load(args.opponent))

    env = MonopolyEnv(seat=args.seat, reward_mode=args.reward_mode,
                      seed=args.seed, opponent_policy=opponent_policy)
    model = MaskablePPO.load(args.model)
    stats = run_evaluation(model, env, episodes=args.episodes,
                           deterministic=not args.stochastic)
    env.close()

    opp = args.opponent if args.opponent else "baseline"
    print(f"Evaluated {stats['episodes']} episodes "
          f"(agent at seat {args.seat}, opponents: {opp}):")
    print(f"  win rate    : {stats['win_rate'] * 100:.1f}%  "
          f"({stats['wins']}/{stats['episodes']})")
    print(f"  mean return : {stats['mean_return']:+.2f}")
    print(f"  mean length : {stats['mean_length']:.0f} decisions")


if __name__ == "__main__":
    main()
