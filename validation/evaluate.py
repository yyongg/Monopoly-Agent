"""Evaluate a trained MaskablePPO agent against the baseline opponents.

Loads a model saved by ``train.py`` and plays it through ``MonopolyEnv`` for a
number of episodes, collecting per-episode metrics (outcome, return, length,
final net worth, holdings) and reporting summary statistics. With ``--plot`` it
also renders a dashboard of graphs and can dump the raw per-episode records to
CSV.

Usage:
    python evaluate.py runs/monopoly_ppo --episodes 200
    python evaluate.py runs/monopoly_ppo --seat 0 --stochastic
    python evaluate.py runs/monopoly_selfplay --opponent runs/monopoly_ppo
    python evaluate.py runs/monopoly_ppo --episodes 500 --plot --csv out.csv
"""

import argparse
import math

import numpy as np

from engine.rl_env import MonopolyEnv
from models.tiles.properties.street_property import StreetProperty


def _liquidation_net_worth(player):
    """Human-readable net worth: cash + property + houses at sell value.

    Mortgaged property counts only its mortgage value; houses count their full
    build cost. This mirrors what a person would read off the board, so the
    reported dollars are interpretable (unlike the shaped-reward net worth).
    """
    total = float(player.balance)
    for prop in player.properties:
        total += prop.mortgage_value if prop.mortgaged else prop.price
        if isinstance(prop, StreetProperty):
            total += prop.houses * prop.house_cost()
    return total


def _count_holdings(player):
    """Returns ``(num_properties, num_houses, num_hotels)`` for ``player``."""
    houses = hotels = 0
    for prop in player.properties:
        if isinstance(prop, StreetProperty):
            if prop.houses >= 5:
                hotels += 1
            else:
                houses += prop.houses
    return len(player.properties), houses, hotels


def _wilson_interval(wins, n, z=1.96):
    """95% Wilson score interval for a binomial proportion (low, high)."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


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
        dict: summary statistics. Always includes the keys ``episodes``,
        ``wins``, ``win_rate``, ``mean_return`` and ``mean_length`` (relied on
        by ``train.py``); plus richer metrics and a ``per_episode`` dict of
        per-game arrays used for plotting.
    """
    records = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total = 0.0
        steps = 0
        truncated = False
        while not done:
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            steps += 1
            done = terminated or truncated

        agent = env.game.players[env.seat]
        opponents = [p for i, p in enumerate(env.game.players) if i != env.seat]
        agent_nw = _liquidation_net_worth(agent)
        best_opp_nw = max((_liquidation_net_worth(p) for p in opponents),
                          default=0.0)
        props, houses, hotels = _count_holdings(agent)
        won = bool(info.get("won"))

        if won:
            outcome = "win"
        elif truncated:
            outcome = "timeout"
        else:
            outcome = "bankrupt"  # game ended with the agent eliminated

        records.append({
            "won": won,
            "outcome": outcome,
            "truncated": truncated,
            "return": total,
            "length": steps,
            "agent_net_worth": agent_nw,
            "best_opp_net_worth": best_opp_nw,
            "net_worth_margin": agent_nw - best_opp_nw,
            "properties": props,
            "houses": houses,
            "hotels": hotels,
            "seat": env.seat,
        })

    wins = sum(r["won"] for r in records)

    def col(key):
        return [r[key] for r in records]

    ci_low, ci_high = _wilson_interval(wins, episodes)
    n_timeout = sum(r["outcome"] == "timeout" for r in records)
    n_bankrupt = sum(r["outcome"] == "bankrupt" for r in records)

    return {
        # --- backwards-compatible keys (used by train.py / train_selfplay.py)
        "episodes": episodes,
        "wins": wins,
        "win_rate": wins / episodes,
        "mean_return": float(np.mean(col("return"))),
        "mean_length": float(np.mean(col("length"))),
        # --- richer summary metrics
        "win_rate_ci": (ci_low, ci_high),
        "loss_bankrupt_rate": n_bankrupt / episodes,
        "timeout_rate": n_timeout / episodes,
        "median_length": float(np.median(col("length"))),
        "mean_agent_net_worth": float(np.mean(col("agent_net_worth"))),
        "mean_net_worth_margin": float(np.mean(col("net_worth_margin"))),
        "mean_properties": float(np.mean(col("properties"))),
        "mean_houses": float(np.mean(col("houses"))),
        "mean_hotels": float(np.mean(col("hotels"))),
        # --- raw per-episode arrays for plotting / CSV
        "per_episode": {key: col(key) for key in records[0]} if records else {},
    }


def plot_dashboard(stats, path, title=""):
    """Renders a multi-panel dashboard of evaluation metrics to ``path``."""
    import matplotlib
    matplotlib.use("Agg")  # no display needed; write straight to a file
    import matplotlib.pyplot as plt

    pe = stats["per_episode"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title or "Monopoly agent evaluation", fontsize=15,
                 fontweight="bold")

    # 1. Outcome breakdown.
    ax = axes[0, 0]
    counts = {
        "win": sum(o == "win" for o in pe["outcome"]),
        "loss\n(bankrupt)": sum(o == "bankrupt" for o in pe["outcome"]),
        "timeout": sum(o == "timeout" for o in pe["outcome"]),
    }
    colors = ["#2e8b57", "#c0392b", "#e0a800"]
    bars = ax.bar(counts.keys(), counts.values(), color=colors)
    ax.set_title(f"Outcomes  (win rate {stats['win_rate'] * 100:.1f}%)")
    ax.set_ylabel("episodes")
    for b, v in zip(bars, counts.values()):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v),
                ha="center", va="bottom")

    # 2. Return distribution.
    ax = axes[0, 1]
    ax.hist(pe["return"], bins=30, color="#4c72b0", edgecolor="white")
    ax.axvline(stats["mean_return"], color="black", linestyle="--",
               label=f"mean {stats['mean_return']:+.2f}")
    ax.set_title("Episode return")
    ax.set_xlabel("total shaped reward")
    ax.set_ylabel("episodes")
    ax.legend()

    # 3. Episode length.
    ax = axes[0, 2]
    ax.hist(pe["length"], bins=30, color="#8172b3", edgecolor="white")
    ax.axvline(stats["mean_length"], color="black", linestyle="--",
               label=f"mean {stats['mean_length']:.0f}")
    ax.set_title("Episode length")
    ax.set_xlabel("decisions per game")
    ax.set_ylabel("episodes")
    ax.legend()

    # 4. Final net worth: agent vs best opponent.
    ax = axes[1, 0]
    ax.hist([pe["agent_net_worth"], pe["best_opp_net_worth"]], bins=25,
            color=["#2e8b57", "#c0392b"], label=["agent", "best opponent"])
    ax.set_title("Final net worth")
    ax.set_xlabel("dollars")
    ax.set_ylabel("episodes")
    ax.legend()

    # 5. Net-worth margin over the best opponent.
    ax = axes[1, 1]
    margin = pe["net_worth_margin"]
    ax.hist(margin, bins=30, color="#55a868", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(float(np.mean(margin)), color="black", linestyle="--",
               label=f"mean {np.mean(margin):+.0f}")
    ax.set_title("Net-worth margin vs best opponent")
    ax.set_xlabel("agent − best opponent ($)")
    ax.set_ylabel("episodes")
    ax.legend()

    # 6. Holdings: properties / houses / hotels.
    ax = axes[1, 2]
    labels = ["properties", "houses", "hotels"]
    means = [stats["mean_properties"], stats["mean_houses"],
             stats["mean_hotels"]]
    bars = ax.bar(labels, means, color=["#4c72b0", "#dd8452", "#c44e52"])
    ax.set_title("Mean agent holdings at game end")
    ax.set_ylabel("count")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _write_csv(per_episode, path):
    import csv
    keys = list(per_episode)
    n = len(per_episode[keys[0]]) if keys else 0
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([per_episode[k][i] for k in keys])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to a saved MaskablePPO model (.zip)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seat", type=int, default=0,
                        help="seat the agent controls (use -1 for a random "
                             "seat each game)")
    parser.add_argument("--reward-mode", choices=["shaped", "sparse"],
                        default="shaped")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of taking the argmax")
    parser.add_argument("--opponent", default=None,
                        help="opponent seats: a model path, the literal 'fp' "
                             "for the hand-crafted FP-A/B/C trio, or omitted "
                             "for the trivial engine baseline")
    parser.add_argument("--plot", nargs="?", const="auto", default=None,
                        help="render a dashboard PNG (optionally give a path; "
                             "default: runs/eval_<model>.png)")
    parser.add_argument("--csv", default=None,
                        help="also write per-episode records to this CSV path")
    args = parser.parse_args()

    from sb3_contrib import MaskablePPO  # imported lazily; heavy dependency

    opponent_policy = None
    if args.opponent == "fp":
        from training.baselines import make_baseline_trio
        opponent_policy = make_baseline_trio()
    elif args.opponent is not None:
        from training.selfplay import policy_from_model
        opponent_policy = policy_from_model(MaskablePPO.load(args.opponent))

    seat = None if args.seat < 0 else args.seat
    env = MonopolyEnv(seat=seat, reward_mode=args.reward_mode,
                      seed=args.seed, opponent_policy=opponent_policy)
    model = MaskablePPO.load(args.model)
    stats = run_evaluation(model, env, episodes=args.episodes,
                           deterministic=not args.stochastic)
    env.close()

    opp = args.opponent if args.opponent else "baseline"
    seat_label = "random" if seat is None else str(args.seat)
    lo, hi = stats["win_rate_ci"]
    print(f"Evaluated {stats['episodes']} episodes "
          f"(agent at seat {seat_label}, opponents: {opp}):")
    print(f"  win rate       : {stats['win_rate'] * 100:.1f}%  "
          f"({stats['wins']}/{stats['episodes']})  "
          f"[95% CI {lo * 100:.1f}–{hi * 100:.1f}%]")
    print(f"  loss (bankrupt): {stats['loss_bankrupt_rate'] * 100:.1f}%")
    print(f"  timeouts       : {stats['timeout_rate'] * 100:.1f}%")
    print(f"  mean return    : {stats['mean_return']:+.2f}")
    print(f"  episode length : mean {stats['mean_length']:.0f}, "
          f"median {stats['median_length']:.0f} decisions")
    print(f"  agent net worth: mean ${stats['mean_agent_net_worth']:,.0f}")
    print(f"  net-worth edge : mean ${stats['mean_net_worth_margin']:+,.0f} "
          f"vs best opponent")
    print(f"  holdings       : {stats['mean_properties']:.1f} props, "
          f"{stats['mean_houses']:.1f} houses, "
          f"{stats['mean_hotels']:.1f} hotels (mean)")

    if args.csv:
        _write_csv(stats["per_episode"], args.csv)
        print(f"  wrote per-episode CSV -> {args.csv}")

    if args.plot is not None:
        import os
        if args.plot == "auto":
            base = os.path.splitext(os.path.basename(args.model))[0]
            os.makedirs("runs", exist_ok=True)
            plot_path = os.path.join("runs", f"eval_{base}.png")
        else:
            plot_path = args.plot
        title = f"Monopoly eval: {os.path.basename(args.model)} vs {opp}"
        plot_dashboard(stats, plot_path, title=title)
        print(f"  wrote dashboard -> {plot_path}")


if __name__ == "__main__":
    main()
