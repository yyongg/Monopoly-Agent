"""Simulate many self-play games with a trained model and analyse its strategy.

Every seat is driven by the *same* trained MaskablePPO policy, so each game is
the model playing against copies of itself. Games are played to completion (one
survivor, or a turn-cap timeout) over many random seeds, and the script reports
what winning play looks like:

* **Seat win rate** -- does turn order (first-move) matter?
* **Winning monopolies** -- which colour sets do winners tend to hold, which
  sets most predict a win (P(win | the player ever completed that set)), and
  which *first* set most predicts a win (P(win | the player's first completed
  monopoly was that set)) -- a joint set-choice + tempo signal.
* **Winning properties** -- the individual tiles most often held by the winner.
* **Tempo** -- how early the winner completed their first monopoly.
* **Why winners win** -- winners' vs losers' monopoly counts and net worth.

Unlike ``evaluate.py`` (which watches one agent seat and stops when that seat is
out), this harness runs the whole game so the true winner is always observed. It
reuses ``MonopolyEnv``'s internals (board construction, per-seat deciders, legal
masks, observations) but runs the game loop synchronously -- with every seat on a
policy there is no need for the env's background worker thread.

Usage:
    python simulate.py                                  # 200 games, default model
    python simulate.py runs/monopoly_ppo --games 500
    python simulate.py runs/monopoly_selfplay --games 300 --plot --csv sim.csv
"""

import argparse
import os
from collections import Counter

import numpy as np

from engine.rl_env import MonopolyEnv
from models.tiles.properties.street_property import StreetProperty
from models.tiles.properties.railroad import Railroad


DEFAULT_MODEL = "runs/monopoly_ppo"


# --------------------------------------------------------------------------- #
# Per-game helpers
# --------------------------------------------------------------------------- #
def group_label(group):
    """A short human label for a monopoly group (a list of tiles)."""
    tile = group[0]
    if isinstance(tile, StreetProperty):
        return tile.color
    if isinstance(tile, Railroad):
        return "Railroads"
    return "Utilities"


def liquidation_net_worth(player):
    """Cash + property + houses at board value (mirrors ``evaluate.py``)."""
    total = float(player.balance)
    for prop in player.properties:
        total += prop.mortgage_value if prop.mortgaged else prop.price
        if isinstance(prop, StreetProperty):
            total += prop.houses * prop.house_cost()
    return total


def completed_groups(player, groups, exclude_ids=frozenset()):
    """Labels of the monopoly groups ``player`` currently fully owns, ignoring
    any tiles in ``exclude_ids`` (properties inherited from a bankrupted
    opponent) -- so a set only "counts" if it was completed through play."""
    return [group_label(g) for g in groups
            if all(t.owner is player and id(t) not in exclude_ids for t in g)]


def count_holdings(player, exclude_ids=frozenset()):
    """``(num_properties, num_houses, num_hotels)`` for ``player``, excluding
    tiles inherited from a bankrupted opponent (``exclude_ids``)."""
    houses = hotels = 0
    props = [p for p in player.properties if id(p) not in exclude_ids]
    for prop in props:
        if isinstance(prop, StreetProperty):
            if prop.houses >= 5:
                hotels += 1
            else:
                houses += prop.houses
    return len(props), houses, hotels


def play_one_game(env, policy, seed, max_turns):
    """Plays one full self-play game and returns a per-player result record.

    All seats are routed through ``policy``. The game runs to completion (a sole
    survivor) or until ``max_turns`` turns have been played (a timeout). During
    play we track, per seat, the monopolies ever completed, the turn the first
    was completed, and peak net worth / property count -- so a player's
    contribution is captured even though bankruptcy later wipes their assets.
    """
    # Build a fresh game with a seeded RNG, without touching the env's worker
    # thread: seed np_random, then let _build_game derive this game's dice/deck.
    env.np_random = np.random.default_rng(seed)
    env._build_game()
    g = env.game
    groups = env.encoder._groups
    n = env.num_players

    # Route every seat through the policy (synchronous deciders), and wire the
    # nested purchase / shortfall hooks the same way reset() would. The env now
    # keeps a per-seat opponent map (``_opponent_policies``) that
    # ``_policy_decide`` reads; point every seat at this policy.
    env._opponent_policies = {s: policy for s in range(n)}
    env._deciders = {s: env._make_policy_decider(s) for s in range(n)}
    for s, decide in env._deciders.items():
        g.players[s].decide_purchase = env._make_buy_hook(s, decide)
        g.players[s].decide_bid = env._make_bid_hook(s, decide)
    g.on_shortfall = env._on_shortfall

    # Auction analytics: for each auction that draws a bid, record how hard it
    # was contested and whether the winner completed their own set or blocked an
    # opponent who held the group's other tiles.
    auction = {"won": 0, "blocking": 0, "self_complete": 0,
               "self_ratios": [], "all_ratios": []}

    def on_auction_end(prop, winner, bid):
        if winner is None:
            return
        auction["won"] += 1
        ratio = bid / prop.price if prop.price else 0.0
        auction["all_ratios"].append(ratio)
        completer = next((p for p in g.players if not p.bankrupt
                          and env.encoder._completes_monopoly_for(p, prop)), None)
        if completer is winner:
            auction["self_complete"] += 1
            auction["self_ratios"].append(ratio)
        elif completer is not None:
            auction["blocking"] += 1  # took an opponent's last-missing tile

    g.on_auction_end = on_auction_end

    # Exclude properties seized by bankrupting an opponent from the holdings /
    # monopoly analytics: they were inherited, not won through play. Track the
    # tiles currently held as inheritance -- add a bankrupt player's whole estate
    # when a creditor inherits it, and clear a tile the moment it is next
    # acquired legitimately (bought, won at auction, or traded for).
    inherited_ids = set()

    def on_bankrupt(player, creditor, estate):
        if creditor is not None and not creditor.bankrupt:
            inherited_ids.update(id(p) for p in estate)   # passes to the creditor
        else:
            inherited_ids.difference_update(id(p) for p in estate)  # to the bank

    g.on_bankrupt = on_bankrupt
    # A legitimate (re)acquisition -- buy, auction win, or trade -- clears the
    # inherited flag. (Inheritance fires on_acquire with source="inherit", which
    # must *not* clear it.)
    def on_acquire(player, prop, source="trade"):
        if source != "inherit":
            inherited_ids.discard(id(prop))

    g.on_acquire = on_acquire

    # Trade analytics: every proposal actually offered to a partner, and whether
    # it was accepted. Reported by the env itself (``on_trade_offer``) rather
    # than reconstructed by monkey-patching its private trade builder -- an
    # offer of ``None`` means no sane deal existed, so nothing was proposed.
    trades = {"accepted": 0, "rejected": 0, "accepted_set_completing": 0}

    def on_trade_offer(initiator, partner, offer, accepted, completes):
        if offer is None:
            return  # nothing was put to the partner
        if accepted:
            trades["accepted"] += 1
            if completes:
                trades["accepted_set_completing"] += 1
        else:
            trades["rejected"] += 1

    env.on_trade_offer = on_trade_offer

    ever_monopolies = [set() for _ in range(n)]
    first_monopoly_turn = [None] * n
    first_monopoly_set = [None] * n
    peak_net_worth = [0.0] * n
    peak_properties = [0] * n

    def snapshot(turn):
        for s in range(n):
            p = g.players[s]
            if p.bankrupt:
                continue
            for label in completed_groups(p, groups, inherited_ids):
                if label not in ever_monopolies[s]:
                    ever_monopolies[s].add(label)
                    if first_monopoly_turn[s] is None:
                        first_monopoly_turn[s] = turn
                        first_monopoly_set[s] = label
            peak_net_worth[s] = max(peak_net_worth[s], liquidation_net_worth(p))
            peak_properties[s] = max(peak_properties[s], len(p.properties))

    turns = 0
    snapshot(0)
    while not g.is_over() and turns < max_turns:
        env._play_turn(g.players[g.current_player], env._deciders[g.current_player])
        turns += 1
        snapshot(turns)

    winner = g.winner()
    winner_seat = g.players.index(winner) if winner is not None else None

    players = []
    for s in range(n):
        p = g.players[s]
        props, houses, hotels = count_holdings(p, inherited_ids)
        players.append({
            "seat": s,
            "won": winner_seat == s,
            "bankrupt": p.bankrupt,
            "final_net_worth": liquidation_net_worth(p),
            "peak_net_worth": peak_net_worth[s],
            "final_properties": [pr.name for pr in p.properties
                                 if id(pr) not in inherited_ids],
            "final_monopolies": completed_groups(p, groups, inherited_ids),
            "ever_monopolies": sorted(ever_monopolies[s]),
            "num_ever_monopolies": len(ever_monopolies[s]),
            "first_monopoly_turn": first_monopoly_turn[s],
            "first_monopoly_set": first_monopoly_set[s],
            "properties": props,
            "houses": houses,
            "hotels": hotels,
        })

    return {
        "seat_winner": winner_seat,   # None on a timeout
        "length": turns,
        "timeout": winner_seat is None,
        "players": players,
        "auction": auction,
        "trades": trades,
    }


# --------------------------------------------------------------------------- #
# Aggregation across games
# --------------------------------------------------------------------------- #
def simulate(model_path, games=200, num_players=4, max_turns=1000, seed=0,
             deterministic=True, progress=True):
    """Runs ``games`` self-play games and returns aggregated analytics."""
    from sb3_contrib import MaskablePPO
    from training.selfplay import policy_from_model

    model = MaskablePPO.load(model_path)
    policy = policy_from_model(model, deterministic=deterministic)
    env = MonopolyEnv(seat=0, num_players=num_players, max_turns=max_turns)

    records = []
    for i in range(games):
        rec = play_one_game(env, policy, seed=seed + i, max_turns=max_turns)
        records.append(rec)
        if progress and (i + 1) % max(1, games // 20) == 0:
            done = sum(not r["timeout"] for r in records)
            print(f"  played {i + 1}/{games} games "
                  f"({done} decisive, {i + 1 - done} timeouts)", flush=True)
    env.close()

    return aggregate(records, num_players)


def aggregate(records, num_players):
    """Turns raw per-game records into summary statistics for report/plots."""
    games = len(records)
    decisive = [r for r in records if not r["timeout"]]
    timeouts = games - len(decisive)

    seat_wins = Counter()
    winner_monopoly_freq = Counter()   # sets held by the winner at game end
    winner_property_freq = Counter()   # tiles held by the winner at game end
    winner_net_worths = []
    winner_num_monopolies = []        # distinct sets the winner *ever* completed
    winner_final_num_monopolies = []  # sets the winner still holds at game end
    winner_first_monopoly_turns = []
    loser_num_monopolies = []
    loser_peak_net_worths = []

    # For each monopoly group: how many player-games ever completed it, and how
    # many of those went on to win -> P(win | ever completed the set).
    group_ever = Counter()
    group_ever_win = Counter()

    # For each group, restricted to the player-games where it was the *first*
    # monopoly completed: how many, and how many won -> P(win | first set was X).
    # A joint set-choice + tempo signal (which opening set best predicts a win).
    first_set_count = Counter()
    first_set_win = Counter()

    # Across decisive games in which anyone completed a monopoly: how many, and
    # in how many the player who completed the *first* monopoly of the game went
    # on to win -> % of games won by whoever got the first monopoly.
    first_monopolist_games = 0
    first_monopolist_wins = 0

    for r in records:
        for p in r["players"]:
            for label in p["ever_monopolies"]:
                group_ever[label] += 1
                if p["won"]:
                    group_ever_win[label] += 1
            fs = p["first_monopoly_set"]
            if fs is not None:
                first_set_count[fs] += 1
                if p["won"]:
                    first_set_win[fs] += 1
            if not p["won"]:
                loser_num_monopolies.append(p["num_ever_monopolies"])
                loser_peak_net_worths.append(p["peak_net_worth"])

        if r["timeout"]:
            continue
        w = next(p for p in r["players"] if p["won"])
        seat_wins[w["seat"]] += 1
        for label in w["final_monopolies"]:
            winner_monopoly_freq[label] += 1
        for name in w["final_properties"]:
            winner_property_freq[name] += 1
        winner_net_worths.append(w["final_net_worth"])
        winner_num_monopolies.append(w["num_ever_monopolies"])
        winner_final_num_monopolies.append(len(w["final_monopolies"]))
        if w["first_monopoly_turn"] is not None:
            winner_first_monopoly_turns.append(w["first_monopoly_turn"])

        # The player who completed the earliest monopoly this game; did they win?
        # (Ties on the same turn count as a win if the winner was among them.)
        first_turn = min((p["first_monopoly_turn"] for p in r["players"]
                          if p["first_monopoly_turn"] is not None), default=None)
        if first_turn is not None:
            first_monopolist_games += 1
            if w["first_monopoly_turn"] == first_turn:
                first_monopolist_wins += 1

    n_decisive = max(1, len(decisive))
    group_win_rate = {
        label: group_ever_win[label] / group_ever[label]
        for label in group_ever
    }
    first_set_win_rate = {
        label: first_set_win[label] / first_set_count[label]
        for label in first_set_count
    }

    # Auction aggression / blocking analytics (across all games).
    auc_won = sum(r["auction"]["won"] for r in records)
    auc_blocking = sum(r["auction"]["blocking"] for r in records)
    auc_self = sum(r["auction"]["self_complete"] for r in records)
    all_ratios = [x for r in records for x in r["auction"]["all_ratios"]]
    self_ratios = [x for r in records for x in r["auction"]["self_ratios"]]

    # Trade analytics (across all games): how many proposals were offered and how
    # many the partner accepted vs rejected.
    trades_accepted = sum(r["trades"]["accepted"] for r in records)
    trades_rejected = sum(r["trades"]["rejected"] for r in records)
    trades_setcomplete = sum(r["trades"]["accepted_set_completing"] for r in records)
    trades_offered = trades_accepted + trades_rejected

    return {
        "games": games,
        "decisive": len(decisive),
        "timeouts": timeouts,
        "num_players": num_players,
        "mean_length": float(np.mean([r["length"] for r in records])) if records else 0.0,
        "median_length": float(np.median([r["length"] for r in records])) if records else 0.0,
        "min_length": int(np.min([r["length"] for r in records])) if records else 0,
        "max_length": int(np.max([r["length"] for r in records])) if records else 0,
        "seat_wins": dict(seat_wins),
        "seat_win_rate": {s: seat_wins.get(s, 0) / n_decisive
                          for s in range(num_players)},
        "winner_monopoly_freq": winner_monopoly_freq,
        "winner_property_freq": winner_property_freq,
        "group_ever": group_ever,
        "group_win_rate": group_win_rate,
        "first_set_count": first_set_count,
        "first_set_win_rate": first_set_win_rate,
        "first_monopolist_games": first_monopolist_games,
        "first_monopolist_win_rate": (first_monopolist_wins / first_monopolist_games
                                      if first_monopolist_games else 0.0),
        "mean_winner_net_worth": float(np.mean(winner_net_worths)) if winner_net_worths else 0.0,
        "mean_winner_monopolies": float(np.mean(winner_num_monopolies)) if winner_num_monopolies else 0.0,
        "mean_winner_final_monopolies": (float(np.mean(winner_final_num_monopolies))
                                         if winner_final_num_monopolies else 0.0),
        "mean_loser_monopolies": float(np.mean(loser_num_monopolies)) if loser_num_monopolies else 0.0,
        "mean_winner_first_monopoly_turn": (float(np.mean(winner_first_monopoly_turns))
                                            if winner_first_monopoly_turns else None),
        "frac_winners_with_monopoly": (np.mean([m > 0 for m in winner_num_monopolies])
                                       if winner_num_monopolies else 0.0),
        "auctions_won": auc_won,
        "auctions_won_per_game": auc_won / max(1, games),
        "blocking_wins": auc_blocking,
        "blocking_wins_per_game": auc_blocking / max(1, games),
        "set_completion_wins": auc_self,
        "mean_bid_price_ratio": float(np.mean(all_ratios)) if all_ratios else 0.0,
        "mean_set_completion_bid_ratio": (float(np.mean(self_ratios))
                                          if self_ratios else 0.0),
        "trades_accepted": trades_accepted,
        "trades_rejected": trades_rejected,
        "trades_set_completing": trades_setcomplete,
        "accepted_trades_per_game": trades_accepted / max(1, games),
        "rejected_trades_per_game": trades_rejected / max(1, games),
        "set_completing_trades_per_game": trades_setcomplete / max(1, games),
        "trade_accept_rate": trades_accepted / max(1, trades_offered),
        "set_completing_share_of_accepted": trades_setcomplete / max(1, trades_accepted),
        "winner_net_worths": winner_net_worths,
        "winner_num_monopolies": winner_num_monopolies,
        "loser_num_monopolies": loser_num_monopolies,
        "winner_first_monopoly_turns": winner_first_monopoly_turns,
        "records": records,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(stats):
    g = stats["games"]
    print(f"\nSimulated {g} self-play games "
          f"({stats['num_players']} seats, all driven by the model)")
    print(f"  decisive games : {stats['decisive']}  |  timeouts: {stats['timeouts']}")
    print(f"  game length    : mean {stats['mean_length']:.0f}, "
          f"median {stats['median_length']:.0f} turns "
          f"(range {stats['min_length']}-{stats['max_length']})")

    print("\nSeat win rate (first player is seat 0):")
    for s in range(stats["num_players"]):
        wr = stats["seat_win_rate"][s]
        wins = stats["seat_wins"].get(s, 0)
        bar = "#" * int(round(wr * 40))
        print(f"  seat {s}: {wr * 100:5.1f}%  ({wins:>3})  {bar}")

    print("\nWhy winners win:")
    print(f"  monopolies ever completed   : winner {stats['mean_winner_monopolies']:.2f}  "
          f"vs loser {stats['mean_loser_monopolies']:.2f}  "
          f"(cumulative; a set counts even if later traded away or re-completed "
          f"by another player)")
    print(f"  monopolies held at game end : winner "
          f"{stats['mean_winner_final_monopolies']:.2f}  "
          f"(what the winner still owns when the game ends)")
    print(f"  winners holding >=1 monopoly : "
          f"{stats['frac_winners_with_monopoly'] * 100:.1f}%")
    if stats["mean_winner_first_monopoly_turn"] is not None:
        print(f"  winner's first monopoly     : turn "
              f"{stats['mean_winner_first_monopoly_turn']:.0f} (mean)")
    print(f"  first monopolist wins       : "
          f"{stats['first_monopolist_win_rate'] * 100:.1f}%  "
          f"(of {stats['first_monopolist_games']} games where a set was completed)")
    print(f"  winner net worth            : ${stats['mean_winner_net_worth']:,.0f} (mean)")

    print("\nAuction play (aggression & blocking):")
    print(f"  auctions won            : {stats['auctions_won']} "
          f"({stats['auctions_won_per_game']:.2f}/game)")
    print(f"  blocking wins           : {stats['blocking_wins']} "
          f"({stats['blocking_wins_per_game']:.2f}/game "
          f"-- opponents' sets denied)")
    print(f"  own-set-completion wins : {stats['set_completion_wins']}")
    print(f"  mean winning bid / price: {stats['mean_bid_price_ratio']:.2f}  "
          f"(set-completers {stats['mean_set_completion_bid_ratio']:.2f})")

    print("\nTrades (proposals offered to a partner):")
    print(f"  accepted : {stats['accepted_trades_per_game']:.2f}/game "
          f"({stats['trades_accepted']} total)")
    print(f"  rejected : {stats['rejected_trades_per_game']:.2f}/game "
          f"({stats['trades_rejected']} total)")
    print(f"  accept rate : {stats['trade_accept_rate'] * 100:.1f}%")
    print(f"  set-completing (accepted) : "
          f"{stats['set_completing_trades_per_game']:.2f}/game "
          f"({stats['trades_set_completing']} total, "
          f"{stats['set_completing_share_of_accepted'] * 100:.1f}% of accepted)")

    print("\nSets that most predict a win  (P(win | player ever completed the set)):")
    ranked = sorted(stats["group_win_rate"].items(),
                    key=lambda kv: kv[1], reverse=True)
    for label, wr in ranked:
        n = stats["group_ever"][label]
        bar = "#" * int(round(wr * 40))
        print(f"  {label:<12} {wr * 100:5.1f}%  (n={n:>4})  {bar}")

    print("\nFirst sets that most predict a win  "
          "(P(win | player's first completed monopoly was this set)):")
    ranked = sorted(stats["first_set_win_rate"].items(),
                    key=lambda kv: kv[1], reverse=True)
    for label, wr in ranked:
        n = stats["first_set_count"][label]
        bar = "#" * int(round(wr * 40))
        print(f"  {label:<12} {wr * 100:5.1f}%  (n={n:>4})  {bar}")

    print("\nMonopolies most often held by the winner at game end:")
    for label, cnt in stats["winner_monopoly_freq"].most_common():
        share = cnt / max(1, stats["decisive"])
        print(f"  {label:<12} {share * 100:5.1f}%  ({cnt})")

    print("\nTop properties held by the winner at game end:")
    for name, cnt in stats["winner_property_freq"].most_common(10):
        share = cnt / max(1, stats["decisive"])
        print(f"  {name:<22} {share * 100:5.1f}%  ({cnt})")
    print()


def plot_dashboard(stats, path, title=""):
    """Renders a multi-panel analytics dashboard to ``path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(title or "Monopoly self-play strategy analysis",
                 fontsize=15, fontweight="bold")

    n = stats["num_players"]

    # 1. Seat win rate.
    ax = axes[0, 0]
    seats = list(range(n))
    rates = [stats["seat_win_rate"][s] * 100 for s in seats]
    bars = ax.bar([f"seat {s}" for s in seats], rates, color="#4c72b0")
    ax.axhline(100 / n, color="black", linestyle="--", linewidth=1,
               label=f"even ({100 / n:.0f}%)")
    ax.set_title("Win rate by seat (turn order)")
    ax.set_ylabel("% of decisive games")
    ax.legend()
    for b, v in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom")

    # 2. P(win | ever completed set), sorted.
    ax = axes[0, 1]
    ranked = sorted(stats["group_win_rate"].items(), key=lambda kv: kv[1])
    if ranked:
        labels = [k for k, _ in ranked]
        vals = [v * 100 for _, v in ranked]
        ax.barh(labels, vals, color="#55a868")
        ax.axvline(100 / n, color="black", linestyle="--", linewidth=1)
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.0f}%", va="center")
    ax.set_title("Win rate given the set was completed")
    ax.set_xlabel("P(win | ever completed set)  %")

    # 3. Monopolies held by winners at game end.
    ax = axes[0, 2]
    common = stats["winner_monopoly_freq"].most_common()
    if common:
        labels = [k for k, _ in common][::-1]
        vals = [c / max(1, stats["decisive"]) * 100 for _, c in common][::-1]
        ax.barh(labels, vals, color="#dd8452")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.0f}%", va="center")
    ax.set_title("Sets the winner held at game end")
    ax.set_xlabel("% of decisive games")

    # 4. Winner vs loser monopoly count.
    ax = axes[1, 0]
    wm = stats["winner_num_monopolies"]
    lm = stats["loser_num_monopolies"]
    maxm = max([0] + wm + lm)
    bins = np.arange(-0.5, maxm + 1.5, 1)
    ax.hist([wm, lm], bins=bins, density=True,
            color=["#2e8b57", "#c0392b"], label=["winners", "losers"])
    ax.set_title(f"Monopolies completed\n(winner {stats['mean_winner_monopolies']:.2f} "
                 f"vs loser {stats['mean_loser_monopolies']:.2f} mean)")
    ax.set_xlabel("# monopolies ever completed")
    ax.set_ylabel("density")
    ax.legend()

    # 5. Winner's first-monopoly timing.
    ax = axes[1, 1]
    fmt = stats["winner_first_monopoly_turns"]
    if fmt:
        ax.hist(fmt, bins=25, color="#8172b3", edgecolor="white")
        ax.axvline(float(np.mean(fmt)), color="black", linestyle="--",
                   label=f"mean turn {np.mean(fmt):.0f}")
        ax.legend()
    ax.set_title("When the winner completed their first set")
    ax.set_xlabel("turn of first monopoly")
    ax.set_ylabel("games")

    # 6. Top properties held by winners.
    ax = axes[1, 2]
    top = stats["winner_property_freq"].most_common(10)
    if top:
        labels = [k for k, _ in top][::-1]
        vals = [c / max(1, stats["decisive"]) * 100 for _, c in top][::-1]
        ax.barh(labels, vals, color="#937860")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.0f}%", va="center")
    ax.set_title("Properties most held by the winner")
    ax.set_xlabel("% of decisive games")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_csv(records, path):
    """Writes one row per player per game (long format)."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "length", "timeout", "seat", "won", "bankrupt",
                    "final_net_worth", "peak_net_worth", "num_ever_monopolies",
                    "first_monopoly_turn", "first_monopoly_set", "properties",
                    "houses", "hotels", "final_monopolies"])
        for gi, r in enumerate(records):
            for p in r["players"]:
                w.writerow([
                    gi, r["length"], int(r["timeout"]), p["seat"],
                    int(p["won"]), int(p["bankrupt"]),
                    f"{p['final_net_worth']:.0f}", f"{p['peak_net_worth']:.0f}",
                    p["num_ever_monopolies"], p["first_monopoly_turn"],
                    p["first_monopoly_set"] or "", p["properties"],
                    p["houses"], p["hotels"],
                    "|".join(p["final_monopolies"]),
                ])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                        help=f"path to a saved MaskablePPO model "
                             f"(default: {DEFAULT_MODEL})")
    parser.add_argument("--games", type=int, default=200,
                        help="number of self-play games to simulate")
    parser.add_argument("--players", type=int, default=4,
                        help="number of seats (all driven by the model)")
    parser.add_argument("--max-turns", type=int, default=1000,
                        help="turn cap before a game is called a timeout")
    parser.add_argument("--seed", type=int, default=0,
                        help="base RNG seed (game i uses seed + i)")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of the argmax (more varied "
                             "play; default is the model's greedy strategy)")
    parser.add_argument("--plot", nargs="?", const="auto", default=None,
                        help="render a dashboard PNG (optional path; default: "
                             "runs/simulate_<model>.png)")
    parser.add_argument("--csv", default=None,
                        help="also write per-player-per-game records to this CSV")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    stats = simulate(args.model, games=args.games, num_players=args.players,
                     max_turns=args.max_turns, seed=args.seed,
                     deterministic=not args.stochastic)
    print_report(stats)

    if args.csv:
        write_csv(stats["records"], args.csv)
        print(f"wrote per-game CSV -> {args.csv}")

    if args.plot is not None:
        if args.plot == "auto":
            base = os.path.splitext(os.path.basename(args.model))[0]
            os.makedirs("runs", exist_ok=True)
            plot_path = os.path.join("runs", f"simulate_{base}.png")
        else:
            plot_path = args.plot
        title = f"Self-play analysis: {os.path.basename(args.model)}  ({args.games} games)"
        plot_dashboard(stats, plot_path, title=title)
        print(f"wrote dashboard -> {plot_path}")


if __name__ == "__main__":
    main()
