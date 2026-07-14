"""Estimate how often each board tile is landed on, across many simulated games.

Landing frequency in Monopoly is a property of the *movement* rules -- the dice
distribution, the Jail mechanics, and the "advance to ..." Chance / Community
Chest cards -- and is essentially independent of which properties players buy or
build. This script simulates many games of pure movement and counts a **visit**
every time any player's turn resolves onto a tile (i.e. every ``resolve_tile``
call: a normal dice landing, *and* the destination a card teleports them to).

To keep every seat moving for the whole game (so the frequencies are not skewed
by early bankruptcies), players decline all purchases and are given an unlimited
bankroll -- neither choice affects where the dice send them.

The result -- per tile: total visits, average visits per game, and the share of
all landings -- is written to ``data/board_visits.json`` and used by the RL
observation (``engine.observation.load_landing_frequencies``) so the agent can
value a high-traffic property above a quiet one of the same price. Regenerating
it changes what every model sees: it is part of the observation definition, which
is why it is tracked alongside the board rather than left in ``runs/``.

Usage:
    PYTHONPATH=. python -m validation.board_visits             # 2000 games
    PYTHONPATH=. python -m validation.board_visits --games 5000 --turns 250
    PYTHONPATH=. python -m validation.board_visits --jail-strategy pay
"""

import argparse
import json
import os
import random

import numpy as np

from engine.game import Game
from models.board import Board
from models.player import Player
from models.tiles.properties.street_property import StreetProperty
from models.tiles.properties.railroad import Railroad
from models.tiles.properties.utility import Utility
from data.board_tiles import build_board_tiles
from data.decks import build_chance_deck, build_community_deck


# Tracked static data, not a run artifact: this table scales every traffic-based
# observation feature and valuation, so it is part of the observation definition
# and must travel with the code (see engine/observation.load_landing_frequencies).
DEFAULT_OUT = os.path.join("data", "board_visits.json")
_UNLIMITED = 10 ** 9  # bankroll so tax / card charges never bankrupt a mover


def _tile_type(tile):
    """Short category label for a tile (for the saved table / report)."""
    if isinstance(tile, StreetProperty):
        return "street"
    if isinstance(tile, Railroad):
        return "railroad"
    if isinstance(tile, Utility):
        return "utility"
    return type(tile).__name__


def _build_movement_game(names):
    """A fresh game whose players only ever move: they decline every purchase
    and hold an unlimited bankroll, so no one leaves the board mid-game."""
    players = [Player(name) for name in names]
    for p in players:
        p.balance = _UNLIMITED
        p.decide_purchase = lambda prop: False  # never buy -> no rent, no bankruptcy
    board = Board(build_board_tiles())
    game = Game(players, board, build_chance_deck(), build_community_deck())
    random.shuffle(game.chance_deck.cards)
    random.shuffle(game.community_deck.cards)
    return game


def _count_game(game, turns, jail_strategy, visits):
    """Plays ``turns`` movement turns, adding one visit per tile resolution."""
    orig_resolve = game.resolve_tile

    def counting_resolve(player):
        # ``player.pos`` is the tile being resolved -- a dice landing or the
        # square a Chance / Community Chest card advanced the player onto.
        visits[player.pos] += 1
        return orig_resolve(player)

    game.resolve_tile = counting_resolve
    for _ in range(turns):
        game.step(jail_choice=jail_strategy)


def simulate(games=2000, turns=200, players=4, jail_strategy="roll", seed=0,
             progress=True):
    """Runs ``games`` movement games and returns per-tile visit statistics."""
    random.seed(seed)
    names = ["Red", "Blue", "Green", "Yellow"][:players]
    if players > 4:
        names = [f"P{i}" for i in range(players)]

    tiles = build_board_tiles()
    n_tiles = len(tiles)
    totals = np.zeros(n_tiles, dtype=np.int64)

    for i in range(games):
        game = _build_movement_game(names)
        per_game = np.zeros(n_tiles, dtype=np.int64)
        _count_game(game, turns, jail_strategy, per_game)
        totals += per_game
        if progress and (i + 1) % max(1, games // 20) == 0:
            print(f"  simulated {i + 1}/{games} games", flush=True)

    grand_total = int(totals.sum())
    table = []
    for tile in tiles:
        v = int(totals[tile.pos])
        table.append({
            "pos": tile.pos,
            "name": tile.name,
            "type": _tile_type(tile),
            "price": getattr(tile, "price", None),
            "color": getattr(tile, "color", None),
            "total_visits": v,
            "avg_visits_per_game": v / games,
            "frequency": v / grand_total if grand_total else 0.0,
        })

    return {
        "meta": {
            "games": games,
            "turns_per_game": turns,
            "players": players,
            "jail_strategy": jail_strategy,
            "seed": seed,
            "board_size": n_tiles,
            "total_landings": grand_total,
        },
        "tiles": table,
    }


def save(result, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def print_report(result):
    meta = result["meta"]
    print(f"\n{meta['total_landings']:,} landings over {meta['games']} games "
          f"({meta['turns_per_game']} turns, {meta['players']} players, "
          f"jail={meta['jail_strategy']})")
    uniform = 1.0 / meta["board_size"]
    print(f"\n{'tile':<24}{'type':<10}{'visits':>10}{'/game':>9}"
          f"{'share':>8}{'  vs even':>10}")
    print("-" * 71)
    for t in sorted(result["tiles"], key=lambda t: t["frequency"], reverse=True):
        mult = t["frequency"] / uniform if uniform else 0.0
        print(f"{t['name']:<24}{t['type']:<10}{t['total_visits']:>10,}"
              f"{t['avg_visits_per_game']:>9.1f}{t['frequency'] * 100:>7.2f}%"
              f"{mult:>9.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=2000,
                        help="number of movement games to simulate")
    parser.add_argument("--turns", type=int, default=200,
                        help="turns (across all seats) played per game")
    parser.add_argument("--players", type=int, default=4,
                        help="number of players moving on the board")
    parser.add_argument("--jail-strategy", choices=["roll", "pay"], default="roll",
                        help="how jailed players leave: roll for doubles (pure "
                             "mechanic) or pay the fine immediately (shorter jail)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"where to write the JSON table (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    result = simulate(games=args.games, turns=args.turns, players=args.players,
                      jail_strategy=args.jail_strategy, seed=args.seed)
    print_report(result)
    path = save(result, args.out)
    print(f"\nwrote visit table -> {path}")


if __name__ == "__main__":
    main()
