"""Shared fixtures: a live game + a bound encoder, without the RL worker thread.

``MonopolyEnv`` runs the engine on a background thread and only yields at
decision points, which makes it awkward to assert on a *specific* board state.
These fixtures build the same game the env builds (same board, same ownable
order, so tile indices match the action space) and bind an ``ObsEncoder`` to it
directly, so a test can hand-place properties and inspect the result.
"""

import random

import pytest

from data.board_tiles import build_board_tiles
from data.decks import build_chance_deck, build_community_deck
from engine.game import Game
from engine.observation import ObsEncoder
from models.board import Board
from models.player import Player
from models.tiles.properties.railroad import Railroad
from models.tiles.properties.street_property import StreetProperty
from models.tiles.properties.utility import Utility


@pytest.fixture(autouse=True)
def _seeded_rng():
    """The engine draws dice and shuffles decks from the *global* ``random``
    module (see engine/game.py), so pin it for reproducible tests."""
    random.seed(1234)


@pytest.fixture
def game():
    """A fresh 4-player game on the standard board."""
    players = [Player(n) for n in ("Red", "Blue", "Green", "Yellow")]
    board = Board(build_board_tiles())
    return Game(players, board, build_chance_deck(), build_community_deck())


@pytest.fixture
def ownable(game):
    """The 28 ownable tiles in board order -- the order that fixes every tile's
    index in the observation and the action space."""
    return [t for t in game.board.tiles
            if isinstance(t, (StreetProperty, Railroad, Utility))]


@pytest.fixture
def encoder(game, ownable):
    return ObsEncoder().bind(game, ownable)


def give(player, *tiles):
    """Hands ``tiles`` to ``player`` (bypassing purchase, for test setup)."""
    for tile in tiles:
        tile.owner = player
        player.properties.append(tile)


def group_named(encoder, color):
    """The monopoly group whose streets have colour ``color``."""
    for grp in encoder._groups:
        if isinstance(grp[0], StreetProperty) and grp[0].color == color:
            return grp
    raise AssertionError(f"no street group coloured {color!r}")
