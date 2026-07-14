"""Train/play parity for trade construction.

The project's core guarantee is that the agent you train is the agent you play
against: one ``ObsEncoder`` computes the observation, the masks, and every
valuation for both the training env and the GUI. Trades are the one place that
guarantee is maintained by hand -- ``engine/rl_env.py::_attempt_trade`` and
``ui/ai_player.py::_attempt_trade`` are two separate implementations, the latter
carrying the comment *"Kept byte-identical to the env path"*.

These tests hold the two to that claim: given the same board, the same seats and
the same cash tier, both must construct the *same offer* -- the same tile handed
over and the same cash. If they don't, the policy is trained against economics it
will never meet in the GUI.
"""

import numpy as np
import pytest

from engine.constants import A_TRADE_ACCEPT
from engine.rl_env import MonopolyEnv
from tests.conftest import give, group_named
from ui.ai_player import GUIAIDecider


@pytest.fixture
def env():
    """A live env, built without starting the worker thread (as simulate.py does)."""
    env = MonopolyEnv(seat=0, num_players=4, seed=0)
    env.np_random = np.random.default_rng(0)
    env._build_game()
    env._deciders = {}           # no policy seats: the partner uses the formula
    env._opponent_policies = {}  # ...and seat 0 counts as a learned policy
    return env


def _capture(game, run):
    """Runs ``run()`` with acceptance forced and ``execute_trade`` stubbed, and
    returns the offer that was constructed as ``(give_tile, cash)`` -- or None if
    the path never got as far as proposing one."""
    captured = []
    real_execute = game.execute_trade

    def spy(initiator, partner, give_tiles, receive, cash):
        captured.append((give_tiles[0], cash))
        return True

    game.execute_trade = spy
    try:
        run()
    finally:
        game.execute_trade = real_execute
    return captured[0] if captured else None


def _offers(env, seat, target, tier):
    """The offer each of the two code paths builds for the same situation.

    Both partners are forced to accept -- via the partner's *decider* in the env
    and via the app's *arbiter* in the GUI -- so acceptance never masks a
    difference in the offer itself, and neither stub perturbs the offer builder.
    """
    game, initiator = env.game, env.game.players[seat]
    partner_seat = game.players.index(target.owner)

    # -- training path: MonopolyEnv._attempt_trade
    env._deciders = {partner_seat: lambda phase, prop=None, amount=0: A_TRADE_ACCEPT}
    env_offer = _capture(game, lambda: env._attempt_trade(initiator, target, tier))

    # -- play path: GUIAIDecider._attempt_trade, on the same live game
    gui = GUIAIDecider(env.num_players, model=None)
    gui.bind(game, env.ownable)
    gui.trade_arbiter = lambda *a, **k: True
    gui_offer = _capture(game, lambda: gui._attempt_trade(initiator, target, tier))

    return env_offer, gui_offer


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_set_completing_offer_matches(env, tier):
    """A rich agent buying the tile that completes its own monopoly -- the
    'overpay from surplus' path both sides implement."""
    enc = env.encoder
    red, blue = env.game.players[0], env.game.players[1]
    orange = group_named(enc, "orange")
    brown = group_named(enc, "brown")

    give(red, orange[0], orange[1], brown[0])   # brown[0] is the spare to hand over
    give(blue, orange[2])                       # the contested set-completer
    red.balance = 5000

    assert enc._completes_monopoly_for(red, orange[2])

    env_offer, gui_offer = _offers(env, 0, orange[2], tier)

    assert env_offer is not None, "the training path proposed nothing"
    assert gui_offer is not None, "the GUI path proposed nothing"
    assert env_offer == gui_offer


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_denial_offer_matches(env, tier):
    """Prying a tile off an opponent who is cornering a set. This is the path
    where the two implementations diverge: the GUI caps the cash at its own
    break-even, the env has no such cap."""
    enc = env.encoder
    red, blue = env.game.players[0], env.game.players[1]
    orange = group_named(enc, "orange")
    brown = group_named(enc, "brown")

    give(red, orange[0], brown[0])
    give(blue, orange[1], orange[2])            # Blue is one tile off the set
    red.balance = 5000

    assert not enc._completes_monopoly_for(red, orange[1])
    assert enc._denies_monopoly(red, blue, orange[1])

    env_offer, gui_offer = _offers(env, 0, orange[1], tier)

    assert env_offer is not None, "the training path proposed nothing"
    assert gui_offer is not None, "the GUI path proposed nothing"
    assert env_offer == gui_offer


def test_a_poor_agent_keeps_its_rent_cushion(env):
    """The surplus-overpay rule exists to stop the agent buying a set and going
    broke: it may spend only the cash above its rent-sized reserve."""
    enc = env.encoder
    red, blue = env.game.players[0], env.game.players[1]
    orange = group_named(enc, "orange")
    brown = group_named(enc, "brown")

    give(red, orange[0], orange[1], brown[0])
    give(blue, orange[2])
    # Give Blue enough developed board presence that Red's reserve is real.
    for tile in group_named(enc, "red"):
        give(blue, tile)
    red.balance = 800

    reserve = enc._cash_reserve(red)
    assert reserve > 0, "expected Blue's holdings to create rent exposure"

    env_offer, gui_offer = _offers(env, 0, orange[2], tier=2)

    assert env_offer == gui_offer
    if env_offer is not None:
        _, cash = env_offer
        assert cash <= max(0, int(red.balance - reserve))
