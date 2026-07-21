"""Train/play parity for trades.

The project's core guarantee is that the agent you train is the agent you play
against. Trades used to be where that guarantee was weakest: two offer builders
(``rl_env._attempt_trade`` and ``ui/ai_player._attempt_trade``, the latter
carrying the comment *"Kept byte-identical to the env path"*) and three accept
rules, held together by comment discipline -- and they had already drifted. Worse,
the GUI put one-for-one offers to the *model* while a human stacking tiles fell
through to the formula, so the accept rate training measured was not the one a
human faced.

There is now one :class:`~engine.trade.TradeEngine` and one
:meth:`ObsEncoder.accepts`, so parity is structural rather than maintained. These
tests are the guard against a second path quietly growing back.
"""

import numpy as np
import pytest

from engine.rl_env import MonopolyEnv
from tests.conftest import give, group_named
from ui.ai_player import GUIAIDecider


@pytest.fixture
def env():
    """A live env, built without starting the worker thread (as simulate.py does)."""
    env = MonopolyEnv(seat=0, num_players=4, seed=0)
    env.np_random = np.random.default_rng(0)
    env._build_game()
    env._deciders = {}
    env._opponent_policies = {}
    return env


def _one_away(env):
    """Red holds 2/3 of orange and the cash to buy the third from Blue."""
    enc = env.encoder
    red, blue = env.game.players[0], env.game.players[1]
    orange = group_named(enc, "orange")
    give(red, orange[0], orange[1], group_named(enc, "brown")[0])
    give(blue, orange[2])
    red.balance = 5000
    return red, blue, orange[2]


def test_the_env_and_the_gui_build_the_same_offer(env):
    """Same board, same seats, same offer -- tile for tile and dollar for dollar."""
    red, blue, _ = _one_away(env)

    gui = GUIAIDecider(env.num_players, model=None)
    gui.bind(env.game, env.ownable)

    env_offer = env.trades.best_offer(red, blue)
    gui_offer = gui.trades.best_offer(red, blue)

    assert env_offer is not None, "the training path proposed nothing"
    assert gui_offer is not None, "the GUI path proposed nothing"
    assert [t.name for t in env_offer.give] == [t.name for t in gui_offer.give]
    assert [t.name for t in env_offer.receive] == [t.name for t in gui_offer.receive]
    assert env_offer.cash == gui_offer.cash


def test_the_gui_answers_an_offer_with_the_shared_rule(env):
    """``GUIAIDecider.evaluate_trade`` *is* ``ObsEncoder.accepts``.

    It used to route one-for-one offers to the model instead, which accepted ~85%
    of everything -- so a human could hand the AI junk for a monopoly. A model is
    attached here precisely to prove it no longer gets a vote.
    """
    red, blue, target = _one_away(env)

    gui = GUIAIDecider(env.num_players, model=None)
    gui.bind(env.game, env.ownable)

    class _AlwaysAccepts:
        def predict(self, *a, **k):
            raise AssertionError("the model must not decide trades")

    gui.model = _AlwaysAccepts()

    brown = group_named(env.encoder, "brown")[0]
    gui_verdict, gui_value = gui.evaluate_trade(blue, red, [brown], [target], 50)
    enc_verdict, enc_value = env.encoder.accepts(blue, [brown], [target], 50)
    assert (gui_verdict, gui_value) == (enc_verdict, enc_value)


def test_a_human_stacking_tiles_is_priced_the_same_way(env):
    """The offer shape that used to fall through to a *different* rule. Any number
    of tiles either way, priced by the one valuation."""
    enc = env.encoder
    red, blue = env.game.players[0], env.game.players[1]
    orange = group_named(enc, "orange")
    browns = group_named(enc, "brown")
    lights = group_named(enc, "light_blue")

    give(red, orange[0], orange[1])
    give(blue, *browns, *lights)

    gui = GUIAIDecider(env.num_players, model=None)
    gui.bind(env.game, env.ownable)
    gui.model = object()   # must be ignored

    stacked = list(browns) + list(lights)
    assert gui.evaluate_trade(red, blue, stacked, [orange[0]], 0) == \
        enc.accepts(red, stacked, [orange[0]], 0)


def test_the_junk_for_monopoly_trade_is_refused_in_the_gui(env):
    """End to end, on the path the bug was actually reported on: a human offering
    a brown and $200 for the orange tile the AI is one short of."""
    red, blue = env.game.players[0], env.game.players[1]
    orange = group_named(env.encoder, "orange")
    brown = group_named(env.encoder, "brown")[0]
    give(red, orange[0], orange[1])
    give(blue, brown)

    gui = GUIAIDecider(env.num_players, model=None)
    gui.bind(env.game, env.ownable)
    gui.model = object()

    accepted, _ = gui.evaluate_trade(red, blue, [brown], [orange[0]], 200)
    assert not accepted
