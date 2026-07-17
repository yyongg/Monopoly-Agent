"""The learning signal.

Each test here pins one property the reward *must* have. They are cheap to run
and they guard changes that are otherwise invisible: a reward bug does not crash,
it just quietly trains the wrong policy.
"""

import numpy as np
import pytest

from engine.config import RewardConfig
from engine.rl_env import MonopolyEnv
from models.tiles.properties.utility import Utility
from tests.conftest import give, group_named


@pytest.fixture
def env():
    """A live env with a built game, no worker thread."""
    env = MonopolyEnv(seat=0, num_players=4, seed=0, max_turns=1000, gamma=0.99)
    env.np_random = np.random.default_rng(0)
    env._build_game()
    env._prev_potential = 0.0
    env._pending_bonus = 0.0
    env._penalty_turn = -1
    return env


def _env_with_strength_weight(w):
    env = MonopolyEnv(seat=0, num_players=4, seed=0, max_turns=1000, gamma=0.99,
                      cfg=RewardConfig(set_strength_reward_weight=w))
    env.np_random = np.random.default_rng(0)
    env._build_game()
    return env


class TestPotentialContinuity:
    """The shaping potential is the agent's net worth relative to the *mean*
    opponent. That mean used to be taken over the survivors only, so eliminating
    a player moved it -- and knocking out the poorest opponent *raised* the mean
    of those left, lowering the agent's advantage and paying it a **negative**
    reward for a strictly good outcome."""

    def test_bankrupting_a_worthless_opponent_changes_nothing(self, env):
        game = env.game
        red, yellow = game.players[0], game.players[3]
        give(red, *group_named(env.encoder, "orange"))
        yellow.balance = 0            # nothing to their name already, so their
        assert not yellow.properties  # net worth is 0 before *and* after

        before = env._potential()
        game.declare_bankrupt(yellow)
        after = env._potential()

        assert after == pytest.approx(before)

    def test_eliminating_an_opponent_is_never_punished(self, env):
        game = env.game
        red, blue, yellow = game.players[0], game.players[1], game.players[3]
        give(red, *group_named(env.encoder, "orange"))
        give(blue, *group_named(env.encoder, "green"))   # a rich rival
        give(yellow, group_named(env.encoder, "brown")[0])
        yellow.balance = 40                               # the poorest opponent

        before = env._potential()
        game.declare_bankrupt(yellow)     # their property goes back to the bank
        after = env._potential()

        assert after >= before, (
            "removing the weakest opponent must not lower the agent's advantage")


class TestTerminalPotential:
    """``phi`` is zeroed at a real terminal so the shaping telescopes away and
    the +-1 win/loss is what survives. A *truncated* episode stops at an ordinary
    state, whose value the learner bootstraps -- zeroing it there (or paying a
    made-up final reward) would count the rest of the game twice."""

    def _rich_agent_alone(self, env):
        game = env.game
        red = game.players[0]
        give(red, *group_named(env.encoder, "green"))
        red.balance = 8000
        for other in game.players[1:]:
            game.declare_bankrupt(other)
        return red

    def test_a_real_ending_pays_the_win_and_nothing_else(self, env):
        self._rich_agent_alone(env)
        env._prev_potential = 0.0

        reward = env._reward(terminal=True)

        # phi(terminal) == 0, so no net-worth spike rides along with the win.
        assert reward == pytest.approx(env._terminal_reward())
        assert reward == pytest.approx(1.0)      # sole survivor

    def test_a_truncated_ending_pays_no_made_up_final_reward(self, env):
        self._rich_agent_alone(env)
        env._prev_potential = 0.0

        reward = env._reward(terminal=True, truncated=True)

        # The state is ordinary: its potential is real, and no terminal reward is
        # added on top of the value the learner will bootstrap.
        assert reward == pytest.approx(env.gamma * env._potential())
        assert reward != pytest.approx(env._terminal_reward())


class TestSolvencyPenalty:
    """Charged once per *turn*. Charging it per decision made it a tax the agent
    could dodge by taking fewer actions -- a gradient toward passivity, exactly
    when a cash-poor agent most needs to act."""

    def test_charged_once_per_turn_however_many_decisions(self, env):
        game = env.game
        red, blue = game.players[0], game.players[1]
        give(blue, *group_named(env.encoder, "red"))   # rent threat for Red
        red.balance = 0                                # maximum deficit

        penalty = env._solvency_penalty(red)
        assert penalty > 0, "expected a live rent threat"

        env._prev_potential = env._potential()
        first = env._reward(terminal=False)   # first decision of the turn
        again = env._reward(terminal=False)   # a second decision, same turn

        # The two differ by exactly the penalty: it rode along with the first
        # decision and was not charged again for the second.
        assert first - again == pytest.approx(-penalty)

        game.advance_turn()                   # a new turn: charged once more
        after_turn = env._reward(terminal=False)
        assert after_turn - again == pytest.approx(-penalty)


class TestSetStrengthReward:
    """The reward prizes a strong monopoly above a weak one of equal sticker
    price, so losing (or giving away) a money set costs more shaped reward than
    trading a cheap one -- the reward-side lever behind smarter trading. ``w=0``
    recovers the old flat behaviour exactly."""

    def _util_group(self, env):
        return next(g for g in env.encoder._groups
                    if isinstance(g[0], Utility))

    def test_flat_weight_recovers_unweighted_value(self):
        env = _env_with_strength_weight(0.0)
        red = env.game.players[0]
        orange = group_named(env.encoder, "orange")
        give(red, *orange)
        # w=0 -> every set weighted 1.0, so the owned-monopoly value is just the
        # group's list price and the effective strength is 1.
        assert env._effective_set_strength(orange) == pytest.approx(1.0)
        assert env._owned_monopoly_value(red) == pytest.approx(
            sum(t.price for t in orange))

    def test_a_strong_set_is_worth_more_than_flat(self):
        env = _env_with_strength_weight(1.0)
        red = env.game.players[0]
        orange = group_named(env.encoder, "orange")
        give(red, *orange)
        s = env.encoder._set_strength(orange)
        assert s > 1.0
        assert env._owned_monopoly_value(red) == pytest.approx(
            sum(t.price for t in orange) * s)

    def test_a_weak_set_is_worth_less_than_flat(self):
        env = _env_with_strength_weight(1.0)
        red = env.game.players[0]
        util = self._util_group(env)
        give(red, *util)
        s = env.encoder._set_strength(util)
        assert s < 1.0
        assert env._owned_monopoly_value(red) == pytest.approx(
            sum(t.price for t in util) * s)
