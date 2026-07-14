"""Legal-action masks.

The env clamps an out-of-mask action to ``safe_default(phase)`` instead of
raising (engine/rl_env.py ``_safe_default``), and on the *opponent* path it does
so with no record at all -- so a systematically mis-masked phase would look like
a passive policy rather than a bug. These tests pin the mask invariants that
clamping would otherwise hide.
"""

import pytest

from engine.constants import (
    PHASE_JAIL, PHASE_BUY, PHASE_MANAGE, PHASE_LIQUIDATE,
    PHASE_AUCTION, PHASE_TRADE_RESPOND,
    A_PAY_JAIL, A_USE_CARD, A_ROLL_JAIL, A_BUY, A_DECLINE, A_END_MANAGE,
    A_BUILD, A_MORTGAGE, A_TRADE, A_AUCTION_BID,
    NUM_OWNABLE, NUM_TRADE_TIERS, BID_FRACTIONS, decode_trade_action,
)
from engine.observation import safe_default
from tests.conftest import give, group_named

ALL_PHASES = [PHASE_JAIL, PHASE_BUY, PHASE_MANAGE, PHASE_LIQUIDATE,
              PHASE_AUCTION, PHASE_TRADE_RESPOND]


@pytest.mark.parametrize("phase", ALL_PHASES)
def test_mask_is_never_empty(encoder, ownable, phase):
    """An all-zero mask would make MaskablePPO's categorical distribution
    degenerate. Every phase must always leave the agent something to do."""
    prop = ownable[0]
    assert encoder._legal_mask(phase, prop, 0).sum() > 0


@pytest.mark.parametrize("phase", ALL_PHASES)
def test_the_clamp_target_is_always_legal(encoder, ownable, phase):
    """``safe_default(phase)`` is what an illegal action is rewritten to, so it
    must itself be legal in that phase -- otherwise the clamp corrupts state."""
    mask = encoder._legal_mask(phase, ownable[0], 0)
    assert mask[safe_default(phase)] == 1


class TestJail:
    def test_paying_requires_the_fine(self, game, encoder):
        game.players[0].balance = 49
        assert encoder._legal_mask(PHASE_JAIL, None, 0)[A_PAY_JAIL] == 0
        game.players[0].balance = 50
        assert encoder._legal_mask(PHASE_JAIL, None, 0)[A_PAY_JAIL] == 1

    def test_the_card_needs_to_be_held(self, game, encoder):
        assert encoder._legal_mask(PHASE_JAIL, None, 0)[A_USE_CARD] == 0
        game.players[0].jail_cards.append(object())
        assert encoder._legal_mask(PHASE_JAIL, None, 0)[A_USE_CARD] == 1

    def test_rolling_is_always_available(self, encoder):
        assert encoder._legal_mask(PHASE_JAIL, None, 0)[A_ROLL_JAIL] == 1


class TestBuy:
    def test_declining_is_always_available(self, encoder, ownable):
        assert encoder._legal_mask(PHASE_BUY, ownable[0], 0)[A_DECLINE] == 1

    def test_buy_is_offered_exactly_when_the_price_is_reachable(
            self, game, encoder, ownable):
        prop = ownable[0]
        red = game.players[0]
        red.balance = prop.price
        assert encoder._legal_mask(PHASE_BUY, prop, 0)[A_BUY] == 1
        # Broke, and nothing to mortgage: out of reach.
        red.balance = 0
        assert encoder._legal_mask(PHASE_BUY, prop, 0)[A_BUY] == 0
        # Broke, but a mortgageable tile puts it back within reach.
        spare = ownable[5]
        give(red, spare)
        reachable = red.balance + encoder._raisable_cash(red) >= prop.price
        assert encoder._legal_mask(PHASE_BUY, prop, 0)[A_BUY] == int(reachable)


class TestAuction:
    def test_a_bid_bucket_is_legal_only_when_cash_covers_it(
            self, game, encoder, ownable):
        prop = ownable[0]
        red = game.players[0]
        red.balance = 100
        mask = encoder._legal_mask(PHASE_AUCTION, prop, 0)
        for k, frac in enumerate(BID_FRACTIONS):
            affordable = red.balance >= int(round(frac * prop.price))
            assert mask[A_AUCTION_BID + k] == int(affordable)


class TestManage:
    def test_ending_the_phase_is_always_available(self, encoder):
        assert encoder._legal_mask(PHASE_MANAGE, None, 0)[A_END_MANAGE] == 1

    def test_building_is_offered_only_on_a_monopoly(
            self, game, encoder, ownable):
        red = game.players[0]
        brown = group_named(encoder, "brown")
        give(red, brown[0])
        i = ownable.index(brown[0])
        # One of two browns: no monopoly, so no build.
        assert encoder._legal_mask(PHASE_MANAGE, None, 0)[A_BUILD + i] == 0
        give(red, brown[1])
        assert encoder._legal_mask(PHASE_MANAGE, None, 0)[A_BUILD + i] == 1

    def test_mortgaging_is_forced_liquidation_only(self, game, encoder, ownable):
        red = game.players[0]
        prop = ownable[0]
        give(red, prop)
        i = ownable.index(prop)
        # Masked out of voluntary MANAGE (it enabled a mortgage-flip exploit)...
        assert encoder._legal_mask(PHASE_MANAGE, None, 0)[A_MORTGAGE + i] == 0
        # ...but available when raising cash under duress.
        assert encoder._legal_mask(PHASE_LIQUIDATE, None, 0)[A_MORTGAGE + i] == 1

    def test_every_offered_trade_action_is_actually_proposable(
            self, game, encoder, ownable):
        """The trade band is the largest slice of the action space (84 of 211).
        Every action it offers must survive ``_can_propose_trade``."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0])
        give(blue, orange[1], orange[2])          # a set Red can deny
        give(red, group_named(encoder, "brown")[0])   # something to hand over

        mask = encoder._legal_mask(PHASE_MANAGE, None, 0)
        offered = [a for a in range(A_TRADE, A_TRADE + NUM_OWNABLE * NUM_TRADE_TIERS)
                   if mask[a]]
        assert offered, "expected Red to be able to pry an orange off Blue"
        for action in offered:
            i, tier = decode_trade_action(action)
            assert 0 <= tier < NUM_TRADE_TIERS
            assert encoder._can_propose_trade(red, ownable[i])

    def test_a_target_is_offered_at_most_once_per_manage_phase(
            self, game, encoder, ownable):
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0])
        give(blue, orange[1], orange[2])
        give(red, group_named(encoder, "brown")[0])

        target_i = ownable.index(orange[1])
        assert encoder._legal_mask(PHASE_MANAGE, None, 0)[
            A_TRADE + target_i * NUM_TRADE_TIERS] == 1

        encoder._traded_this_manage.add(target_i)
        assert encoder._legal_mask(PHASE_MANAGE, None, 0)[
            A_TRADE + target_i * NUM_TRADE_TIERS] == 0
