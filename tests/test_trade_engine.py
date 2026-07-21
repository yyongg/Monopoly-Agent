"""The heuristic trade engine: what it offers, and what it refuses.

:class:`engine.trade.TradeEngine` replaced the policy's 84-action trade band.
These tests pin the behaviours that band got wrong -- giving away sets, buying
what it did not need, and re-proposing forever -- plus the multi-tile packaging
the band could not express at all.
"""

import pytest

from engine.config import RewardConfig
from engine.observation import ObsEncoder
from engine.trade import TradeEngine
from tests.conftest import give, group_named


@pytest.fixture
def trades(encoder):
    return TradeEngine(encoder)


class TestOfferConstruction:
    def test_it_buys_the_tile_that_completes_its_set(self, game, encoder, trades):
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0], orange[1])
        give(blue, orange[2])
        give(red, group_named(encoder, "brown")[0])
        red.balance = 4000

        offer = trades.best_offer(red, blue)
        assert offer is not None
        assert orange[2] in offer.receive

    def test_it_asks_for_both_tiles_when_one_is_not_enough(
            self, game, encoder, trades):
        """Multi-tile receive. Blue holds two of the three oranges, so *no*
        one-for-one swap can complete Red's set -- the shape the old 84-id trade
        band was structurally unable to name."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0])
        give(blue, orange[1], orange[2])
        give(red, *group_named(encoder, "brown"))
        give(red, *group_named(encoder, "light_blue"))
        red.balance = 6000

        offer = trades.best_offer(red, blue)
        assert offer is not None
        assert orange[1] in offer.receive and orange[2] in offer.receive

    def test_it_pays_in_tiles_when_cash_alone_falls_short(
            self, game, encoder, trades):
        """Multi-tile give: the paper's rule -- offer what is cheap to you and
        dear to them. Red cannot cover Blue's asking price in cash, so it makes up
        the difference in junk rather than failing to deal."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0], orange[1])
        give(blue, orange[2])
        give(red, *group_named(encoder, "brown"))
        give(red, group_named(encoder, "light_blue")[0])
        red.balance = 1000

        asking = encoder._trade_value(orange[2], blue)
        assert red.balance < asking, "setup: cash alone must not be enough"

        offer = trades.best_offer(red, blue)
        assert offer is not None, "expected junk to close the gap"
        assert len(offer.give) >= 1
        assert offer.cash <= red.balance

    def test_a_rich_buyer_just_pays_cash(self, game, encoder, trades):
        """The mirror: when cash covers the ask, don't hand over tiles as well.
        ``_price`` takes the first package that clears, fewest tiles first."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0], orange[1])
        give(blue, orange[2])
        give(red, *group_named(encoder, "brown"))
        red.balance = 8000

        offer = trades.best_offer(red, blue)
        assert offer is not None
        assert offer.give == []
        assert offer.cash > 0

    def test_packages_respect_the_cap(self, game, ownable):
        cfg = RewardConfig(trade_max_package=1)
        enc = ObsEncoder(cfg).bind(game, ownable)
        engine = TradeEngine(enc)
        red, blue = game.players[0], game.players[1]
        orange = group_named(enc, "orange")
        give(red, orange[0])
        give(blue, orange[1], orange[2])
        give(red, *group_named(enc, "brown"))
        give(red, *group_named(enc, "light_blue"))
        red.balance = 6000

        offer = engine.best_offer(red, blue)
        if offer is not None:
            assert len(offer.give) <= 1 and len(offer.receive) <= 1

    def test_nothing_to_gain_means_no_offer(self, game, encoder, trades):
        """Blue holds only a tile Red has no use for, and Red holds nothing that
        would make the swap worth Blue's while."""
        red, blue = game.players[0], game.players[1]
        give(blue, group_named(encoder, "brown")[0])
        assert trades.best_offer(red, blue) is None

    def test_it_will_not_gift_the_partner_a_monopoly(self, game, encoder, trades):
        """The no-gift rule. Red holds the one red tile Blue needs; Blue holds the
        one orange tile Red needs.

        Caught in a live probe: without this rule the engine *sought out* such
        tiles, because ``trade_denial_weight`` (0.5) means gifting a set costs the
        giver half what it hands the taker -- pure joint surplus, and the greedy
        package builder maximises exactly that. It was handing over completed sets
        to buy single tiles, 47 trades a game. The weight has to stay below 1.0 or
        nothing trades at all, so this is a rule, not a coefficient.
        """
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        reds = group_named(encoder, "red")
        give(red, orange[0], orange[1], reds[0])   # reds[0] completes Blue's set
        give(blue, reds[1], reds[2], orange[2])
        red.balance = blue.balance = 2000

        assert not trades._may_hand_over(red, blue, [reds[0]], [orange[2]])
        offer = trades.best_offer(red, blue)
        if offer is not None:
            assert reds[0] not in offer.give

    def test_a_genuine_set_for_set_is_still_allowed(self, game, encoder, trades):
        """The exception the rule exists around: arming a rival is worth it when
        we complete a *stronger* set in the same breath. Red hands over the last
        light_blue (a weak set) to complete orange (the strongest)."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        light = group_named(encoder, "light_blue")
        give(red, orange[0], orange[1], light[0])
        give(blue, light[1], light[2], orange[2])
        red.balance = blue.balance = 3000

        assert trades._may_hand_over(red, blue, [light[0]], [orange[2]])

    def test_a_package_that_gifts_a_set_collectively_is_barred(
            self, game, encoder, trades):
        """The gift rule reads the whole package. Either brown alone is harmless;
        both together are a monopoly for the partner -- and here Red completes
        nothing in return, so there is no set-for-set to justify it."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        brown = group_named(encoder, "brown")
        give(red, orange[0], *brown)      # only 1/3 of orange: buying [2] wins nothing
        give(blue, orange[2])

        assert trades._may_hand_over(red, blue, [brown[0]], [orange[2]])
        assert not trades._may_hand_over(red, blue, list(brown), [orange[2]])

    def test_it_does_not_trade_oranges_to_buy_oranges(
            self, game, encoder, trades):
        """A tile from the group we are buying into is never part of the payment.
        Caught in a live probe: those tiles are a wash by construction, so they
        sorted to the *top* of the surplus order and crowded out the junk that
        would actually have paid for the deal."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0], orange[1], group_named(encoder, "brown")[0])
        give(blue, orange[2])
        red.balance = 5000

        offer = trades.best_offer(red, blue)
        assert offer is not None
        assert not any(t in orange for t in offer.give)

    def test_a_package_that_changes_nothing_is_not_proposed(
            self, game, encoder, trades):
        """Set value is not additive: giving one orange to receive another is a
        wash, not two separate completion jumps.

        Caught in a live probe -- summing per-tile marginals had the engine paying
        $2,000 for a swap that left both sides exactly where they started.
        """
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0], orange[1])
        give(blue, orange[2])
        assert encoder.sets.swap_delta(
            red, [orange[2]], [orange[0]], partner=blue) == pytest.approx(0.0)

    def test_two_tiles_of_a_group_are_worth_more_together(
            self, game, encoder, trades):
        """The other half of non-additivity: taking two tiles of a group must be
        worth ``f(2)``, not ``2 x f(1)``."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(blue, orange[0], orange[1])
        both = encoder.sets.swap_delta(red, [orange[0], orange[1]], [], partner=blue)
        singly = (encoder.sets.swap_delta(red, [orange[0]], [], partner=blue)
                  + encoder.sets.swap_delta(red, [orange[1]], [], partner=blue))
        assert both > singly

    def test_it_never_offers_a_tile_out_of_its_own_set(self, game, encoder, trades):
        """Red owns orange outright. Nothing Blue has is worth breaking it up
        for, and the valuation -- not a special case -- is what says so."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, *orange)
        give(red, group_named(encoder, "brown")[0])
        give(blue, group_named(encoder, "light_blue")[0])
        give(blue, group_named(encoder, "pink")[0])
        red.balance = blue.balance = 3000

        offer = trades.best_offer(red, blue)
        if offer is not None:
            assert not any(t in orange for t in offer.give)


class TestBargaining:
    def test_both_sides_gain(self, game, encoder, trades):
        """The engine prices at the midpoint of the bargaining range, so a deal
        it proposes is strictly good for both sides -- never knife-edge."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0], orange[1])
        give(blue, orange[2])
        give(red, group_named(encoder, "brown")[0])
        red.balance = 4000

        offer = trades.best_offer(red, blue)
        assert offer is not None
        mine = encoder.trade_delta(red, offer.receive, offer.give, offer.cash)
        theirs = encoder.trade_delta(blue, offer.give, offer.receive, offer.cash)
        assert mine > 0 and theirs > 0

    def test_a_partner_accepts_what_the_engine_proposes(
            self, game, encoder, trades):
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0], orange[1])
        give(blue, orange[2])
        give(red, group_named(encoder, "brown")[0])
        red.balance = 4000

        offer = trades.best_offer(red, blue)
        assert offer is not None
        accepted, _ = encoder.accepts(blue, offer.give, offer.receive, offer.cash)
        assert accepted

    def test_zero_sum_denial_kills_every_deal(self, game, ownable):
        """``trade_denial_weight = 1.0`` makes a tile worth exactly as much to the
        blocker as to the acquirer, so no swap has any surplus to split and
        nothing can clear. Pinned because the failure is silent: monopolies simply
        stop forming and games run to the turn cap."""
        enc = ObsEncoder(RewardConfig(trade_denial_weight=1.0)).bind(game, ownable)
        engine = TradeEngine(enc)
        red, blue = game.players[0], game.players[1]
        orange = group_named(enc, "orange")
        give(red, orange[0], orange[1])
        give(blue, orange[2])
        give(red, group_named(enc, "brown")[0])
        red.balance = 10_000
        assert engine.best_offer(red, blue) is None

    def test_it_will_not_spend_its_rent_cushion_on_a_speculative_trade(
            self, game, encoder, trades):
        """Cash below the solvency cushion is not available for a tile that does
        not finish a set -- the same cushion the solvency reward sizes."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0])
        give(blue, orange[1])
        give(red, group_named(encoder, "brown")[0])
        # Blue owns a developed set, so Red's rent exposure (and cushion) is real.
        pink = group_named(encoder, "pink")
        give(blue, *pink)
        for tile in pink:
            tile.houses = 3
        red.balance = 1500

        offer = trades.best_offer(red, blue)
        if offer is not None and offer.cash > 0:
            reserve = encoder._cash_reserve(red)
            assert offer.cash <= max(0, red.balance - reserve) + 1


class TestSettlement:
    def test_an_accepted_deal_moves_the_tiles_and_the_cash(
            self, game, encoder, trades):
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        brown = group_named(encoder, "brown")
        give(red, orange[0], orange[1])
        give(blue, orange[2])
        give(red, brown[0])
        red.balance = 4000
        before = blue.balance

        seen = []
        trades.on_offer = lambda *a: seen.append(a)
        trades.run_round(red)

        assert seen, "expected a proposal"
        _, _, offer, executed, completes = seen[0]
        assert executed and completes
        assert orange[2].owner is red
        assert all(t.owner is blue for t in offer.give)
        assert blue.balance == before + offer.cash

    def test_the_arbiter_can_veto(self, game, encoder, trades):
        """The GUI puts an AI's offer to a *human* partner, who is under no
        obligation to be rational about it."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0], orange[1])
        give(blue, orange[2])
        give(red, group_named(encoder, "brown")[0])
        red.balance = 4000

        trades.arbiter = lambda *a: False
        trades.run_round(red)
        assert orange[2].owner is blue

    def test_at_most_one_deal_per_partner_per_round(self, game, encoder, trades):
        """The churn backstop. The old code re-proposed without bound and
        ping-ponged one tile 424 times in a single measured game."""
        red = game.players[0]
        orange = group_named(encoder, "orange")
        give(red, orange[0], orange[1])
        give(game.players[1], orange[2])
        give(red, *group_named(encoder, "brown"))
        give(red, *group_named(encoder, "light_blue"))
        red.balance = 8000

        seen = []
        trades.on_offer = lambda *a: seen.append(a)
        trades.run_round(red)
        partners = [a[1] for a in seen]
        assert len(partners) == len(set(id(p) for p in partners))

    def test_a_bankrupt_partner_is_skipped(self, game, encoder, trades):
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        give(red, orange[0], orange[1])
        give(blue, orange[2])
        give(red, group_named(encoder, "brown")[0])
        red.balance = 4000
        blue.bankrupt = True

        seen = []
        trades.on_offer = lambda *a: seen.append(a)
        trades.run_round(red)
        assert not seen
