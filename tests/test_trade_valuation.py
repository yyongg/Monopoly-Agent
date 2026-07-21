"""Set valuation: per-set strength, the completion curve, and the game stage.

:class:`engine.valuation.SetValuer` decides what a monopoly -- and a *share* of
one -- is worth. Two properties matter enough to pin:

* **strength** ranks sets by traffic x full-development rent, tilted by what the
  set costs to buy and build. Without the cost tilt the ranking puts green on top
  (highest hotel rent, $3,000 to develop, worst payback on the board) instead of
  orange.
* **the completion curve** makes value continuous in how much of a group you
  hold. It replaced a step function that paid a set premium only at exactly
  one-tile-from-complete, which is what let a human trade junk for monopolies --
  see ``TestNoCliff``, which pins the bug that motivated all of this.
"""

import pytest

from engine.config import RewardConfig
from engine.observation import ObsEncoder
from models.tiles.properties.utility import Utility
from tests.conftest import give, group_named


def _util_group(encoder):
    return next(g for g in encoder._groups if isinstance(g[0], Utility))


class TestSetStrength:
    def test_money_sets_outrank_cheap_sets(self, encoder):
        def s(color):
            return encoder.sets.strength(group_named(encoder, color))

        # Orange/red sit in the high-traffic band past Jail with strong rents;
        # light_blue is middling; utilities are the weakest set on the board.
        assert s("orange") > s("light_blue")
        assert s("red") > s("light_blue")
        assert s("light_blue") > encoder.sets.strength(_util_group(encoder))
        assert s("orange") > s("brown")

    def test_orange_outranks_green(self, encoder):
        """The cost tilt, and the reason it exists. Green has the highest
        full-development rent of any group, so on earning power alone it outranks
        orange -- but it costs $3,000 to build out and takes the longest of any
        street set to pay that back. Ranking them the other way round is what a
        Monopoly player would call getting it backwards."""
        assert (encoder.sets.strength(group_named(encoder, "orange"))
                > encoder.sets.strength(group_named(encoder, "green")))

    def test_dropping_the_cost_tilt_puts_green_back_on_top(self, game, ownable):
        """The counterfactual: with the efficiency tilt clamped to a no-op, the
        ranking reverts to raw earning power and green wins. Pins *which* term is
        doing the work, so a future tweak to it can't quietly stop mattering."""
        flat = ObsEncoder(RewardConfig(set_quality_clamp=(1.0, 1.0)))
        flat.bind(game, ownable)
        assert (flat.sets.strength(group_named(flat, "green"))
                > flat.sets.strength(group_named(flat, "orange")))

    def test_all_strengths_within_clamp(self, encoder):
        lo, hi = RewardConfig().set_strength_clamp
        for grp in encoder._groups:
            assert lo <= encoder.sets.strength(grp) <= hi

    def test_clamp_floor_and_ceiling_are_exercised(self, encoder):
        lo, _ = RewardConfig().set_strength_clamp
        strengths = [encoder.sets.strength(grp) for grp in encoder._groups]
        # The weakest set (utilities) is pinned to the floor, and a money set
        # clears 1.0 -- so strength genuinely differentiates, it doesn't collapse
        # to a constant.
        assert min(strengths) == pytest.approx(lo)
        assert max(strengths) > 1.0

    def test_normalised_to_unit_mean(self, game, ownable):
        # With the clamps effectively disabled, the per-set multipliers average
        # ~1.0 -- the property that makes strength a *redistribution* of the flat
        # trade_monopoly_mult rather than a change to the overall scale.
        enc = ObsEncoder(RewardConfig(set_strength_clamp=(0.0, 100.0),
                                      set_quality_clamp=(0.0, 100.0)))
        enc.bind(game, ownable)
        strengths = [enc.sets.strength(grp) for grp in enc._groups]
        assert sum(strengths) / len(strengths) == pytest.approx(1.0)

    def test_unrecognised_group_falls_back_to_one(self, encoder):
        assert encoder.sets.strength([]) == pytest.approx(1.0)


class TestCompletionCurve:
    def test_endpoints(self, encoder):
        assert encoder.sets.completion(0, 3) == pytest.approx(0.0)
        assert encoder.sets.completion(3, 3) == pytest.approx(1.0)

    def test_monotone_and_convex(self, encoder):
        """Strictly increasing (every tile is worth something) and convex (the
        tile that completes the set is the biggest single step)."""
        f = [encoder.sets.completion(k, 3) for k in range(4)]
        steps = [f[k + 1] - f[k] for k in range(3)]
        assert all(b > a for a, b in zip(f, f[1:]))
        assert all(b > a for a, b in zip(steps, steps[1:]))

    def test_exponent_of_one_prices_every_tile_alike(self, game, ownable):
        enc = ObsEncoder(RewardConfig(set_progress_exponent=1.0))
        enc.bind(game, ownable)
        steps = [enc.sets.completion(k + 1, 3) - enc.sets.completion(k, 3)
                 for k in range(3)]
        assert steps[0] == pytest.approx(steps[1]) == pytest.approx(steps[2])


class TestNoCliff:
    """The bug this whole revamp exists for.

    Measured against the old valuation: a player holding 2/3 of orange priced
    St. James at **$227** -- exactly what it priced the tile at with 1/3 of the
    set, because the old rule only paid a set premium at one-from-complete -- and
    happily sold it for a brown and $200. The same tile was worth **$4,937** once
    the set was whole: a 21.7x cliff with nothing underneath it.
    """

    @staticmethod
    def _value_holding(encoder, game, held):
        orange = group_named(encoder, "orange")
        red = game.players[0]
        give(red, *orange[:held])
        return encoder._trade_value(orange[0], red)

    def test_progress_is_priced(self, game, encoder):
        """2/3 of a set must be worth materially more than 1/3 of it. Under the
        old step function these were *equal*, which is the whole bug."""
        one = self._value_holding(encoder, game, 1)
        red = game.players[0]
        orange = group_named(encoder, "orange")
        give(red, orange[1])                       # now 2/3
        two = encoder._trade_value(orange[0], red)
        assert two > 2.0 * one

    def test_completing_is_a_step_not_a_cliff(self, game, encoder):
        red = game.players[0]
        orange = group_named(encoder, "orange")
        give(red, orange[0], orange[1])
        two = encoder._trade_value(orange[0], red)
        give(red, orange[2])
        three = encoder._trade_value(orange[0], red)
        # Completing is still the biggest jump -- but a jump, not the 21.7x cliff
        # that made every tile below it free.
        assert three > two
        assert three < 2.5 * two

    def test_junk_plus_cash_cannot_buy_two_thirds_of_orange(self, game, encoder):
        """The reported exploit, verbatim: Red holds 2/3 of orange, Blue offers a
        brown and $200 for the tile Red needs. The old rule took the deal."""
        red, blue = game.players[0], game.players[1]
        orange = group_named(encoder, "orange")
        brown = group_named(encoder, "brown")
        give(red, orange[0], orange[1])
        give(blue, brown[0])

        accepted, _ = encoder.accepts(red, [brown[0]], [orange[0]], 200)
        assert not accepted
        # Not merely a threshold effect -- it is nowhere near.
        accepted, _ = encoder.accepts(red, [brown[0]], [orange[0]], 1500)
        assert not accepted

    def test_railroads_are_not_free_at_three_of_four(self, game, encoder):
        from models.tiles.properties.railroad import Railroad
        rails = next(g for g in encoder._groups if isinstance(g[0], Railroad))
        red, blue = game.players[0], game.players[1]
        brown = group_named(encoder, "brown")
        give(red, *rails[:3])
        give(blue, brown[0])
        accepted, _ = encoder.accepts(red, [brown[0]], [rails[0]], 200)
        assert not accepted


class TestStageInflation:
    def test_a_fresh_board_is_unscaled(self, encoder):
        assert encoder.sets.stage() == pytest.approx(1.0)

    def test_set_prices_rise_with_the_board_s_cash(self, game, encoder):
        orange = group_named(encoder, "orange")
        early = encoder.sets.monopoly_value(orange)
        for p in game.players:
            p.balance = 4000            # everyone has lapped Go a few times
        late = encoder.sets.monopoly_value(orange)
        assert late > early

    def test_a_poor_board_never_discounts(self, game, encoder):
        for p in game.players:
            p.balance = 50
        assert encoder.sets.stage() == pytest.approx(1.0)

    def test_the_cap_binds(self, game, encoder):
        for p in game.players:
            p.balance = 1_000_000
        assert encoder.sets.stage() == pytest.approx(
            RewardConfig().stage_inflation_cap)

    def test_weight_of_zero_disables_it(self, game, ownable):
        enc = ObsEncoder(RewardConfig(stage_inflation_weight=0.0))
        enc.bind(game, ownable)
        for p in game.players:
            p.balance = 9999
        assert enc.sets.stage() == pytest.approx(1.0)
