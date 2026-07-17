"""Per-set strength valuation.

``ObsEncoder._set_strength`` weighs each monopoly by traffic x full-development
rent, normalised so the *average* set is ~1.0, then clamped. It replaced the flat
``trade_monopoly_mult`` in trade valuations so a cheap low-traffic set (utilities)
is no longer priced like the orange money set -- the fix for the set-ping-pong
loop the learned agent had fallen into.
"""

import pytest

from engine.config import RewardConfig
from engine.observation import ObsEncoder
from models.tiles.properties.utility import Utility
from tests.conftest import group_named


def _util_group(encoder):
    return next(g for g in encoder._groups if isinstance(g[0], Utility))


class TestSetStrength:
    def test_money_sets_outrank_cheap_sets(self, encoder):
        def s(color):
            return encoder._set_strength(group_named(encoder, color))

        # Orange/red sit in the high-traffic band past Jail with strong rents;
        # light_blue is middling; utilities are the weakest set on the board.
        assert s("orange") > s("light_blue")
        assert s("red") > s("light_blue")
        assert s("light_blue") > encoder._set_strength(_util_group(encoder))
        assert s("orange") > s("brown")

    def test_all_strengths_within_clamp(self, encoder):
        lo, hi = RewardConfig().set_strength_clamp
        for grp in encoder._groups:
            assert lo <= encoder._set_strength(grp) <= hi

    def test_clamp_floor_and_ceiling_are_exercised(self, encoder):
        lo, hi = RewardConfig().set_strength_clamp
        strengths = [encoder._set_strength(grp) for grp in encoder._groups]
        # The weakest set (utilities / brown) is pinned to the floor, and a money
        # set clears 1.0 -- so strength genuinely differentiates, it doesn't
        # collapse to a constant.
        assert min(strengths) == pytest.approx(lo)
        assert max(strengths) > 1.0

    def test_normalised_to_unit_mean(self, game, ownable):
        # With the clamp effectively disabled, the per-set multipliers average
        # ~1.0 -- the property that makes _set_strength a *redistribution* of the
        # flat trade_monopoly_mult rather than a change to the overall scale.
        enc = ObsEncoder(RewardConfig(set_strength_clamp=(0.0, 100.0)))
        enc.bind(game, ownable)
        strengths = [enc._set_strength(grp) for grp in enc._groups]
        assert sum(strengths) / len(strengths) == pytest.approx(1.0)

    def test_unrecognised_group_falls_back_to_one(self, encoder):
        assert encoder._set_strength([]) == pytest.approx(1.0)
