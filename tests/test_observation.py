"""What the policy can actually see.

The observation is shape-locked to a trained model, so these tests double as the
record of what dimension ``265`` means.
"""

import numpy as np

from engine.constants import PHASE_AUCTION, PHASE_LIQUIDATE, PHASE_MANAGE
from engine.observation import observation_length
from tests.conftest import give, group_named


def test_observation_length_matches_the_encoder(encoder):
    obs = encoder._encode_obs(0, PHASE_MANAGE, None)
    assert obs.shape == (observation_length(4),) == (265,)
    assert obs.dtype == np.float32


def test_the_agent_can_see_the_debt_it_owes(game, encoder, ownable):
    """``_on_shortfall`` asks the agent what to sell to cover ``amount``. That
    number was handed to the heuristic baselines but never encoded, so the policy
    was choosing what to mortgage while blind to whether it owed $50 or $2,000."""
    red = game.players[0]
    give(red, *ownable[:4])
    red.balance = 100

    small = encoder._encode_obs(0, PHASE_LIQUIDATE, None, amount=50)
    large = encoder._encode_obs(0, PHASE_LIQUIDATE, None, amount=2000)

    assert not np.array_equal(small, large), (
        "a $50 debt and a $2,000 debt must not look identical to the policy")


def test_the_agent_can_see_how_far_the_bidding_has_climbed(
        game, encoder, ownable):
    """The auction hook is called once per ascending round. With ``min_bid``
    absent from the observation, every round looked like the same state."""
    prop = ownable[0]
    opening = encoder._encode_obs(0, PHASE_AUCTION, prop, amount=10)
    contested = encoder._encode_obs(0, PHASE_AUCTION, prop, amount=400)

    assert not np.array_equal(opening, contested)


def test_the_agent_can_see_the_clock(game, encoder):
    """The episode is capped and the first-monopoly bonus decays with the turn
    count, so a time-blind state cannot predict its own return."""
    early = encoder._encode_obs(0, PHASE_MANAGE, None)
    for _ in range(200):
        game.advance_turn()
    late = encoder._encode_obs(0, PHASE_MANAGE, None)

    assert not np.array_equal(early, late)
    assert late[-1] > early[-1]          # the clock feature advanced
    assert 0.0 <= late[-1] <= 1.0


def test_coverage_feature_reports_whether_the_debt_is_reachable(
        game, encoder, ownable):
    red = game.players[0]
    give(red, *group_named(encoder, "orange"))
    red.balance = 0

    reach = red.balance + encoder._raisable_cash(red)
    covered = encoder._encode_obs(0, PHASE_LIQUIDATE, None, amount=int(reach // 2))
    hopeless = encoder._encode_obs(0, PHASE_LIQUIDATE, None, amount=int(reach * 10))

    # The coverage feature sits just before the clock.
    assert covered[-2] == 1.0            # comfortably raisable
    assert hopeless[-2] < 0.2            # nowhere near enough
