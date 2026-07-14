"""Hand-crafted fixed-policy Monopoly agents (FP-A / FP-B / FP-C).

These are *state-aware* opponents modelled on the tournament-style strategies in
Bonjour et al. (2021): they read live game state and emit an action id in the
env's flat scheme, validated against the legal mask. Unlike the trivial engine
baseline (:meth:`models.player.Player.decide_purchase`, which buys anything
affordable and never bids/builds/trades), they buy toward monopolies, bid in
auctions, build houses, and propose set-completing trades -- a *meaningful*
stationary yardstick and a stronger self-play opponent than the raw baseline.

They plug into :class:`engine.rl_env.MonopolyEnv` via the state-aware opponent
protocol (see ``MonopolyEnv._policy_decide``): each agent exposes
``bind(game, ownable)`` and ``decide(seat, phase, prop, amount, mask, offer)``.
All valuations reuse the shared :class:`~engine.observation.ObsEncoder`, so the
bots' economics stay consistent with the env and the GUI.

The three variants differ *only* in their per-group priority weights:

* ``FP_A`` -- equal priority to every group.
* ``FP_B`` -- railroads and dark-blue (Park Place / Boardwalk) high, utilities low.
* ``FP_C`` -- orange and light-blue (the tournament "money" groups) high.
"""

from engine.constants import (
    PHASE_JAIL, PHASE_BUY, PHASE_MANAGE, PHASE_LIQUIDATE,
    PHASE_AUCTION, PHASE_TRADE_RESPOND,
    A_PAY_JAIL, A_USE_CARD, A_ROLL_JAIL, A_BUY, A_DECLINE, A_END_MANAGE,
    A_BUILD, A_SELL, A_MORTGAGE, A_UNMORTGAGE,
    A_TRADE_ACCEPT, A_TRADE_REJECT, A_AUCTION_PASS, A_AUCTION_BID,
    NUM_OWNABLE, NUM_BID_LEVELS, BID_FRACTIONS, trade_action,
)
from engine.observation import ObsEncoder, safe_default
from models.tiles.properties.street_property import StreetProperty
from models.tiles.properties.railroad import Railroad


def _tile_key(tile):
    """The priority-map key for a tile: its street colour, or ``"railroad"`` /
    ``"utility"`` for the non-street groups."""
    if isinstance(tile, StreetProperty):
        return tile.color
    if isinstance(tile, Railroad):
        return "railroad"
    return "utility"


class HeuristicAgent:
    """A fixed-policy opponent driven by simple tournament-style rules.

    Args:
        priorities (dict[str, float] | None): Per-group weight overrides keyed by
            :func:`_tile_key` (default weight 1.0). Groups weighted >= 2 are
            pursued aggressively (bought/bid on even without completing a set);
            groups weighted <= 0.3 are largely skipped in auctions.
        cash_buffer (int): Cash the agent tries to keep in hand before spending
            on non-essential buys / builds, so it can pay rent.
        name (str): Label for debugging.
    """

    def __init__(self, priorities=None, cash_buffer=150, name="FP"):
        self.priorities = priorities or {}
        self.cash_buffer = cash_buffer
        self.name = name
        self.game = None
        self.ownable = []
        self.encoder = ObsEncoder()

    def bind(self, game, ownable):
        """Attach to a fresh game (called each episode by the env)."""
        self.game = game
        self.ownable = ownable
        self.encoder.bind(game, ownable)
        return self

    def _priority(self, tile):
        return self.priorities.get(_tile_key(tile), 1.0)

    # -- Top-level dispatch -------------------------------------------------
    def decide(self, seat, phase, prop, amount, mask, offer=None):
        """Return a mask-legal action id for the decision ``seat`` faces.

        ``offer`` is the ``{recv, give, cash}`` trade dict during
        ``PHASE_TRADE_RESPOND`` (supplied by the env), ``None`` otherwise.
        """
        player = self.game.players[seat]
        if phase == PHASE_JAIL:
            return self._decide_jail(player, mask)
        if phase == PHASE_BUY:
            return self._decide_buy(player, prop, mask)
        if phase == PHASE_AUCTION:
            return self._decide_auction(player, prop, mask)
        if phase == PHASE_TRADE_RESPOND:
            return self._decide_trade_respond(player, offer, mask)
        if phase == PHASE_LIQUIDATE:
            return self._decide_liquidate(player, mask)
        if phase == PHASE_MANAGE:
            return self._decide_manage(player, mask)
        return safe_default(phase)

    # -- Per-phase policies -------------------------------------------------
    def _decide_jail(self, player, mask):
        """Use a free card if held; otherwise pay out when flush, else roll."""
        if mask[A_USE_CARD]:
            return A_USE_CARD
        if mask[A_PAY_JAIL] and player.balance >= self.cash_buffer + 50:
            return A_PAY_JAIL
        return A_ROLL_JAIL

    def _decide_buy(self, player, prop, mask):
        """Buy set-completers unconditionally; buy priority/affordable tiles with
        a cash buffer; otherwise decline (it goes to auction)."""
        if not mask[A_BUY]:
            return A_DECLINE
        if self.encoder._completes_monopoly_for(player, prop):
            return A_BUY
        buffer = 0 if self._priority(prop) >= 2.0 else self.cash_buffer
        if player.balance - prop.price >= buffer:
            return A_BUY
        return A_DECLINE

    def _decide_auction(self, player, prop, mask):
        """Bid up to a priority-scaled fraction of the tile's value; pick the
        highest legal bucket at or below that target, else pass."""
        if self.encoder._completes_monopoly_for(player, prop):
            target = 1.5
        else:
            pr = self._priority(prop)
            if pr <= 0.3:
                return A_AUCTION_PASS
            target = 1.0 if pr >= 2.0 else 0.75
        for k in range(NUM_BID_LEVELS - 1, -1, -1):
            if BID_FRACTIONS[k] <= target and mask[A_AUCTION_BID + k]:
                return A_AUCTION_BID + k
        return A_AUCTION_PASS

    def _decide_trade_respond(self, player, offer, mask):
        """Accept when the offer is non-negative by the shared trade valuation
        (receive ``offer['recv']`` + cash, give up ``offer['give']``)."""
        if offer is None:
            return A_TRADE_REJECT
        recv, give, cash = offer.get("recv"), offer.get("give"), offer.get("cash", 0)
        gain = [recv] if recv is not None else []
        lose = [give] if give is not None else []
        accepted, _ = self.encoder.accepts(player, gain, lose, cash)
        if accepted and mask[A_TRADE_ACCEPT]:
            return A_TRADE_ACCEPT
        return A_TRADE_REJECT

    def _decide_manage(self, player, mask):
        """One management step: build on the best completed set, else propose a
        set-completing / denial trade, else lift a mortgage when flush, else stop.

        Only ever returns a mask-legal action, so each step makes real progress
        (houses rise, a trade target is consumed, a mortgage lifts) and the env's
        uncapped manage loop is guaranteed to terminate at ``END_MANAGE``.
        """
        # 1. Build a house on the highest-priority buildable tile, keeping a cash
        #    buffer so we don't build ourselves into bankruptcy.
        best_build, best_score = None, -1.0
        for i, p in enumerate(self.ownable):
            if not mask[A_BUILD + i]:
                continue
            if player.balance - p.house_cost() < self.cash_buffer:
                continue
            score = self._priority(p)
            if score > best_score:
                best_build, best_score = i, score
        if best_build is not None:
            return A_BUILD + best_build

        # 2. Propose a trade (fair cash tier) for a proposable target, preferring
        #    one that completes our own set over a pure denial.
        best_trade, best_completes = None, False
        for i, p in enumerate(self.ownable):
            if not mask[trade_action(i, 1)]:
                continue
            completes = self.encoder._completes_monopoly_for(player, p)
            if best_trade is None or (completes and not best_completes):
                best_trade, best_completes = i, completes
        if best_trade is not None:
            return trade_action(best_trade, 1)

        # 3. Lift a mortgage when we are cash-flush (reactivates rent income).
        if player.balance >= 2 * self.cash_buffer:
            for i in range(NUM_OWNABLE):
                if mask[A_UNMORTGAGE + i]:
                    return A_UNMORTGAGE + i

        return A_END_MANAGE

    def _decide_liquidate(self, player, mask):
        """Raise cash under a shortfall: mortgage, then sell houses, choosing the
        least valuable tile first; give up (stop) when nothing remains."""
        best_mort, best_val = None, None
        for i, p in enumerate(self.ownable):
            if mask[A_MORTGAGE + i]:
                val = self.encoder._trade_value(p, player)
                if best_val is None or val < best_val:
                    best_mort, best_val = i, val
        if best_mort is not None:
            return A_MORTGAGE + best_mort
        for i in range(NUM_OWNABLE):
            if mask[A_SELL + i]:
                return A_SELL + i
        return A_END_MANAGE


# -- Priority profiles (mirroring the paper's FP-A/B/C) ---------------------
FP_A_PRIORITIES = {}
FP_B_PRIORITIES = {"railroad": 2.0, "dark_blue": 2.0, "utility": 0.3}
FP_C_PRIORITIES = {"orange": 2.0, "light_blue": 2.0}


def make_fp_a(**kw):
    return HeuristicAgent(FP_A_PRIORITIES, name="FP-A", **kw)


def make_fp_b(**kw):
    return HeuristicAgent(FP_B_PRIORITIES, name="FP-B", **kw)


def make_fp_c(**kw):
    return HeuristicAgent(FP_C_PRIORITIES, name="FP-C", **kw)


def make_baseline_trio(**kw):
    """Fresh ``[FP-A, FP-B, FP-C]`` list, ready to hand to ``MonopolyEnv`` as a
    per-seat opponent pool (dealt round-robin across the opponent seats)."""
    return [make_fp_a(**kw), make_fp_b(**kw), make_fp_c(**kw)]
