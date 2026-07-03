"""Observation encoding, tile/trade valuations, and legal-action masks.

``ObsEncoder`` is the single source of truth for everything the trained policy
sees and for the dollar-valued heuristics built on top of it. Both
:class:`engine.rl_env.MonopolyEnv` (training) and
``ui.ai_player.GUIAIDecider`` (play) hold one and delegate to it, so the RL
observation can never drift between training and the GUI -- the drift that used
to be guarded by a hand-maintained "mirrors the env" copy in the UI.

An encoder is bound to a live game with :meth:`ObsEncoder.bind`; the owner sets
two transient per-decision fields on it:

* ``_cur_trade`` -- the offer being judged, for a PHASE_TRADE_RESPOND observation,
* ``_traded_this_manage`` -- trade targets already tried this MANAGE phase, so
  :meth:`_legal_mask` stops re-offering them.
"""

import json
import os

import numpy as np

from engine.config import RewardConfig
from engine.constants import (
    PHASE_JAIL, PHASE_BUY, PHASE_MANAGE, PHASE_LIQUIDATE,
    PHASE_AUCTION, PHASE_TRADE_RESPOND, NUM_PHASES,
    A_PAY_JAIL, A_USE_CARD, A_ROLL_JAIL, A_BUY, A_DECLINE, A_END_MANAGE,
    A_BUILD, A_SELL, A_MORTGAGE, A_UNMORTGAGE, A_TRADE,
    A_TRADE_ACCEPT, A_TRADE_REJECT, A_AUCTION_PASS, A_AUCTION_BID,
    NUM_OWNABLE, NUM_GROUPS, NUM_ACTIONS, BID_FRACTIONS,
)
from models.tiles.properties.street_property import StreetProperty
from models.tiles.properties.railroad import Railroad
from models.tiles.properties.utility import Utility


LANDING_FREQ_PATH = os.path.join("runs", "board_visits.json")
_LAND_FREQ_CACHE = None


def load_landing_frequencies(path=LANDING_FREQ_PATH):
    """Returns ``{board_pos: landing_share}`` from the saved visit table.

    Cached after first load. On a missing/unreadable file returns ``{}``, which
    callers treat as a uniform prior (see ``ObsEncoder._traffic``)."""
    global _LAND_FREQ_CACHE
    if _LAND_FREQ_CACHE is not None:
        return _LAND_FREQ_CACHE
    freqs = {}
    try:
        with open(path) as f:
            data = json.load(f)
        for t in data["tiles"]:
            freqs[int(t["pos"])] = float(t["frequency"])
    except (OSError, ValueError, KeyError):
        freqs = {}
    _LAND_FREQ_CACHE = freqs
    return freqs


def base_rent(prop):
    """A nominal single-ownership rent for ``prop``, used as a scale-normalised
    "rent it can collect" observation feature (exact value is unimportant)."""
    if isinstance(prop, StreetProperty):
        return float(prop.rent_table[0])
    if isinstance(prop, Railroad):
        return 25.0            # nominal one-railroad rent
    return 28.0               # utility: 4x an average roll of 7


def build_groups(ownable):
    """Groups ownable tiles into the sets that form a monopoly, in a stable
    order (by group key) so the per-group observation features line up
    identically between the env and the GUI."""
    groups = {}
    for t in ownable:
        if isinstance(t, StreetProperty):
            key = ("street", t.color)
        elif isinstance(t, Railroad):
            key = ("railroad", "")
        else:
            key = ("utility", "")
        groups.setdefault(key, []).append(t)
    return [groups[k] for k in sorted(groups)]


def observation_length(num_players):
    """Length of the flat observation vector for ``num_players`` seats."""
    return (6 * num_players  # balance, pos, jail, cards, bankrupt,
                             # expected profit/turn
            + NUM_OWNABLE * (num_players + 3)
            + NUM_PHASES + 1
            + 3               # trade context: recv tile, give tile, cash
            + 2               # context flags: completes mine / opp's set
            + NUM_GROUPS * 2  # per-group progress: mine / max-opp frac
            + 5)              # context prop economics: price, rent, traffic,
                              # buy-yield, dev-ROI


def safe_default(phase):
    """The always-legal fallback action for ``phase`` (used to clamp an illegal
    or unavailable action instead of corrupting game state)."""
    if phase == PHASE_JAIL:
        return A_ROLL_JAIL
    if phase == PHASE_BUY:
        return A_DECLINE
    if phase == PHASE_AUCTION:
        return A_AUCTION_PASS
    if phase == PHASE_TRADE_RESPOND:
        return A_TRADE_REJECT
    return A_END_MANAGE  # MANAGE / LIQUIDATE: do nothing further


class ObsEncoder:
    """Encodes observations, valuations, and legal masks for a live game.

    Construct once, then :meth:`bind` to each fresh ``Game`` (the env rebuilds a
    game every episode; the GUI binds once per match).
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or RewardConfig()
        self.game = None
        self.ownable = []
        self.num_players = 0
        self._prop_index = {}
        self._groups = []
        self._group_of = {}
        self._land_freq = {}
        self._board_size = 40
        # Transient per-decision state, set by the owner (env / GUI decider).
        self._cur_trade = None            # offer judged in PHASE_TRADE_RESPOND
        self._traded_this_manage = set()  # trade targets tried this MANAGE phase

    def bind(self, game, ownable):
        """Attach to a live game. Must be called before any encoding."""
        self.game = game
        self.ownable = ownable
        self.num_players = len(game.players)
        self._prop_index = {id(p): i for i, p in enumerate(ownable)}
        self._groups = build_groups(ownable)
        self._group_of = {id(t): grp for grp in self._groups for t in grp}
        self._land_freq = load_landing_frequencies()
        self._board_size = game.board.length
        return self

    @property
    def obs_len(self):
        return observation_length(self.num_players)

    # -- Tile economics -----------------------------------------------------
    def _group_price(self, grp):
        """Total list price of every tile in a monopoly group."""
        return sum(t.price for t in grp)

    def _bid_value(self, player, prop):
        """``prop``'s value to ``player`` for auction bidding: its list price,
        boosted by the group's value when winning it completes ``player``'s own
        set, or (weighted) when it would complete an opponent's set -- so the
        ceiling reflects how pivotal the tile is, not just its sticker price."""
        value = float(prop.price)
        grp = self._group_of.get(id(prop))
        if grp is None:
            return value
        if self._completes_monopoly_for(player, prop):
            value += self.cfg.monopoly_bonus * self._group_price(grp)
        elif any(o is not player and not o.bankrupt
                 and self._completes_monopoly_for(o, prop)
                 for o in self.game.players):
            value += (self.cfg.denial_value_weight * self.cfg.monopoly_bonus
                      * self._group_price(grp))
        return value

    def _traffic(self, prop):
        """How often ``prop`` is landed on, as a multiple of an average tile
        (1.0 == an even 1/board share)."""
        uniform = 1.0 / self._board_size
        return self._land_freq.get(prop.pos, uniform) * self._board_size

    def _expected_income(self, prop):
        """A proxy for the rent ``prop`` will earn its owner: how often it is
        landed on (traffic vs an average tile) times its nominal rent."""
        return self._traffic(prop) * base_rent(prop)

    def _developed_rent(self, prop):
        """The rent ``prop`` collects *as developed right now* (unlike
        ``base_rent``'s nominal undeveloped figure): the street's rent table at
        its current house count (with the unimproved-monopoly double),
        railroads at the owner's count, utilities at the expected 7-pip roll.
        Mortgaged tiles collect nothing."""
        if prop.owner is None or prop.mortgaged:
            return 0.0
        if isinstance(prop, Utility):
            count = sum(isinstance(p, Utility) for p in prop.owner.properties)
            return (4.0 if count == 1 else 10.0) * 7.0
        return float(prop.get_rent(self.game, None))

    def _profits_per_turn(self):
        """Each player's expected net cash flow per full board round (see
        ``RewardConfig.profit_scale``): every live opponent passes the player's
        tiles once a round (income x live opponents) while the player passes
        every opponent-owned tile once itself (loss x 1). Flow per pass is
        landing traffic x developed rent. Returns a list aligned with
        ``game.players``; 0.0 for bankrupt seats."""
        g = self.game
        n = len(g.players)
        income = [0.0] * n
        total = 0.0
        for prop in self.ownable:
            owner = prop.owner
            if owner is None or owner.bankrupt:
                continue
            flow = self._traffic(prop) * self._developed_rent(prop)
            income[g.players.index(owner)] += flow
            total += flow
        n_live = sum(1 for p in g.players if not p.bankrupt)
        return [0.0 if p.bankrupt
                else (n_live - 1) * income[i] - (total - income[i])
                for i, p in enumerate(g.players)]

    def _buy_yield(self, prop):
        """Expected rent per dollar of purchase price: ``_expected_income`` /
        price. The cost-normalised "how cheap is this for what it earns" signal
        (``buy_yield.png``) -- high for the railroads/utilities and the cheap
        high-traffic streets, the best value acquisitions."""
        price = float(getattr(prop, "price", 0) or 0)
        return self._expected_income(prop) / price if price else 0.0

    def _dev_roi(self, prop):
        """Expected extra rent from a full hotel per dollar sunk into houses (a
        hotel is five house-purchases): traffic x (hotel rent - base rent) /
        (5 x house cost). The "best set to develop" signal (``development_roi``)
        -- peaks on the orange and light-blue groups. Zero for railroads and
        utilities, which never build."""
        if not isinstance(prop, StreetProperty):
            return 0.0
        build_cost = 5.0 * prop.house_cost()
        rent_gain = self._traffic(prop) * (prop.rent_table[-1]
                                           - prop.rent_table[0])
        return rent_gain / build_cost if build_cost else 0.0

    def _set_quality(self, grp):
        """Development-ROI multiplier for a colour group (~1.0 average), used to
        prize *completing* an efficient, cheap-to-develop set above a
        sticker-equal but sluggish one. The average dev-ROI over the group's
        streets, expressed against ``set_roi_ref`` and clamped to
        ``set_quality_clamp``: >1 for the orange/light-blue money groups, <1 for
        the pricey greens/dark-blue. Neutral (1.0) for railroad/utility groups,
        which never build."""
        rois = [self._dev_roi(t) for t in grp if isinstance(t, StreetProperty)]
        if not rois:
            return 1.0
        lo, hi = self.cfg.set_quality_clamp
        return max(lo, min(hi, (sum(rois) / len(rois)) / self.cfg.set_roi_ref))

    # -- Trade valuation ----------------------------------------------------
    def _prop_value(self, prop):
        """List-price value of a property, discounted for an unpaid mortgage."""
        if prop.mortgaged:
            return float(prop.price - prop.unmortgage_cost)
        return float(prop.price)

    def _trade_value(self, prop, owner):
        """Value of ``prop`` to ``owner`` for trading purposes.

        Richer than the raw list value: it adds the rent the tile is expected to
        earn (landing traffic * nominal rent, weighted by ``trade_income_weight``)
        and the group's monopoly value when the tile completes ``owner``'s own
        set, or the (weighted) blocking value when it is instead an *opponent*'s
        last-missing piece. Mirrors the bidding valuation in :meth:`_bid_value`,
        plus the traffic/rent term."""
        value = self._prop_value(prop)
        value += self.cfg.trade_income_weight * self._expected_income(prop)
        grp = self._group_of.get(id(prop))
        if grp is None:
            return value
        set_value = (self.cfg.monopoly_bonus * self._group_price(grp)
                     * self._set_quality(grp))
        if self._completes_monopoly_for(owner, prop):
            value += set_value
        elif any(o is not owner and not o.bankrupt
                 and self._completes_monopoly_for(o, prop)
                 for o in self.game.players):
            value += self.cfg.denial_value_weight * set_value
        return value

    def _completes_monopoly_for(self, player, target):
        """Whether acquiring ``target`` would complete a monopoly for ``player``
        (they already own every other tile in ``target``'s group)."""
        grp = self._group_of.get(id(target))
        if grp is None:
            return False
        return all(t.owner is player for t in grp if t is not target)

    def _can_propose_trade(self, initiator, target):
        """Whether ``initiator`` may propose a trade to acquire ``target``:
        a set-completing acquisition from a solvent opponent, of a tradeable
        tile, with something to hand over in return."""
        owner = target.owner
        if owner is None or owner is initiator or owner.bankrupt:
            return False
        if not self.game.can_trade_property(target):
            return False
        if not self._completes_monopoly_for(initiator, target):
            return False
        return self._choose_give_tile(initiator, owner, target) is not None

    def _choose_give_tile(self, initiator, partner, target):
        """Picks the tile ``initiator`` offers ``partner`` for ``target`` (or
        ``None``). Prefers a *spare* (its group's lone tile, so the trade breaks
        no progress) and, among the pool, the tile of least :meth:`_trade_value`
        to the initiator."""
        g = self.game
        target_group = self._group_of.get(id(target))

        def count(owner, group):
            return sum(1 for t in group if t.owner is owner)

        tradeable = [p for p in initiator.properties
                     if g.can_trade_property(p)
                     and self._group_of.get(id(p)) is not target_group]
        if not tradeable:
            return None
        spares = [p for p in tradeable
                  if count(initiator, self._group_of[id(p)]) == 1]
        pool = spares or tradeable
        return min(pool, key=lambda p: self._trade_value(p, initiator))

    def _balancing_cash(self, initiator, partner, give, receive):
        """Cash the initiator pays the partner to balance a trade, valued from
        the *partner*'s side with :meth:`_trade_value`, plus a premium for prying
        loose a set-completing tile. Never negative (only the initiator pays)."""
        recv_val = sum(self._trade_value(t, partner) for t in receive)
        give_val = sum(self._trade_value(t, partner) for t in give)
        grp = self._group_of.get(id(receive[0])) if receive else None
        set_price = sum(t.price for t in grp) if grp else 0
        premium = int(round(0.25 * set_price))
        return max(0, int(round(recv_val - give_val)) + premium)

    def _formula_trade_ok(self, partner, gain, lose, cash):
        """Baseline partner's accept rule: take the deal when it is non-negative
        by :meth:`_trade_value` from the partner's side."""
        value = float(cash)
        value += sum(self._trade_value(p, partner) for p in gain)
        value -= sum(self._trade_value(p, partner) for p in lose)
        return value >= 0

    # -- Liquidation reachability (used by masks / buy hooks) --------------
    def _has_liquidation_options(self, player):
        """Whether ``player`` has any house to sell or property to mortgage."""
        g = self.game
        for prop in player.properties:
            if isinstance(prop, StreetProperty) and prop.can_sell_house(g, player):
                return True
            if prop.can_mortgage(g, player):
                return True
        return False

    def _raisable_cash(self, player):
        """Cash ``player`` could raise beyond its balance by selling every house
        and mortgaging every property -- an upper bound used to decide whether a
        not-yet-affordable purchase is within reach."""
        total = 0
        for prop in player.properties:
            if prop.mortgaged:
                continue
            total += prop.mortgage_value
            if isinstance(prop, StreetProperty):
                total += prop.houses * (prop.house_cost() // 2)
        return total

    def _can_afford(self, player, price):
        """Whether ``player`` can pay ``price`` now or after liquidating."""
        return (player.balance >= price
                or player.balance + self._raisable_cash(player) >= price)

    # -- Legal-action masks -------------------------------------------------
    def _legal_mask(self, phase, prop, seat):
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        g = self.game
        player = g.players[seat]

        if phase == PHASE_JAIL:
            mask[A_ROLL_JAIL] = 1
            if player.balance >= 50:
                mask[A_PAY_JAIL] = 1
            if player.jail_cards:
                mask[A_USE_CARD] = 1
            return mask

        if phase == PHASE_BUY:
            mask[A_DECLINE] = 1
            # Offer BUY whenever the price is reachable -- either in cash now or
            # by mortgaging / selling first (the buy hook runs that liquidation
            # before finalizing), so the agent can afford a set-completing tile.
            if prop is not None and self._can_afford(player, prop.price):
                mask[A_BUY] = 1
            return mask

        if phase == PHASE_AUCTION:
            # Pass is always allowed; a bid bucket is legal when its dollar
            # amount is covered by cash (auctions are settled in cash only).
            mask[A_AUCTION_PASS] = 1
            if prop is not None:
                for k, frac in enumerate(BID_FRACTIONS):
                    if player.balance >= int(round(frac * prop.price)):
                        mask[A_AUCTION_BID + k] = 1
            return mask

        if phase == PHASE_TRADE_RESPOND:
            mask[A_TRADE_ACCEPT] = 1
            mask[A_TRADE_REJECT] = 1
            return mask

        # MANAGE and LIQUIDATE share the property-action masks.
        for i, p in enumerate(self.ownable):
            if p.owner is not player:
                continue
            if isinstance(p, StreetProperty):
                if phase == PHASE_MANAGE and p.can_build_house(g, player):
                    mask[A_BUILD + i] = 1
                # Selling houses is only offered during forced LIQUIDATE, for
                # the same reason as mortgaging: allowing it during voluntary
                # MANAGE let the agent sell<->rebuild houses on a monopoly.
                if phase == PHASE_LIQUIDATE and p.can_sell_house(g, player):
                    mask[A_SELL + i] = 1
            # Mortgaging is only offered during forced LIQUIDATE (raising cash
            # the player actually needs). Allowing it during voluntary MANAGE
            # let the agent mortgage-flip a property it just bought (and
            # oscillate mortgage<->unmortgage), so it is masked out there.
            if phase == PHASE_LIQUIDATE and p.can_mortgage(g, player):
                mask[A_MORTGAGE + i] = 1
            if phase == PHASE_MANAGE:
                if p.can_unmortgage(g, player):
                    mask[A_UNMORTGAGE + i] = 1

        # Trade proposals target tiles owned by *other* players: offer A_TRADE+i
        # for each opponent tile whose acquisition would complete a monopoly for
        # this player (and hasn't already been tried this MANAGE phase).
        if phase == PHASE_MANAGE:
            for i, p in enumerate(self.ownable):
                if i not in self._traded_this_manage \
                        and self._can_propose_trade(player, p):
                    mask[A_TRADE + i] = 1

        mask[A_END_MANAGE] = 1  # always allowed to stop
        return mask

    # -- Observation --------------------------------------------------------
    def _encode_obs(self, perspective, phase, prop):
        """Builds the flat float32 observation from ``perspective``'s view.

        Players are ordered starting at ``perspective`` (so the acting seat is
        always relative player 0) and property owners are encoded by relative
        seat. This makes the observation perspective-invariant, so one policy
        can play any seat -- the basis for self-play.
        """
        g = self.game
        n = self.num_players
        parts = []

        # Per-player block (acting seat first): balance, position, jail flag,
        # #cards, bankrupt, expected profit per turn (net rent flow).
        profits = self._profits_per_turn()
        for k in range(n):
            idx = (perspective + k) % n
            p = g.players[idx]
            parts.extend([
                p.balance / 1500.0,
                p.pos / 39.0,
                1.0 if p.in_jail else 0.0,
                float(len(p.jail_cards)),
                1.0 if p.bankrupt else 0.0,
                profits[idx] / self.cfg.profit_scale,
            ])

        # Per-property block: owner one-hot relative to perspective (slot 0 ==
        # "mine", slot n == unowned), mortgaged, houses.
        for p in self.ownable:
            owner_onehot = [0.0] * (n + 1)
            if p.owner is None:
                owner_onehot[n] = 1.0
            else:
                rel = (g.players.index(p.owner) - perspective) % n
                owner_onehot[rel] = 1.0
            parts.extend(owner_onehot)
            parts.append(1.0 if p.mortgaged else 0.0)
            parts.append(getattr(p, "houses", 0) / 5.0)

        # Phase one-hot and the context property index (BUY/AUCTION/LIQUIDATE),
        # or -1 when no single property is in play.
        phase_onehot = [0.0] * NUM_PHASES
        phase_onehot[phase] = 1.0
        parts.extend(phase_onehot)
        if prop is not None:
            parts.append(self._prop_index[id(prop)] / NUM_OWNABLE)
        else:
            parts.append(-1.0)

        # Trade context (PHASE_TRADE_RESPOND): the tile this player would
        # receive, the tile it would give up, and the net cash it gets, all from
        # its own perspective; -1/-1/0 outside a trade response.
        tc = self._cur_trade
        if phase == PHASE_TRADE_RESPOND and tc is not None:
            recv, give = tc["recv"], tc["give"]
            parts.append(self._prop_index[id(recv)] / NUM_OWNABLE
                         if recv is not None else -1.0)
            parts.append(self._prop_index[id(give)] / NUM_OWNABLE
                         if give is not None else -1.0)
            parts.append(tc["cash"] / 1500.0)
        else:
            parts.extend([-1.0, -1.0, 0.0])

        # Set-awareness. Two context flags for the property in play: does
        # acquiring it complete *my* set, and is it an opponent's last-missing
        # tile (completing it would finish *their* set)?
        me = g.players[perspective]
        opponents = [g.players[(perspective + k) % n] for k in range(1, n)]
        parts.append(1.0 if prop is not None
                     and self._completes_monopoly_for(me, prop) else 0.0)
        parts.append(1.0 if prop is not None and any(
            not o.bankrupt and self._completes_monopoly_for(o, prop)
            for o in opponents) else 0.0)

        # Per-group progress: how much of each monopoly group I own, and the
        # most any single opponent owns -- general set-awareness for bidding,
        # building, and trading.
        for grp in self._groups:
            size = len(grp)
            parts.append(sum(1 for t in grp if t.owner is me) / size)
            parts.append(max((sum(1 for t in grp if t.owner is o)
                              for o in opponents), default=0) / size)

        # Economics of the property in play (bought / auctioned / traded for):
        # price, a nominal rent, landing traffic (1.0 == even), and two
        # cost-normalised value signals -- buy yield (rent per $ of price) and
        # development ROI (rent per $ of houses); 0s when none is in play.
        econ = prop
        if econ is None and phase == PHASE_TRADE_RESPOND and tc is not None:
            econ = tc.get("recv")
        if econ is not None:
            parts.append(econ.price / 400.0)
            parts.append(base_rent(econ) / 50.0)
            parts.append(self._traffic(econ))
            parts.append(self._buy_yield(econ) * 5.0)   # ~0.25-0.95
            parts.append(self._dev_roi(econ))           # ~0-2.3, 0 for RR/util
        else:
            parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.asarray(parts, dtype=np.float32)
