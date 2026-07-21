"""Observation encoding, tile/trade valuations, and legal-action masks.

``ObsEncoder`` is the single source of truth for everything the trained policy
sees and for the dollar-valued heuristics built on top of it. Both
:class:`engine.rl_env.MonopolyEnv` (training) and
``ui.ai_player.GUIAIDecider`` (play) hold one and delegate to it, so the RL
observation can never drift between training and the GUI -- the drift that used
to be guarded by a hand-maintained "mirrors the env" copy in the UI.

An encoder is bound to a live game with :meth:`ObsEncoder.bind`.

Trading is **not** part of the action space or the observation: it is resolved by
the heuristic in :mod:`engine.trade`, which is built on the valuations here
(:meth:`ObsEncoder._trade_value` and the shared :meth:`ObsEncoder.accepts` rule).
Group-level economics -- what a set is worth, and what a share of one is worth --
live in :class:`engine.valuation.SetValuer`, which this encoder owns and exposes
as ``self.sets`` so the reward reads the same numbers.
"""

import json
import os
from typing import NamedTuple

import numpy as np

from engine.config import RewardConfig
from engine.constants import (
    PHASE_JAIL, PHASE_BUY, PHASE_MANAGE, PHASE_LIQUIDATE,
    PHASE_AUCTION, NUM_PHASES,
    A_PAY_JAIL, A_USE_CARD, A_ROLL_JAIL, A_BUY, A_DECLINE, A_END_MANAGE,
    A_BUILD, A_SELL, A_MORTGAGE, A_UNMORTGAGE,
    A_AUCTION_PASS, A_AUCTION_BID,
    NUM_OWNABLE, NUM_GROUPS, NUM_ACTIONS, BID_FRACTIONS,
)
from engine.valuation import SetValuer
from models.tiles.properties.street_property import StreetProperty
from models.tiles.properties.railroad import Railroad
from models.tiles.properties.utility import Utility


# The landing-frequency table is *part of the observation definition* -- it scales
# every ``_traffic``-derived feature and valuation -- so the canonical copy is
# tracked static data next to the board itself, and ``runs/`` is only a fallback
# for a locally regenerated one. It used to live solely in gitignored ``runs/``,
# reached by a *relative* path, so a fresh clone (or simply running from another
# directory) silently swapped in a uniform prior: a different observation
# encoding than the shipped model was trained on, with no warning. Hence both the
# repo-root anchor below and the hard failure in ``load_landing_frequencies``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDING_FREQ_PATH = os.path.join(_REPO_ROOT, "data", "board_visits.json")
LANDING_FREQ_FALLBACKS = (os.path.join(_REPO_ROOT, "runs", "board_visits.json"),)
_LAND_FREQ_CACHE = None


def load_landing_frequencies(path=LANDING_FREQ_PATH, required=True):
    """Returns ``{board_pos: landing_share}`` from the saved visit table.

    Cached after first load. Raises when the table cannot be read and
    ``required`` (the default): a missing table does not merely disable a
    feature, it silently *changes the observation* the policy sees. Pass
    ``required=False`` to opt into the uniform prior (used when generating the
    table in the first place).
    """
    global _LAND_FREQ_CACHE
    if _LAND_FREQ_CACHE is not None:
        return _LAND_FREQ_CACHE
    tried = []
    for candidate in (path, *LANDING_FREQ_FALLBACKS):
        tried.append(candidate)
        try:
            with open(candidate) as f:
                data = json.load(f)
            freqs = {int(t["pos"]): float(t["frequency"]) for t in data["tiles"]}
        except (OSError, ValueError, KeyError):
            continue
        _LAND_FREQ_CACHE = freqs
        return freqs

    if required:
        raise FileNotFoundError(
            "No landing-frequency table found (looked in "
            + ", ".join(repr(p) for p in tried)
            + "). It defines part of the observation, so falling back to a "
              "uniform prior would silently change what the policy sees. "
              "Regenerate it with:\n"
              "    PYTHONPATH=. python -m validation.board_visits")
    _LAND_FREQ_CACHE = {}
    return _LAND_FREQ_CACHE


def squash(x):
    """Log-compresses an unbounded dollar-scaled feature, keeping its sign.

    Most features here are naturally bounded (one-hots, fractions of a group,
    a position on the board). The money ones are not: a late-game balance of
    $10,000 lands at ``balance / 1500 == 6.7``, and expected profit per turn can
    swing either way without limit. Feeding those raw into the policy net makes
    the money features dominate the tame ones and saturates a tanh unit, so they
    go through ``sign(x) * log1p(|x|)`` -- monotone, sign-preserving, and gentle
    on the tail (6.7 becomes 2.0), so "rich" and "very rich" stay distinguishable
    without drowning out the rest of the board.
    """
    return float(np.sign(x) * np.log1p(abs(x)))


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
            + 2               # context flags: completes mine / opp's set
            + NUM_GROUPS * 2  # per-group progress: mine / max-opp frac
            + 5               # context prop economics: price, rent, traffic,
                              # buy-yield, dev-ROI
            + 4               # race awareness: any set done, my/best-opp set
                              # count, am-I-first
            + 2               # the money on the table: the amount at stake
                              # (LIQUIDATE debt / AUCTION min bid) and whether
                              # the player can even cover it
            + 1)              # the clock: how far into the game we are


class TradeOffer(NamedTuple):
    """One proposed swap: the initiator hands over ``give`` plus ``cash`` and
    receives ``receive``. Both sides are lists -- the heuristic assembles
    multi-tile packages (:mod:`engine.trade`). ``cash`` is signed: negative means
    the initiator is *asking* the partner for money (a mutual set-for-set)."""

    give: list
    receive: list
    cash: int


def safe_default(phase):
    """The always-legal fallback action for ``phase`` (used to clamp an illegal
    or unavailable action instead of corrupting game state)."""
    if phase == PHASE_JAIL:
        return A_ROLL_JAIL
    if phase == PHASE_BUY:
        return A_DECLINE
    if phase == PHASE_AUCTION:
        return A_AUCTION_PASS
    return A_END_MANAGE  # MANAGE / LIQUIDATE: do nothing further


class ObsEncoder:
    """Encodes observations, valuations, and legal masks for a live game.

    Construct once, then :meth:`bind` to each fresh ``Game`` (the env rebuilds a
    game every episode; the GUI binds once per match).
    """

    def __init__(self, cfg=None, max_turns=1000):
        self.cfg = cfg or RewardConfig()
        # Episode horizon, used to scale the clock feature. The env passes its
        # own turn cap; the GUI (which has no cap) keeps the default, so a GUI
        # game reads the clock on the same scale the policy trained with.
        self.max_turns = max_turns
        self.game = None
        self.ownable = []
        self.num_players = 0
        self._prop_index = {}
        self._groups = []
        self._group_of = {}
        self._land_freq = {}
        self._board_size = 40
        # Group economics -- what a set, and a share of one, is worth. Shared
        # with the reward (engine.rewards reads ``encoder.sets``) so both price a
        # monopoly identically.
        self.sets = SetValuer(self.cfg)

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
        self.sets.bind(game, self._groups, self._traffic)
        return self

    @property
    def obs_len(self):
        return observation_length(self.num_players)

    # -- Tile economics -----------------------------------------------------
    def _group_price(self, grp):
        """Total list price of every tile in a monopoly group."""
        return self.sets.group_price(grp)

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

    # -- Trade valuation ----------------------------------------------------
    def _prop_value(self, prop):
        """List-price value of a property, discounted for an unpaid mortgage."""
        if prop.mortgaged:
            return float(prop.price - prop.unmortgage_cost)
        return float(prop.price)

    def _trade_value(self, prop, owner):
        """Value of ``prop`` to ``owner`` for trading purposes -- four terms:

        * its **list price** (discounted for an unpaid mortgage),
        * the **rent** it is expected to earn (landing traffic x nominal rent,
          weighted by ``trade_income_weight``),
        * its **marginal set value** to ``owner`` -- what ``owner``'s position in
          the tile's colour group is worth holding it, minus what that position is
          worth without it (:meth:`engine.valuation.SetValuer.marginal`),
        * its **blocking value** -- the same marginal, computed for whichever
          opponent it would help most, weighted by ``trade_denial_weight``.

        The marginal term is what killed the junk-for-monopoly exploit. It used to
        be a step: a set premium *only* when the tile was exactly one away from
        completing, and sticker price otherwise -- so an agent holding 2/3 of
        orange priced St. James at $227 and sold it for a brown and $200, while
        the same tile was worth $4,937 once the set was whole. Every rung of
        progress now carries real money, and completion is a 2x step rather than a
        21x cliff.

        ``trade_denial_weight`` must stay below 1.0: at 1.0 the tile is worth the
        same to holder and acquirer, the swap is zero-sum, and nothing can ever
        clear (see the knob's note in :mod:`engine.config`).

        Single-tile only -- :meth:`trade_delta` prices a package, because set
        value is **not** additive across tiles of the same group.
        """
        return (self._tile_value(prop, owner)
                + self.sets.marginal(prop, owner)
                + self.cfg.trade_denial_weight * self.sets.denial(prop, owner))

    def _tile_value(self, prop, owner):
        """The part of a tile's worth that *is* additive across a package: its
        list price (discounted for an unpaid mortgage) plus the rent it is
        expected to earn. Set value is handled per group, in
        :meth:`engine.valuation.SetValuer.swap_delta`."""
        return (self._prop_value(prop)
                + self.cfg.trade_income_weight * self._expected_income(prop))

    def _completes_monopoly_for(self, player, target):
        """Whether acquiring ``target`` would complete a monopoly for ``player``
        (they already own every other tile in ``target``'s group)."""
        grp = self._group_of.get(id(target))
        if grp is None:
            return False
        return all(t.owner is player for t in grp if t is not target)

    def _rent_exposure(self, player):
        """Expected rent ``player`` pays per board round: for each tile owned by
        a live opponent, landing traffic times the rent it collects *as developed
        now*. A proxy for how hard the board can hit ``player``'s cash -- the size
        its liquid cushion should cover so a bad landing does not bankrupt it.
        Shared by the solvency reward (:mod:`engine.rewards`) and the trade
        surplus cap below, so both size the cushion identically."""
        total = 0.0
        for prop in self.ownable:
            owner = prop.owner
            if owner is None or owner is player or owner.bankrupt:
                continue
            total += self._traffic(prop) * self._developed_rent(prop)
        return total

    def _cash_reserve(self, player):
        """Liquid cash ``player`` should keep against the board's rent threat:
        ``solvency_cushion_turns`` rounds of expected rent outflow. Cash *above*
        this reserve is surplus the agent can pour into completing a monopoly."""
        return self.cfg.solvency_cushion_turns * self._rent_exposure(player)

    # -- The trade rule -----------------------------------------------------
    # The single acceptance rule. Every trade decision in the project runs
    # through it -- the training env, the GUI's AI seats, and the FP baselines
    # alike -- and :mod:`engine.trade` builds every offer against it. It used to
    # be three separate accept rules kept in step by comment discipline, and they
    # had already drifted; worse, the learned policy bypassed it entirely and
    # accepted ~85% of everything put to it.

    def trade_delta(self, player, gain, lose, cash, partner=None):
        """Dollar value to ``player`` of receiving ``gain`` plus ``cash`` while
        giving up ``lose``. ``gain`` and ``lose`` are lists of any length.

        Two parts, because only one of them is additive:

        * per tile, its list price and expected rent (:meth:`_tile_value`);
        * per affected colour *group*, the change in set value -- what the whole
          package does to this player's position and to the best-placed
          opponent's (:meth:`engine.valuation.SetValuer.swap_delta`).

        Set value cannot be summed tile by tile: two tiles of one group are worth
        more together than apart, and a tile given while another of its group is
        taken is a wash. ``partner`` is who the tiles move to and from -- inferred
        from ``gain`` when not given.
        """
        if partner is None and gain:
            partner = gain[0].owner
        value = float(cash)
        value += sum(self._tile_value(p, player) for p in gain)
        value -= sum(self._tile_value(p, player) for p in lose)
        value += self.sets.swap_delta(player, gain, lose, partner=partner)
        return value

    def accepts(self, player, gain, lose, cash, partner=None):
        """Whether ``player`` takes this deal, and what it is worth to them.

        Take any swap that is non-negative by :meth:`trade_delta`, provided the
        cash owed is actually covered.

        Returns ``(accepted, value)``.
        """
        if cash < 0 and player.balance < -cash:
            return False, float("-inf")
        value = self.trade_delta(player, gain, lose, cash, partner=partner)
        return value >= 0, value

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

        # No trade band: the policy does not propose trades. The heuristic in
        # :mod:`engine.trade` runs its own round each MANAGE phase.
        mask[A_END_MANAGE] = 1  # always allowed to stop
        return mask

    # -- Observation --------------------------------------------------------
    def _encode_obs(self, perspective, phase, prop, amount=0):
        """Builds the flat float32 observation from ``perspective``'s view.

        Players are ordered starting at ``perspective`` (so the acting seat is
        always relative player 0) and property owners are encoded by relative
        seat. This makes the observation perspective-invariant, so one policy
        can play any seat -- the basis for self-play.

        ``amount`` is the money the decision actually turns on: the debt to be
        covered in ``PHASE_LIQUIDATE``, the minimum bid in ``PHASE_AUCTION``. It
        was previously handed to the heuristic baselines but *never encoded*, so
        the learned policy was choosing what to mortgage without knowing whether
        it owed $50 or $2,000.
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
                squash(p.balance / 1500.0),
                p.pos / 39.0,
                1.0 if p.in_jail else 0.0,
                float(len(p.jail_cards)),
                1.0 if p.bankrupt else 0.0,
                squash(profits[idx] / self.cfg.profit_scale),
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

        # Economics of the property in play (bought / auctioned): price, a
        # nominal rent, landing traffic (1.0 == even), and two cost-normalised
        # value signals -- buy yield (rent per $ of price) and development ROI
        # (rent per $ of houses); 0s when none is in play.
        econ = prop
        if econ is not None:
            parts.append(econ.price / 400.0)
            parts.append(base_rent(econ) / 50.0)
            parts.append(self._traffic(econ))
            parts.append(self._buy_yield(econ) * 5.0)   # ~0.25-0.95
            parts.append(self._dev_roi(econ))           # ~0-2.3, 0 for RR/util
        else:
            parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Race awareness: how the monopoly race stands, so the policy can value
        # being *first* to a set and denying opponents the same. "am I first" is
        # true when I hold a set and no opponent does yet.
        n_groups = len(self._groups)
        my_sets = sum(1 for grp in self._groups
                      if all(t.owner is me for t in grp))
        opp_sets = max((sum(1 for grp in self._groups
                            if all(t.owner is o for t in grp))
                        for o in opponents), default=0)
        parts.append(1.0 if (my_sets or opp_sets) else 0.0)
        parts.append(my_sets / n_groups)
        parts.append(opp_sets / n_groups)
        parts.append(1.0 if my_sets > 0 and opp_sets == 0 else 0.0)

        # The money on the table: how much this decision is *for* (the debt in
        # LIQUIDATE, the minimum bid in AUCTION), and what share of it the player
        # could actually raise. Without these the liquidation phase is a decision
        # about an invisible number.
        amount = float(amount or 0)
        parts.append(squash(amount / 1500.0))
        if amount > 0:
            reach = me.balance + self._raisable_cash(me)
            parts.append(min(1.0, reach / amount))
        else:
            parts.append(1.0)   # nothing owed: trivially covered

        # The clock. The episode is capped (``max_turns``) and the first-monopoly
        # reward decays with the turn count, so a time-blind state cannot predict
        # its own return.
        parts.append(min(1.0, g.turn / float(self.max_turns)))

        return np.asarray(parts, dtype=np.float32)
