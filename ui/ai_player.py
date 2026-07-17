"""AI player for the GUI: drives decisions using a trained MaskablePPO model.

``GUIAIDecider`` wraps a model and exposes simple per-decision methods
(``jail_choice``, ``purchase_decision``, ``manage_loop``, ``liquidate_loop``)
that the GUI can call instead of showing a human prompt. Observation encoding,
tile/trade valuations, and legal masks all come from a shared
:class:`~engine.observation.ObsEncoder` (``self.encoder``) -- the same object the
training env uses -- so the model can never be fed inputs that drift from what it
was trained on.

Trades run through the encoder too: :meth:`GUIAIDecider._attempt_trade` asks
``ObsEncoder.build_offer`` for the offer (the same call the training env makes),
and :meth:`GUIAIDecider.evaluate_trade` puts a one-for-one offer to the model as
a ``PHASE_TRADE_RESPOND`` decision -- the same question training asks -- falling
back to ``ObsEncoder.accepts`` for offer shapes the observation cannot encode.
"""

from engine.constants import (
    PHASE_JAIL, PHASE_BUY, PHASE_MANAGE, PHASE_LIQUIDATE,
    PHASE_AUCTION, PHASE_TRADE_RESPOND,
    NUM_ACTIONS, NUM_OWNABLE, NUM_BID_LEVELS, NUM_TRADE_TIERS,
    BID_FRACTIONS, BID_CEILING_MULT,
    A_PAY_JAIL, A_USE_CARD, A_ROLL_JAIL,
    A_BUY, A_DECLINE, A_END_MANAGE,
    A_BUILD, A_SELL, A_MORTGAGE, A_UNMORTGAGE, A_TRADE,
    A_TRADE_ACCEPT, A_TRADE_REJECT, A_AUCTION_PASS, A_AUCTION_BID,
    decode_trade_action,
)
from engine.observation import ObsEncoder
from training.baselines import HeuristicAgent


class GUIAIDecider:
    """Drives one AI player's decisions in the GUI using a trained model.

    Call ``bind(game, ownable)`` once after the board is built, then use the
    public methods at each decision point during the turn loop.
    """

    def __init__(self, num_players, model, deterministic=False, log=None,
                 cfg=None):
        self.num_players = num_players
        self.model = model
        self.deterministic = deterministic
        # Callback used to report each AI move to the GUI game log. Defaults to
        # a no-op so the decider also works headless; ``MonopolyApp`` sets it to
        # ``add_log`` so the player can see what the AI did.
        self.log = log or (lambda message: None)
        self.game = None
        self.ownable = []
        # Observation encoder: the single shared implementation of obs
        # encoding, tile/trade valuations, and legal masks (also used by the
        # training env), so the GUI can never drift from what the policy was
        # trained on. ``cfg`` is the model's *training* RewardConfig (read from
        # its metadata sidecar by the app) rather than today's defaults, so the
        # economics it plays under are the ones it learned. The app sets
        # ``trade_arbiter`` to resolve trades this AI proposes to other players.
        self.encoder = ObsEncoder(cfg)
        self.trade_arbiter = None
        # Learned policy: when completing its own monopoly it spends *surplus*
        # cash (above the rent-sized reserve) to secure the contested tile,
        # overriding the conservative break-even cap. FP bots set this False to
        # stay the fixed benchmark (see FPBaselineDecider).
        self._overpay_sets = True

    def bind(self, game, ownable):
        """Attach to a live game instance. Must be called before any decisions."""
        self.game = game
        self.ownable = ownable
        self.encoder.bind(game, ownable)
        # This decider only ever decides for its own seat, so ``_overpay_sets``
        # applies to every seat it asks about. Keeps the trade mask gating
        # proposals on the same offer _attempt_trade will build (mirrors
        # MonopolyEnv._wire_deciders).
        self.encoder.overpay_seats = None if self._overpay_sets else set()

    # --- Public decision methods ---

    def jail_choice(self, player):
        """Returns "pay", "card", or "roll" for a jailed player."""
        seat = self.game.players.index(player)
        action = self._decide(seat, PHASE_JAIL)
        if action == A_PAY_JAIL:
            return "pay"
        if action == A_USE_CARD:
            return "card"
        return "roll"

    def purchase_decision(self, player, prop):
        """Returns True to buy ``prop``, False to decline.

        If the model chooses to buy but is short on cash (e.g. to complete a
        set), it first mortgages / sells to reach the price; it declines if it
        can't raise enough.
        """
        seat = self.game.players.index(player)
        action = self._decide(seat, PHASE_BUY, prop)
        if action != A_BUY:
            self.log(f"{player.name} [AI] declined {prop.name}.")
            return False
        if player.balance < prop.price:
            self.log(f"{player.name} [AI] is raising cash to buy {prop.name}.")
            self.liquidate_loop(player, prop.price)
            if player.balance < prop.price:
                self.log(f"{player.name} [AI] couldn't raise enough for "
                         f"{prop.name}; declined.")
                return False
        self.log(f"{player.name} [AI] bought {prop.name} for ${prop.price}.")
        return True

    def bid_choice(self, player, prop, min_bid=0):
        """Returns this AI's bid for one ascending-auction round (0 passes).

        Mirrors the env's bid hook: the model picks a bucket read as a valuation
        ceiling -- a fraction of the tile's *value to this bidder*
        (``_bid_value``), bounded by ``BID_CEILING_MULT`` * list price and by
        cash. The AI matches the round's ``min_bid`` while that ceiling covers
        it, otherwise it drops out.
        """
        seat = self.game.players.index(player)
        action = self._decide(seat, PHASE_AUCTION, prop, min_bid)
        k = action - A_AUCTION_BID
        if 0 <= k < NUM_BID_LEVELS:
            value = self.encoder._bid_value(player, prop)
            ceiling = min(int(round(BID_FRACTIONS[k] * value)),
                          int(round(BID_CEILING_MULT * prop.price)),
                          player.balance)
            if min_bid <= 0:
                return ceiling
            if ceiling >= min_bid:
                self.log(f"{player.name} [AI] bids ${min_bid} for {prop.name}.")
                return min_bid
        return 0

    def manage_loop(self, player):
        """Issues management actions until the model chooses END_MANAGE."""
        seat = self.game.players.index(player)
        # A trade target is proposed at most once per manage pass (re-proposing a
        # rejected offer changes nothing); mirrors the env's guard.
        self.encoder._traded_this_manage = set()
        for _ in range(50):
            action = self._decide(seat, PHASE_MANAGE)
            if action == A_END_MANAGE:
                return
            self._apply_manage_action(player, action)
            if player.bankrupt:
                return

    def liquidate_loop(self, player, amount):
        """Liquidates assets until ``player`` covers ``amount`` or gives up."""
        seat = self.game.players.index(player)
        for _ in range(50):
            if player.balance >= amount:
                return
            if not self.encoder._has_liquidation_options(player):
                return
            # The debt is what the decision is *about*: pass it so the policy
            # sees the same shortfall the training env shows it.
            action = self._decide(seat, PHASE_LIQUIDATE, None, amount)
            if action == A_END_MANAGE:
                return
            self._apply_manage_action(player, action)

    # --- Internal helpers ---

    def _decide(self, seat, phase, prop=None, amount=0):
        obs = self.encoder._encode_obs(seat, phase, prop, amount)
        mask = self.encoder._legal_mask(phase, prop, seat)
        action, _ = self.model.predict(
            obs, action_masks=mask.astype(bool), deterministic=self.deterministic)
        action = int(action)
        if not (0 <= action < NUM_ACTIONS and mask[action]):
            action = _safe_default(phase)
        return action

    def _apply_manage_action(self, player, action):
        g = self.game
        if A_BUILD <= action < A_BUILD + NUM_OWNABLE:
            prop = self.ownable[action - A_BUILD]
            if g.build_house(prop, player):
                self.log(f"{player.name} [AI] built on {prop.name} "
                         f"(now {prop.houses}).")
        elif A_SELL <= action < A_SELL + NUM_OWNABLE:
            prop = self.ownable[action - A_SELL]
            if g.sell_house(prop, player):
                self.log(f"{player.name} [AI] sold a house on {prop.name}.")
        elif A_MORTGAGE <= action < A_MORTGAGE + NUM_OWNABLE:
            prop = self.ownable[action - A_MORTGAGE]
            if g.mortgage_property(prop, player):
                self.log(f"{player.name} [AI] mortgaged {prop.name} "
                         f"for ${prop.mortgage_value}.")
        elif A_UNMORTGAGE <= action < A_UNMORTGAGE + NUM_OWNABLE:
            prop = self.ownable[action - A_UNMORTGAGE]
            cost = prop.unmortgage_cost
            if g.unmortgage_property(prop, player):
                self.log(f"{player.name} [AI] lifted the mortgage on "
                         f"{prop.name} for ${cost}.")
        elif A_TRADE <= action < A_TRADE + NUM_OWNABLE * NUM_TRADE_TIERS:
            i, tier = decode_trade_action(action)
            self.encoder._traded_this_manage.add(i)
            self._attempt_trade(player, self.ownable[i], tier)

    # --- Trade initiation (proposing a trade) ------------------------------

    def _attempt_trade(self, initiator, target, tier=1):
        """Offers a trade to acquire ``target`` at cash ``tier``.

        The offer is built by :meth:`ObsEncoder.build_offer` -- the same call the
        training env makes -- so the AI proposes in the GUI exactly the deal it
        was trained to propose. The partner then answers via the app's arbiter
        (a modal for a human, :meth:`evaluate_trade` for an AI), or via
        ``evaluate_trade`` directly when running headless.
        """
        partner = target.owner
        offer = self.encoder.build_offer(initiator, target, tier,
                                         overpay=self._overpay_sets)
        if offer is None:
            return
        give, receive, cash = offer

        if self.trade_arbiter is not None:
            accepted = self.trade_arbiter(
                initiator, partner, [give], receive, cash)
        else:  # headless fallback: ask the partner directly
            accepted, _ = self.evaluate_trade(
                partner, initiator, [give], receive, cash)
        if not accepted:
            self.log(f"{partner.name} declined {initiator.name} [AI]'s trade.")
            return
        if self.game.execute_trade(initiator, partner, [give], receive, cash):
            self.log(f"{initiator.name} [AI] traded {give.name} + ${cash} to "
                     f"{partner.name} for {target.name}.")

    # --- Trade evaluation (responding to a proposal) -----------------------

    def evaluate_trade(self, me, other, gain, lose, cash_delta):
        """Decides whether ``me`` accepts a trade, and what it is worth to them.

        From ``me``'s perspective the deal hands over the properties in ``lose``
        (to ``other``) and brings back the properties in ``gain`` plus
        ``cash_delta`` dollars.

        A **one-for-one** offer is the shape the policy was trained on, so it is
        put to the *model* as a ``PHASE_TRADE_RESPOND`` decision -- the same
        question, encoded the same way, that the training env asks. Anything else
        (a human stacking several tiles into one offer, which training never
        produced and the observation cannot even encode) falls back to the
        engine's dollar valuation, :meth:`ObsEncoder.accepts`.

        Returns ``(accepted: bool, value: float)`` -- the value is the engine's
        appraisal either way, so the GUI can show it in the log.

        NOTE the two rules are **not** interchangeable: on identical one-for-one
        swaps the model accepts ~83% where the formula accepts ~57%, and they
        disagree on nearly half of all offers. Since ``build_offer`` only ever
        emits one-for-one, self-play and ``validation.simulate`` exercise the
        model branch exclusively -- so the accept rate they report is *not* the
        rate a human sees when they stack tiles or buy with cash and land in the
        formula branch. Widening the trade action space (so the observation can
        encode the offers people actually make) is the real fix.
        """
        accepted, value = self.encoder.accepts(me, gain, lose, cash_delta)
        if value == float("-inf"):
            return False, value  # can't cover the cash it owes
        if self.model is None or len(gain) != 1 or len(lose) != 1:
            return accepted, value

        seat = self.game.players.index(me)
        self.encoder._cur_trade = {"recv": gain[0], "give": lose[0],
                                   "cash": cash_delta}
        try:
            # ``amount`` must be the trade cash, exactly as the env passes it
            # (rl_env._attempt_trade: ``decide(PHASE_TRADE_RESPOND, target,
            # cash)``). Letting it default to 0 leaves two of the 265 features --
            # the amount and its coverage -- describing a different offer than
            # the one on the table, which is a state the policy never saw in
            # training. Pinned by tests/test_trade_parity.py.
            action = self._decide(seat, PHASE_TRADE_RESPOND, lose[0],
                                  amount=cash_delta)
        finally:
            self.encoder._cur_trade = None
        return action == A_TRADE_ACCEPT, value

def _safe_default(phase):
    if phase == PHASE_JAIL:
        return A_ROLL_JAIL
    if phase == PHASE_BUY:
        return A_DECLINE
    if phase == PHASE_AUCTION:
        return A_AUCTION_PASS
    if phase == PHASE_TRADE_RESPOND:
        return A_TRADE_REJECT
    return A_END_MANAGE


class FPBaselineDecider(GUIAIDecider):
    """A GUI AI seat driven by an FP heuristic bot instead of the trained model.

    Lets the user watch the trained agent play against the FP-A/B/C trio: seat
    the agent in one chair and ``FPBaselineDecider`` bots in the others. It
    subclasses :class:`GUIAIDecider` so it reuses every decision method
    (``jail_choice`` / ``purchase_decision`` / ``bid_choice`` / ``manage_loop`` /
    ``liquidate_loop``) and the app wiring unchanged; only the per-decision
    *source* is swapped from ``model.predict`` to a
    :class:`~training.baselines.HeuristicAgent`, which already emits action ids
    in the same flat scheme and shares the :class:`ObsEncoder` valuations.
    """

    def __init__(self, num_players, priorities=None, name="FP", log=None,
                 cfg=None):
        # No model: _decide is overridden below to consult the FP bot, and
        # evaluate_trade falls through to the encoder, so model=None is safe.
        # deterministic is irrelevant (the bot is a fixed policy). ``cfg`` is the
        # match's economics -- in training the FP opponents share the env's one
        # encoder, so they judge trades under the same config the agent does.
        super().__init__(num_players, model=None, deterministic=True, log=log,
                         cfg=cfg)
        # Keep FP the fixed benchmark: no surplus-overpay, so its trade offers
        # stay at the conservative break-even/fair cap (matches training).
        self._overpay_sets = False
        self._bot = HeuristicAgent(priorities=priorities, name=name)

    def bind(self, game, ownable):
        super().bind(game, ownable)  # binds self.encoder
        # Share the SAME encoder as GUIAIDecider so per-manage state
        # (``_traded_this_manage``) and the offer being judged (``_cur_trade``)
        # stay consistent between manage_loop and the FP bot's decisions.
        self._bot.game = game
        self._bot.ownable = ownable
        self._bot.encoder = self.encoder

    def _decide(self, seat, phase, prop=None, amount=0):
        # Source the action from the FP bot instead of the model. ``offer`` feeds
        # trade-respond, which the GUI routes through evaluate_trade instead, so
        # it is moot here.
        mask = self.encoder._legal_mask(phase, prop, seat)
        action = int(self._bot.decide(seat, phase, prop, amount,
                                      mask.astype(bool),
                                      offer=self.encoder._cur_trade))
        if not (0 <= action < NUM_ACTIONS and mask[action]):
            action = _safe_default(phase)
        return action

    # ``evaluate_trade`` is inherited: with ``model is None`` it falls straight
    # through to ``ObsEncoder.accepts``, which *is* the FP accept rule
    # (:meth:`HeuristicAgent._decide_trade_respond` calls the same method), so
    # the GUI's FP bot judges an offer exactly as the training/eval one does.
