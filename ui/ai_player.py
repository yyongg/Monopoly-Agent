"""AI player for the GUI: drives decisions using a trained MaskablePPO model.

``GUIAIDecider`` wraps a model and exposes simple per-decision methods
(``jail_choice``, ``purchase_decision``, ``manage_loop``, ``liquidate_loop``)
that the GUI can call instead of showing a human prompt. Observation encoding,
tile/trade valuations, and legal masks all come from a shared
:class:`~engine.observation.ObsEncoder` (``self.encoder``) -- the same object the
training env uses -- so the model can never be fed inputs that drift from what it
was trained on.

The model does **not** decide trades -- nothing does. Trading is resolved by the
shared heuristic (:class:`engine.trade.TradeEngine`, held as ``self.trades``),
the same engine and the same valuations the training env uses, so an AI seat
negotiates identically in the GUI and in training. See :mod:`engine.trade` for
why trading left the policy.
"""

from engine.constants import (
    PHASE_JAIL, PHASE_BUY, PHASE_MANAGE, PHASE_LIQUIDATE,
    PHASE_AUCTION,
    NUM_ACTIONS, NUM_OWNABLE, NUM_BID_LEVELS,
    BID_FRACTIONS, BID_CEILING_MULT,
    A_PAY_JAIL, A_USE_CARD, A_ROLL_JAIL,
    A_BUY, A_DECLINE, A_END_MANAGE,
    A_BUILD, A_SELL, A_MORTGAGE, A_UNMORTGAGE,
    A_AUCTION_PASS, A_AUCTION_BID,
)
from engine.observation import ObsEncoder
from engine.trade import TradeEngine
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
        # Trades are decided by heuristic, not by the model -- the same engine
        # the training env runs. The app sets ``trade_arbiter`` to resolve trades
        # this AI proposes (a modal for a human partner, ``evaluate_trade`` for an
        # AI one).
        self.trades = TradeEngine(self.encoder, on_offer=self._log_trade)
        self.trade_arbiter = None

    def bind(self, game, ownable):
        """Attach to a live game instance. Must be called before any decisions."""
        self.game = game
        self.ownable = ownable
        self.encoder.bind(game, ownable)

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
        """Trades, then issues management actions until the model chooses
        END_MANAGE.

        Trading runs first and is not a model decision -- the heuristic proposes
        for this seat exactly as it does in training. Doing it first lets a set
        won by trade be built on in the same pass.
        """
        seat = self.game.players.index(player)
        self.trades.arbiter = self.trade_arbiter
        self.trades.run_round(player)
        if player.bankrupt:
            return
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

    # --- Trades ------------------------------------------------------------

    def _log_trade(self, initiator, partner, offer, executed, completes):
        """``TradeEngine.on_offer``: narrates this seat's proposals to the log."""
        give, receive, cash = offer
        gave = ", ".join(t.name for t in give) or "nothing"
        got = ", ".join(t.name for t in receive)
        if cash > 0:
            price = f" + ${cash}"
        elif cash < 0:
            price = f" for ${-cash} back"
        else:
            price = ""
        if not executed:
            self.log(f"{partner.name} declined {initiator.name} [AI]'s offer of "
                     f"{gave}{price} for {got}.")
            return
        note = " (completing a set)" if completes else ""
        self.log(f"{initiator.name} [AI] traded {gave}{price} to "
                 f"{partner.name} for {got}{note}.")

    def evaluate_trade(self, me, other, gain, lose, cash_delta):
        """Decides whether ``me`` accepts a trade, and what it is worth to them.

        From ``me``'s perspective the deal hands over the properties in ``lose``
        (to ``other``) and brings back the properties in ``gain`` plus
        ``cash_delta`` dollars. ``gain`` and ``lose`` are lists of any length, so
        a human stacking six tiles into one offer is priced as readily as a
        straight swap.

        This is the shared valuation and nothing else. The model used to get a
        say on one-for-one offers, and it was the whole problem: it accepted ~85%
        of everything, so a human could hand it junk for a monopoly and it would
        take the deal. Returns ``(accepted: bool, value: float)``; the GUI logs
        the value.
        """
        return self.encoder.accepts(me, gain, lose, cash_delta)


def _safe_default(phase):
    if phase == PHASE_JAIL:
        return A_ROLL_JAIL
    if phase == PHASE_BUY:
        return A_DECLINE
    if phase == PHASE_AUCTION:
        return A_AUCTION_PASS
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
        # No model: _decide is overridden below to consult the FP bot, so
        # model=None is safe. deterministic is irrelevant (the bot is a fixed
        # policy). ``cfg`` is the match's economics -- in training the FP
        # opponents share the env's one encoder, so they judge trades under the
        # same config the agent does.
        super().__init__(num_players, model=None, deterministic=True, log=log,
                         cfg=cfg)
        self._bot = HeuristicAgent(priorities=priorities, name=name)

    def bind(self, game, ownable):
        super().bind(game, ownable)  # binds self.encoder
        # Share the SAME encoder as GUIAIDecider, so the FP bot's valuations and
        # the seat's trade engine cannot disagree.
        self._bot.game = game
        self._bot.ownable = ownable
        self._bot.encoder = self.encoder

    def _decide(self, seat, phase, prop=None, amount=0):
        # Source the action from the FP bot instead of the model.
        mask = self.encoder._legal_mask(phase, prop, seat)
        action = int(self._bot.decide(seat, phase, prop, amount,
                                      mask.astype(bool)))
        if not (0 <= action < NUM_ACTIONS and mask[action]):
            action = _safe_default(phase)
        return action

    # ``manage_loop`` / ``evaluate_trade`` are inherited unchanged: trading is
    # the shared heuristic for every seat, so the GUI's FP bot negotiates exactly
    # as the learned agent and the training env do.
