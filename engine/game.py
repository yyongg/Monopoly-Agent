"""Create a new game"""

import random

class Game:
    """Class to define the game."""
    def __init__(self,players,board,chance_deck,community_deck):
        self.players = players
        self.board = board
        self.roll = 0
        self.last_dice = (0, 0)
        self.current_player = 0

        # Positions
        self.jail_position = 10
        self.go_salary = 200

        # Card Decks
        self.chance_deck = chance_deck
        self.community_deck = community_deck

        # Optional UI hook: called as on_card(pile_name, card) when a card is
        # drawn, before its effect is applied, so the UI can show the player
        # what they drew. Left as None for headless play.
        self.on_card = None

        # Optional UI hook: called as notify(message) for automatic events the
        # player should be told about (rent paid, tax paid, GO salary, etc.).
        # The UI binds this to an acknowledged "Continue" prompt; left as None
        # for headless play and the RL layer (which skips these prompts).
        self.notify = None

        # Optional UI hook: called as on_shortfall(payer, amount) when a player
        # owes more than they hold, before any forced bankruptcy, so they can
        # raise cash by selling houses / mortgaging. Left as None for headless
        # play and the RL layer (where liquidation is part of the policy).
        self.on_shortfall = None

    def announce(self, message):
        """
        Reports an automatic game event via the optional ``notify`` hook so the
        UI can inform the player. A no-op when no hook is set (headless / RL).

        Args:
            message (str): Human-readable description of what just happened.
        """
        if self.notify is not None:
            self.notify(message)

    def roll_dice(self):
        """
        Rolls two 6-sided dice, records them on ``last_dice`` (so the UI can
        show the result of any roll) and returns them.

        Returns:
            tuple: The two die values.
        """
        self.last_dice = (random.randint(1, 6), random.randint(1, 6))
        return self.last_dice

    def draw_chance(self, player):
        """Draws and resolves a card from the Chance pile."""
        self._draw_card(self.chance_deck, player, "Chance")

    def draw_community(self, player):
        """Draws and resolves a card from the Community Chest pile."""
        self._draw_card(self.community_deck, player, "Community Chest")

    def _draw_card(self, deck, player, pile):
        """
        Draws a card from `deck`, applies its effect, and returns it to `deck`
        unless it is a Get Out of Jail Free card.

        The `on_card` hook (if set) is called with the pile name and card before
        the effect is applied, so the UI can show the drawn card to the player.

        Get Out of Jail Free cards are kept out of the deck and held by the
        player until used; the source deck is recorded on the card so it can be
        returned to the correct pile (Chance or Community Chest) at that point.
        """
        card = deck.draw()

        if self.on_card is not None:
            self.on_card(pile, card)

        card.execute(self, player)

        if card.escape_jail:
            card.source_deck = deck
            player.jail_cards.append(card)
        else:
            deck.return_card(card)

    def resolve_tile(self, player):
        """Triggers the on-land action for the tile the player is standing on."""
        tile = self.board.get_tile(player.pos)
        tile.on_land(self, player)

    def offer_purchase(self, prop, player):
        """
        Offers an unowned property to the player who landed on it. The buy or
        decline decision is delegated to the player (the RL hook); the purchase
        itself is carried out by the property, which re-checks affordability.

        Args:
            prop (type[Property]): The unowned property being offered.
            player (type[Player]): The player who landed on it.

        Returns:
            bool: True if the property was bought, False otherwise.
        """
        if player.decide_purchase(prop):
            return prop.buy(player)
        return False

    def build_house(self, prop, player):
        """
        Entry point to build one house/hotel on a street the player owns. Rules
        (full color set, even building, cost, hotel cap) are enforced by the
        street tile.

        Args:
            prop (type[StreetProperty]): The street to build on.
            player (type[Player]): The owner building the house.

        Returns:
            bool: True if a house was built, False if the build was not allowed.
        """
        return prop.build_house(self, player)

    def sell_house(self, prop, player):
        """
        Entry point to sell one house/hotel back to the bank from a street the
        player owns. Even-selling rules are enforced by the street tile.

        Args:
            prop (type[StreetProperty]): The street to sell from.
            player (type[Player]): The owner selling the house.

        Returns:
            bool: True if a house was sold, False if the sale was not allowed.
        """
        return prop.sell_house(self, player)

    def mortgage_property(self, prop, player):
        """
        Entry point to mortgage a property the player owns, paying them its
        mortgage value. Rules (no buildings on the color group, not already
        mortgaged) are enforced by the property.

        Returns:
            bool: True if the property was mortgaged.
        """
        return prop.mortgage(self, player)

    def unmortgage_property(self, prop, player):
        """
        Entry point to lift the mortgage on a property the player owns, charging
        the mortgage value plus 10% interest.

        Returns:
            bool: True if the mortgage was lifted.
        """
        return prop.unmortgage(self, player)

    def can_trade_property(self, prop):
        """
        Returns whether a property may currently change hands in a trade. A
        property with houses standing anywhere in its color group can't be
        traded (the houses must be sold first); mortgaged properties may be
        traded and carry their mortgage to the new owner.
        """
        return not prop.has_buildings(self)

    def execute_trade(self, initiator, partner, give, receive, cash):
        """
        Swaps properties and cash between two players if the trade is valid.

        Args:
            initiator (Player): The player proposing the trade.
            partner (Player): The other player in the trade.
            give (list): Initiator's properties going to the partner.
            receive (list): Partner's properties coming to the initiator.
            cash (int): Net cash the initiator pays the partner. Negative means
                the partner pays the initiator.

        Returns:
            bool: True if the trade was valid and carried out, else False (no
                state is changed on failure).
        """
        # Validate ownership and that nothing carries buildings.
        if any(p.owner is not initiator for p in give):
            return False
        if any(p.owner is not partner for p in receive):
            return False
        if any(p.has_buildings(self) for p in list(give) + list(receive)):
            return False

        # The cash payer must be able to cover the net amount.
        if cash > 0 and initiator.balance < cash:
            return False
        if cash < 0 and partner.balance < -cash:
            return False

        initiator.balance -= cash
        partner.balance += cash
        for prop in give:
            prop.transfer_ownership(partner)
        for prop in receive:
            prop.transfer_ownership(initiator)
        return True

    def advance_to(self, player, dest):
        """
        Moves the player forward to absolute position `dest`, paying the GO
        salary if GO is passed, then resolves the destination tile.
        """
        # A destination behind the current square means GO was passed.
        if dest < player.pos:
            player.balance += self.go_salary
            self.announce(
                f"{player.name} passed GO and collected ${self.go_salary}.")
        player.pos = dest
        self.resolve_tile(player)

    def send_to_jail(self, player):
        """
        Sends a player to jail. Does not advance the turn; the caller (``step``
        for the headless sim, or the UI) is responsible for ending the turn.
        """
        player.pos = self.jail_position
        player.in_jail = True
        player.jail_turns = 0
        player.double_count = 0

    def roll_once(self, player):
        """
        Plays a single dice roll for a non-jailed player: applies the
        three-doubles-to-jail rule and moves the player, but does NOT resolve
        the destination tile or advance the turn. This lets the UI animate the
        move and stop between rolls (so doubles are re-rolled on demand rather
        than automatically).

        Args:
            player (type[Player]): The player rolling.

        Returns:
            tuple: ``(die1, die2, is_double, sent_to_jail)``. When sent_to_jail
                is True (a third double) the player did not move forward.
        """
        die1, die2 = self.roll_dice()
        is_double = die1 == die2

        if is_double:
            player.double_count += 1
            if player.double_count >= 3:
                self.send_to_jail(player)  # resets double_count
                return die1, die2, True, True

        self.roll = die1 + die2
        player.move(self.roll)
        return die1, die2, is_double, False

    def pay(self, payer, amount, payee=None):
        """
        Transfers `amount` from `payer`. If `payee` is given the money goes to
        them, otherwise it goes to the bank. A payer short of the amount is first
        given a chance to raise cash (via the ``on_shortfall`` hook); if they
        still cannot cover it, they go bankrupt.

        Args:
            payer (type[Player]): Player making the payment.
            amount (int): Amount owed.
            payee (type[Player]): Recipient, or None to pay the bank.
        """
        # Let the payer liquidate assets before the payment forces bankruptcy.
        if payer.balance < amount and self.on_shortfall is not None:
            self.on_shortfall(payer, amount)

        payer.balance -= amount
        if payee is not None:
            payee.balance += amount

        if payer.balance < 0:
            self.declare_bankrupt(payer)

    def declare_bankrupt(self, player):
        """
        Removes a player from play: returns their properties (and any houses) to
        the bank and their held Get Out of Jail Free cards to their decks.

        Note: this simplifies the real rule (assets to the creditor) by always
        returning assets to the bank, which is enough for a complete game.
        """
        player.bankrupt = True

        for prop in list(player.properties):
            prop.owner = None
            prop.mortgaged = False
            if hasattr(prop, "houses"):
                prop.houses = 0
        player.properties.clear()

        for card in list(player.jail_cards):
            card.source_deck.return_card(card)
        player.jail_cards.clear()

        player.balance = 0

    def active_players(self):
        """Returns the players still in the game (not bankrupt)."""
        return [p for p in self.players if not p.bankrupt]

    def is_over(self):
        """Returns whether the game has ended (one or zero players left)."""
        return len(self.active_players()) <= 1

    def winner(self):
        """Returns the sole surviving player, or None if the game is not over."""
        survivors = self.active_players()
        return survivors[0] if len(survivors) == 1 else None

    def advance_turn(self):
        """Passes play to the next player who is still in the game."""
        for _ in range(len(self.players)):
            self.current_player = (self.current_player + 1) % len(self.players)
            if not self.players[self.current_player].bankrupt:
                return

    def handle_jail_turn(self, player, choice="roll"):
        """
        Resolves a jailed player's turn according to a choice made *before* any
        dice are rolled. Paying, using a card, and rolling are mutually
        exclusive: a player cannot roll and then decide to pay or use a card.

        Args:
            player (type[Player]): The jailed player taking their turn.
            choice (str): The escape action, one of:
                "pay"  - pay the $50 fine to leave jail, then take a normal turn.
                "card" - spend a Get Out of Jail Free card to leave jail, then
                         take a normal turn.
                "roll" - roll for doubles. Doubles free the player and move them
                         by the roll (no bonus turn). After a second failed
                         attempt the player is released without moving and takes
                         a normal turn on their next turn.
                "pay"/"card" fall back to "roll" if the player cannot afford the
                fine or holds no card.

        Returns:
            str: One of
                "released" - left jail without moving (paid/used a card); the
                    caller should play out a normal turn now.
                "moved"    - left jail and already moved by the roll; the caller
                    should resolve the landed tile, with no bonus turn.
                "freed"    - released after failing to roll doubles twice; no
                    move this turn, a normal turn follows on the next turn.
                "jailed"   - still in jail; the turn is over.
        """
        # Pay or use a card up front, before rolling.
        if choice == "pay" and player.balance >= 50:
            player.balance -= 50
            player.in_jail = False
            player.jail_turns = 0
            return "released"

        if choice == "card" and player.jail_cards:
            # Spend a held card and return it to the pile it came from.
            card = player.jail_cards.pop()
            card.source_deck.return_card(card)
            player.in_jail = False
            player.jail_turns = 0
            return "released"

        # Otherwise roll to try for doubles.
        roll = self.roll_dice()

        # Doubles free the player and move them this turn (no bonus turn).
        if len(set(roll)) <= 1:
            player.in_jail = False
            player.jail_turns = 0
            self.roll = sum(roll)
            player.move(self.roll)
            return "moved"

        player.jail_turns += 1

        # Second failed attempt: released without moving; the player takes a
        # normal turn on their next turn.
        if player.jail_turns >= 2:
            player.in_jail = False
            player.jail_turns = 0
            return "freed"

        # First failed attempt: stay in jail, turn over.
        return "jailed"


    def step(self, jail_choice="roll"):
        """
        Plays out one Monopoly turn for one player.

        Args:
            jail_choice (str): Action for a jailed player, passed through to
                handle_jail_turn ("pay", "card" or "roll"). Ignored if the
                current player is not in jail.
        """

        # define current player
        player = self.players[self.current_player]

        # A jailed player resolves their jail turn first.
        if player.in_jail:
            result = self.handle_jail_turn(player, jail_choice)

            # Stuck in jail, or released without moving: turn is over.
            if result in ("jailed", "freed"):
                self.advance_turn()
                return

            # Escaped via the roll: already moved, resolve the tile, no bonus.
            if result == "moved":
                self.resolve_tile(player)
                self.advance_turn()
                return

            # result == "released": paid or used a card, now take a normal turn.

        # Keep rolling while doubles are rolled.
        while True:
            _, _, is_double, sent_to_jail = self.roll_once(player)

            # Three doubles in one turn sends the player straight to jail.
            if sent_to_jail:
                self.advance_turn()
                return

            # Resolve the tile landed on.
            self.resolve_tile(player)

            # A tile or card may have sent the player to jail, ending the turn.
            if player.in_jail:
                player.double_count = 0
                self.advance_turn()
                return

            # Going bankrupt this turn ends it immediately.
            if player.bankrupt:
                player.double_count = 0
                self.advance_turn()
                return

            # A non-double ends the turn; doubles grant another roll.
            if not is_double:
                break

        # reset double count and pass to the next player
        player.double_count = 0
        self.advance_turn()
