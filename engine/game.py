"""Create a new game"""

import random

class Game:
    """Class to define the game."""
    def __init__(self,players,board,chance_deck,community_deck):
        self.players = players
        self.board = board
        self.roll = 0
        self.current_player = 0

        # Positions
        self.jail_position = 10
        self.go_salary = 200

        # Card Decks
        self.chance_deck = chance_deck
        self.community_deck = community_deck

    def roll_dice(self):
        """
        Rolls two 6-sided die and returns the results.
        
        Returns:
            tuple: Results from the two rolls
        """
        return (random.randint(1,6), random.randint(1,6))

    def draw_chance(self, player):
        """Draws and resolves a card from the Chance pile."""
        self._draw_card(self.chance_deck, player)

    def draw_community(self, player):
        """Draws and resolves a card from the Community Chest pile."""
        self._draw_card(self.community_deck, player)

    def _draw_card(self, deck, player):
        """
        Draws a card from `deck`, applies its effect, and returns it to `deck`
        unless it is a Get Out of Jail Free card.

        Get Out of Jail Free cards are kept out of the deck and held by the
        player until used; the source deck is recorded on the card so it can be
        returned to the correct pile (Chance or Community Chest) at that point.
        """
        card = deck.draw()
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

    def advance_to(self, player, dest):
        """
        Moves the player forward to absolute position `dest`, paying the GO
        salary if GO is passed, then resolves the destination tile.
        """
        # A destination behind the current square means GO was passed.
        if dest < player.pos:
            player.balance += self.go_salary
        player.pos = dest
        self.resolve_tile(player)

    def send_to_jail(self, player):
        """Sends player to jail."""

        player.pos = self.jail_position
        player.in_jail = True
        player.jail_turns = 0
        self.current_player = (self.current_player + 1) % len(self.players)

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
                self.current_player = (self.current_player + 1) % len(self.players)
                return

            # Escaped via the roll: already moved, resolve the tile, no bonus.
            if result == "moved":
                self.resolve_tile(player)
                self.current_player = (self.current_player + 1) % len(self.players)
                return

            # result == "released": paid or used a card, now take a normal turn.

        # Keep rolling while doubles are rolled.
        while True:
            roll = self.roll_dice()
            is_double = len(set(roll)) <= 1

            # Three doubles in one turn sends the player straight to jail.
            if is_double:
                player.double_count += 1
                if player.double_count >= 3:
                    player.double_count = 0
                    self.send_to_jail(player)  # advances current_player
                    return

            # Move and resolve the tile landed on.
            self.roll = sum(roll)
            player.move(self.roll)
            self.resolve_tile(player)

            # A tile or card may have sent the player to jail, ending the turn.
            # send_to_jail already advanced current_player, so just stop here.
            if player.in_jail:
                player.double_count = 0
                return

            # A non-double ends the turn; doubles grant another roll.
            if not is_double:
                break

        # reset player double count
        player.double_count = 0

        # change action to next player
        self.current_player = (self.current_player + 1) % len(self.players)
