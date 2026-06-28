"""Card classes for the Chance and Community Chest decks."""


class Card:
    """
    Base class for a single Chance or Community Chest card.

    Subclasses override execute() to apply their effect. The escape_jail flag
    marks Get Out of Jail Free cards, which the game keeps out of the deck
    (held by the player) until used.
    """

    def __init__(self, text, escape_jail=False):
        self.text = text
        self.escape_jail = escape_jail

    def execute(self, game, player):
        """Apply this card's effect to the player. Overridden per card type."""
        raise NotImplementedError


class MoneyCard(Card):
    """Collect from (positive amount) or pay (negative amount) the bank."""

    def __init__(self, text, amount):
        super().__init__(text)
        self.amount = amount

    def execute(self, game, player):
        # Collecting is a plain credit; paying may bankrupt the player.
        if self.amount >= 0:
            player.balance += self.amount
        else:
            game.pay(player, -self.amount)


class PerPlayerCard(Card):
    """
    Exchange money with every other player.

    A positive amount collects that sum from each opponent; a negative amount
    pays that sum to each opponent.
    """

    def __init__(self, text, amount):
        super().__init__(text)
        self.amount = amount

    def execute(self, game, player):
        for other in game.players:
            if other is player or other.bankrupt:
                continue
            if self.amount >= 0:
                # Collect from each opponent; an opponent may go bankrupt.
                game.pay(other, self.amount, player)
            else:
                # Pay each opponent; the card holder may go bankrupt.
                game.pay(player, -self.amount, other)


class GoToJailCard(Card):
    """Send the player directly to jail."""

    def execute(self, game, player):
        game.send_to_jail(player)


class AdvanceToCard(Card):
    """
    Advance forward to an absolute board position, collecting the GO salary if
    GO is passed, then resolve the destination tile.
    """

    def __init__(self, text, dest):
        super().__init__(text)
        self.dest = dest

    def execute(self, game, player):
        game.advance_to(player, self.dest)


class MoveCard(Card):
    """
    Move a relative number of steps (e.g. "Go back 3 spaces") and resolve the
    tile landed on. Does not award the GO salary.
    """

    def __init__(self, text, steps):
        super().__init__(text)
        self.steps = steps

    def execute(self, game, player):
        player.move(self.steps)
        game.resolve_tile(player)


class GetOutOfJailCard(Card):
    """
    Get Out of Jail Free. Held by the player until used, then returned to the
    deck it was drawn from. Has no immediate effect when drawn.
    """

    def __init__(self, text="Get Out of Jail Free"):
        super().__init__(text, escape_jail=True)
        # Set by the game when the card is kept, so it can be returned to the
        # correct pile (Chance or Community Chest) once used.
        self.source_deck = None

    def execute(self, game, player):
        pass
