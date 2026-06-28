"""Class for go to tax tile."""

from models.tile import Tile

class Tax(Tile):
    """Tax tile for board. Inherits from Tile class."""

    def __init__(self, name, pos, amount):
        super().__init__(name, pos)
        self.amount = amount

    def on_land(self, game, player):
        game.announce(f"{player.name} paid ${self.amount} in {self.name}.")
        game.pay(player, self.amount)
