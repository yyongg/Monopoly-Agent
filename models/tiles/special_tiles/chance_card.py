"""Class for chance card tile."""

from models.tile import Tile

class ChanceCard(Tile):
    """Chance tile for board. Inherits from Tile class."""

    def __init__(self,pos):
        super().__init__("Chance", pos)

    def on_land(self, game, player):
        game.chance_deck.draw(game, player)
