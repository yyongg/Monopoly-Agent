"""Class for utility properties"""

from models.tiles.property import Property

class Utility(Property):
    """Inherits from Property class"""

    def get_rent(self, game, player):
        count = sum(isinstance(p, Utility) for p in self.owner.properties)
        return (4 if count == 1 else 10) * game.roll
