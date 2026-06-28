"""Class for railroad properties"""

from models.tiles.property import Property

class Railroad(Property):
    """Inherits from Property class"""

    def get_rent(self, game, player):
        railroad_count = sum(isinstance(p, Railroad) for p in self.owner.properties)
        return 25 * 2**railroad_count
