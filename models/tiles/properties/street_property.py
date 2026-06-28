"""Class for street properties"""

from models.tiles.property import Property

class StreetProperty(Property):
    """Inherits from Property class"""

    def __init__(self, name, pos, price, rent_table, color):
        super().__init__(name, pos, price)
        self.rent_table = rent_table
        self.color = color
        self.houses = 0

    def get_rent(self,game,player):
        return self.rent_table[self.houses]
