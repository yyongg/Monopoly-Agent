"""Base class for a property"""

from tile import Tile

class Property(Tile):
    """
    Class for a Property. Inherits from Tile class.
    """

    def __init__(self, name, pos, price, rent_table):
        super().__init__(name, pos)

        self.houses = 0
        self.price = price
        self.rent_table = rent_table
        self.owner = None

    def on_land(self, game, player):
        if self.owner is None:
            game.offer_purchase(self, player)

        elif self.owner != player:
            self.pay_rent(player)

    def buy(self, player):
        """
        Attempts to purchase property for input player.

        Args:
            player (type[Player]): Player who wishes to purchase property
        """

        player.balance -= self.price
        self.transfer_ownership(player)

    def get_rent(self):
        """
        Gets the current rent of the property.
        """

        return self.rent_table[self.houses]

    def pay_rent(self,player):
        """
        Transfers the rent of the property from the current player to owner.
        """

        rent = self.get_rent
        player.balance -= rent
        self.owner.balance += rent

    def build_house(self, count):
        """
        Builds number of houses based on input.

        Args:
            count (int): Number of houses to build
        """

        self.houses += count

    def transfer_ownership(self, new_owner):
        """
        Transfers ownership of property from current owner to a new owner.

        Args:
            new_owner (type[Player]): Recipient of property ownership
        """

        if self.owner is not None:
            self.owner.properties.remove(self)

        self.owner = new_owner
        new_owner.properties.append(self)
