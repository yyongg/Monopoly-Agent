"""Base class for a property"""

from models.tile import Tile

class Property(Tile):
    """
    Class for a Property. Inherits from Tile class.
    """

    def __init__(self, name, pos, price):
        super().__init__(name, pos)
        self.price = price
        self.owner = None

    def on_land(self, game, player):
        if self.owner is None:
            game.offer_purchase(self, player)

        elif self.owner != player:
            self.pay_rent(game, player)

    def buy(self, player):
        """
        Attempts to purchase property for input player.

        Args:
            player (type[Player]): Player who wishes to purchase property
        """

        if self.owner is not None:
            return False

        if player.balance < self.price:
            return False

        player.balance -= self.price
        self.transfer_ownership(player)
        return True

    def get_rent(self, game, player):
        """
        Get rent of the property.
        """

        raise NotImplementedError

    def pay_rent(self,game,player):
        """
        Transfers the rent of the property from the current player to owner.
        """

        rent = self.get_rent(game,player)
        player.balance -= rent
        self.owner.balance += rent

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
