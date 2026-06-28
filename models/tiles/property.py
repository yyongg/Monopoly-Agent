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
        self.mortgaged = False

    def on_land(self, game, player):
        if self.owner is None:
            game.offer_purchase(self, player)

        # A mortgaged property collects no rent.
        elif self.owner != player and not self.mortgaged:
            self.pay_rent(game, player)

    @property
    def mortgage_value(self):
        """Cash the bank pays the owner to mortgage this property (half price)."""
        return self.price // 2

    @property
    def unmortgage_cost(self):
        """Cost to lift a mortgage: the mortgage value plus 10% interest."""
        return int(round(self.mortgage_value * 1.1))

    def has_buildings(self, game):
        """
        Returns whether buildings stand on this property (or, for a street, on
        any property in its color group), which blocks mortgaging. The base
        property has no buildings; StreetProperty overrides this.
        """
        return False

    def can_mortgage(self, game, player):
        """Returns whether `player` may mortgage this property right now."""
        return (self.owner is player and not self.mortgaged
                and not self.has_buildings(game))

    def mortgage(self, game, player):
        """
        Mortgages this property, paying the owner its mortgage value. Fails if
        it is already mortgaged or has buildings standing on its color group.

        Returns:
            bool: True if the property was mortgaged.
        """
        if not self.can_mortgage(game, player):
            return False
        self.mortgaged = True
        player.balance += self.mortgage_value
        return True

    def can_unmortgage(self, game, player):
        """Returns whether `player` can afford to lift this mortgage now."""
        return (self.owner is player and self.mortgaged
                and player.balance >= self.unmortgage_cost)

    def unmortgage(self, game, player):
        """
        Lifts the mortgage on this property, charging the unmortgage cost.

        Returns:
            bool: True if the mortgage was lifted.
        """
        if not self.can_unmortgage(game, player):
            return False
        player.balance -= self.unmortgage_cost
        self.mortgaged = False
        return True

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
        game.announce(
            f"{player.name} paid ${rent} rent to {self.owner.name} "
            f"for {self.name}.")
        game.pay(player, rent, self.owner)

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
