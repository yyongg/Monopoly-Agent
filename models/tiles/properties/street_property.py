"""Class for street properties"""

from models.tiles.property import Property

class StreetProperty(Property):
    """Inherits from Property class"""

    # Cost to add one house (or the hotel) on a street, by color group.
    HOUSE_COSTS = {
        "brown": 50,
        "light_blue": 50,
        "pink": 100,
        "orange": 100,
        "red": 150,
        "yellow": 150,
        "green": 200,
        "dark_blue": 200,
    }

    # houses == 5 represents a hotel (4 houses upgraded).
    MAX_HOUSES = 5

    def __init__(self, name, pos, price, rent_table, color):
        super().__init__(name, pos, price)
        self.rent_table = rent_table
        self.color = color
        self.houses = 0

    def color_group(self, game):
        """
        Returns every street sharing this tile's color group.

        Args:
            game (type[Game]): Game class, used to read the board layout.
        """
        return [
            tile for tile in game.board.tiles
            if isinstance(tile, StreetProperty) and tile.color == self.color
        ]

    def has_monopoly(self, game):
        """
        Returns whether this street's owner holds every street in its color
        group, which is the condition for double rent and building houses.

        Args:
            game (type[Game]): Game class, used to read the board layout.
        """
        return all(tile.owner is self.owner for tile in self.color_group(game))

    def house_cost(self):
        """Returns the price of one house (or hotel) for this color group."""
        return self.HOUSE_COSTS[self.color]

    def can_build_house(self, game, player):
        """
        Returns whether `player` may add one house/hotel to this street right
        now, enforcing the standard building rules:
            - the player owns this street and the full color set,
            - the street is not already at a hotel,
            - houses are built evenly (no street may get a house while another
              in the group has fewer), and
            - the player can afford the house.

        Args:
            game (type[Game]): Game class, used to read the board layout.
            player (type[Player]): Player wishing to build.
        """
        if self.owner is not player or not self.has_monopoly(game):
            return False

        if self.houses >= self.MAX_HOUSES:
            return False

        # Even-build rule: only build on a street tied for the group minimum.
        if self.houses != min(tile.houses for tile in self.color_group(game)):
            return False

        return player.balance >= self.house_cost()

    def build_house(self, game, player):
        """
        Adds one house (or upgrades to a hotel) on this street if the building
        rules allow it, charging the player the house cost.

        Args:
            game (type[Game]): Game class.
            player (type[Player]): Player building the house.

        Returns:
            bool: True if a house was built, False if the build was not allowed.
        """
        if not self.can_build_house(game, player):
            return False

        player.balance -= self.house_cost()
        self.houses += 1
        return True

    def can_sell_house(self, game, player):
        """
        Returns whether `player` may sell one house/hotel from this street,
        enforcing even selling: a street can only sell down from the group
        maximum so house counts stay within one of each other.

        Args:
            game (type[Game]): Game class, used to read the board layout.
            player (type[Player]): Player wishing to sell.
        """
        if self.owner is not player or self.houses <= 0:
            return False

        # Even-sell rule: only sell from a street tied for the group maximum.
        return self.houses == max(tile.houses for tile in self.color_group(game))

    def sell_house(self, game, player):
        """
        Sells one house (or downgrades the hotel) back to the bank for half the
        house cost if the selling rules allow it.

        Args:
            game (type[Game]): Game class.
            player (type[Player]): Player selling the house.

        Returns:
            bool: True if a house was sold, False if the sale was not allowed.
        """
        if not self.can_sell_house(game, player):
            return False

        player.balance += self.house_cost() // 2
        self.houses -= 1
        return True

    def get_rent(self, game, player):
        rent = self.rent_table[self.houses]

        # An unimproved street in a full color set collects double base rent.
        if self.houses == 0 and self.has_monopoly(game):
            rent *= 2

        return rent
