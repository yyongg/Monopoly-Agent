"""Factory that builds the standard 40-tile Monopoly board in order.

`build_board_tiles()` returns a list of tile instances indexed by board
position (0-39), ready to hand to `Board(tiles)`. Street rent tables follow the
standard US edition and are indexed by house count: [base, 1, 2, 3, 4, hotel].
"""

from models.tiles.special_tiles.go import Go
from models.tiles.special_tiles.jail import Jail
from models.tiles.special_tiles.go_jail import GoJail
from models.tiles.special_tiles.free_parking import FreeParking
from models.tiles.special_tiles.tax import Tax
from models.tiles.special_tiles.chance_card import ChanceCard
from models.tiles.special_tiles.community_chest import CommunityChest
from models.tiles.properties.street_property import StreetProperty
from models.tiles.properties.railroad import Railroad
from models.tiles.properties.utility import Utility


def build_board_tiles():
    """Returns the ordered list of the 40 standard Monopoly board tiles."""
    return [
        Go(0),
        StreetProperty("Mediterranean Avenue", 1, 60, [2, 10, 30, 90, 160, 250]),
        CommunityChest(2),
        StreetProperty("Baltic Avenue", 3, 60, [4, 20, 60, 180, 320, 450]),
        Tax("Income Tax", 4, 200),
        Railroad("Reading Railroad", 5, 200),
        StreetProperty("Oriental Avenue", 6, 100, [6, 30, 90, 270, 400, 550]),
        ChanceCard(7),
        StreetProperty("Vermont Avenue", 8, 100, [6, 30, 90, 270, 400, 550]),
        StreetProperty("Connecticut Avenue", 9, 120, [8, 40, 100, 300, 450, 600]),
        Jail(10),
        StreetProperty("St. Charles Place", 11, 140, [10, 50, 150, 450, 625, 750]),
        Utility("Electric Company", 12, 150),
        StreetProperty("States Avenue", 13, 140, [10, 50, 150, 450, 625, 750]),
        StreetProperty("Virginia Avenue", 14, 160, [12, 60, 180, 500, 700, 900]),
        Railroad("Pennsylvania Railroad", 15, 200),
        StreetProperty("St. James Place", 16, 180, [14, 70, 200, 550, 750, 950]),
        CommunityChest(17),
        StreetProperty("Tennessee Avenue", 18, 180, [14, 70, 200, 550, 750, 950]),
        StreetProperty("New York Avenue", 19, 200, [16, 80, 220, 600, 800, 1000]),
        FreeParking(20),
        StreetProperty("Kentucky Avenue", 21, 220, [18, 90, 250, 700, 875, 1050]),
        ChanceCard(22),
        StreetProperty("Indiana Avenue", 23, 220, [18, 90, 250, 700, 875, 1050]),
        StreetProperty("Illinois Avenue", 24, 240, [20, 100, 300, 750, 925, 1100]),
        Railroad("B&O Railroad", 25, 200),
        StreetProperty("Atlantic Avenue", 26, 260, [22, 110, 330, 800, 975, 1150]),
        StreetProperty("Ventnor Avenue", 27, 260, [22, 110, 330, 800, 975, 1150]),
        Utility("Water Works", 28, 150),
        StreetProperty("Marvin Gardens", 29, 280, [24, 120, 360, 850, 1025, 1200]),
        GoJail(30),
        StreetProperty("Pacific Avenue", 31, 300, [26, 130, 390, 900, 1100, 1275]),
        StreetProperty("North Carolina Avenue", 32, 300, [26, 130, 390, 900, 1100, 1275]),
        CommunityChest(33),
        StreetProperty("Pennsylvania Avenue", 34, 320, [28, 150, 450, 1000, 1200, 1400]),
        Railroad("Short Line", 35, 200),
        ChanceCard(36),
        StreetProperty("Park Place", 37, 350, [35, 175, 500, 1100, 1300, 1500]),
        Tax("Luxury Tax", 38, 100),
        StreetProperty("Boardwalk", 39, 400, [50, 200, 600, 1400, 1700, 2000]),
    ]
