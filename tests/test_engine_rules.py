"""Engine invariants the RL layer silently depends on."""

from tests.conftest import give, group_named


class TestBankruptcy:
    """The reward shaping prices a bankrupt player at *zero* net worth (cash +
    property). Both bankruptcy paths must actually leave them with nothing, or
    the relative-advantage potential jumps at an elimination."""

    def test_bankruptcy_to_the_bank_leaves_nothing(self, game, ownable):
        red, blue = game.players[0], game.players[1]
        street = next(t for t in ownable if hasattr(t, "houses"))
        give(red, street)
        street.houses = 3
        red.balance = 400

        game.declare_bankrupt(red)

        assert red.bankrupt
        assert red.balance == 0
        assert red.properties == []
        assert street.owner is None
        assert street.houses == 0
        assert not street.mortgaged
        assert blue.properties == []  # the bank took it, not another player

    def test_bankruptcy_to_a_creditor_hands_everything_over(self, game, ownable):
        red, blue = game.players[0], game.players[1]
        first, second = ownable[0], ownable[1]
        give(red, first, second)
        red.balance = 250

        game.declare_bankrupt(red, creditor=blue)

        assert red.bankrupt
        assert red.balance == 0
        assert red.properties == []          # transfer_ownership empties the list
        assert set(blue.properties) == {first, second}
        assert first.owner is blue and second.owner is blue


class TestExecuteTrade:
    """``execute_trade`` is the only way property changes hands in a trade, and
    it validates rather than raising -- so every guard needs a test."""

    def test_valid_trade_swaps_property_and_cash(self, game, ownable):
        red, blue = game.players[0], game.players[1]
        mine, theirs = ownable[0], ownable[5]
        give(red, mine)
        give(blue, theirs)

        assert game.execute_trade(red, blue, [mine], [theirs], 200) is True

        assert mine.owner is blue and theirs.owner is red
        assert red.balance == 1500 - 200
        assert blue.balance == 1500 + 200

    def test_negative_cash_means_the_partner_pays(self, game, ownable):
        red, blue = game.players[0], game.players[1]
        mine, theirs = ownable[0], ownable[5]
        give(red, mine)
        give(blue, theirs)

        assert game.execute_trade(red, blue, [mine], [theirs], -300) is True

        assert red.balance == 1500 + 300
        assert blue.balance == 1500 - 300

    def test_trade_the_payer_cannot_afford_is_refused_and_changes_nothing(
            self, game, ownable):
        red, blue = game.players[0], game.players[1]
        mine, theirs = ownable[0], ownable[5]
        give(red, mine)
        give(blue, theirs)
        red.balance = 100

        assert game.execute_trade(red, blue, [mine], [theirs], 500) is False

        assert mine.owner is red and theirs.owner is blue
        assert red.balance == 100 and blue.balance == 1500

    def test_developed_property_cannot_be_traded(self, game, encoder):
        red, blue = game.players[0], game.players[1]
        street = group_named(encoder, "brown")[0]
        other = group_named(encoder, "orange")[0]
        give(red, street)
        give(blue, other)
        street.houses = 1

        assert game.execute_trade(red, blue, [street], [other], 0) is False
        assert street.owner is red

    def test_trading_property_you_do_not_own_is_refused(self, game, ownable):
        red, blue, green = game.players[0], game.players[1], game.players[2]
        not_mine, theirs = ownable[0], ownable[5]
        give(green, not_mine)
        give(blue, theirs)

        assert game.execute_trade(red, blue, [not_mine], [theirs], 0) is False
        assert not_mine.owner is green
