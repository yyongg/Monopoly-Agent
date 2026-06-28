"""Pygame hot-seat UI for the Monopoly game.

All players are controlled from this computer: on each turn the game pauses and
asks the human at the keyboard for every decision (build/sell, jail action, roll,
and whether to buy a property), mirroring the choices in the physical game.

The engine is unchanged; this layer drives it through the hooks it already
exposes:
    - per-player ``decide_purchase`` (overridden here to prompt the user),
    - the ``jail_choice`` argument to ``Game.step``,
    - ``Game.build_house`` / ``Game.sell_house`` for the management phase.
"""

import os
import sys

import pygame

from ui.board_layout import tile_center, token_offset
from models.tiles.properties.street_property import StreetProperty

# Window / board geometry.
WIDTH, HEIGHT = 1280, 800
BOARD_X, BOARD_Y, BOARD_PX = 20, 20, 760
SIDE_X = 800
SIDE_W = WIDTH - SIDE_X - 20
TOKEN_PX = 40

# Colors.
BG = (12, 70, 40)
PANEL = (244, 244, 238)
PANEL_EDGE = (180, 180, 170)
TEXT = (25, 25, 25)
MUTED = (110, 110, 110)
HIGHLIGHT = (245, 211, 90)
BUTTON = (66, 120, 170)
BUTTON_HOVER = (90, 150, 200)
BUTTON_TEXT = (255, 255, 255)

ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


class QuitGame(Exception):
    """Raised when the player closes the window, to unwind cleanly."""


class MonopolyApp:
    """Renders the board and drives a hot-seat game loop."""

    def __init__(self, game, auto=None):
        """
        Args:
            game (Game): A constructed game (players, board and decks set up).
            auto (callable): Optional headless responder ``(question, options)
                -> value`` used for testing; when set, prompts return its value
                instead of waiting for real input.
        """
        self.game = game
        self._auto = auto
        self.log = []

        pygame.init()
        pygame.display.set_caption("Monopoly")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.title_font = pygame.font.Font(None, 34)
        self.font = pygame.font.Font(None, 26)
        self.small_font = pygame.font.Font(None, 22)

        self.board_img = pygame.transform.smoothscale(
            pygame.image.load(os.path.join(ASSETS, "board.jpg")).convert(),
            (BOARD_PX, BOARD_PX),
        )
        self.tokens = {}
        for player in game.players:
            path = os.path.join(ASSETS, f"{player.name.lower()}_player.png")
            img = pygame.image.load(path).convert_alpha()
            self.tokens[player.name] = pygame.transform.smoothscale(
                img, (TOKEN_PX, TOKEN_PX)
            )

        # Route every player's purchase decision to a UI prompt.
        for player in game.players:
            player.decide_purchase = (
                lambda prop, _p=player: self._prompt_purchase(_p, prop)
            )

    # ----- logging -------------------------------------------------------

    def add_log(self, message):
        """Appends a line to the on-screen event log (keeping the latest)."""
        self.log.append(message)
        self.log = self.log[-9:]

    # ----- rendering -----------------------------------------------------

    def _draw_board(self):
        self.screen.blit(self.board_img, (BOARD_X, BOARD_Y))
        for index, player in enumerate(self.game.players):
            if player.bankrupt:
                continue
            cx, cy = tile_center(player.pos, BOARD_X, BOARD_Y, BOARD_PX)
            dx, dy = token_offset(index, BOARD_PX)
            token = self.tokens[player.name]
            rect = token.get_rect(center=(cx + dx, cy + dy))
            self.screen.blit(token, rect)

    def _draw_text(self, text, pos, font=None, color=TEXT):
        font = font or self.font
        self.screen.blit(font.render(text, True, color), pos)

    def _draw_players_panel(self):
        x, y = SIDE_X, 20
        self._draw_text("Players", (x, y), self.title_font)
        y += 44
        current = self.game.players[self.game.current_player]
        for player in self.game.players:
            box = pygame.Rect(x, y, SIDE_W, 66)
            color = HIGHLIGHT if player is current and not player.bankrupt else PANEL
            pygame.draw.rect(self.screen, color, box, border_radius=8)
            pygame.draw.rect(self.screen, PANEL_EDGE, box, 1, border_radius=8)

            token = self.tokens[player.name]
            self.screen.blit(
                pygame.transform.smoothscale(token, (30, 30)), (x + 8, y + 8)
            )
            if player.bankrupt:
                self._draw_text(f"{player.name} - BANKRUPT", (x + 48, y + 8),
                                self.font, MUTED)
            else:
                tile = self.game.board.get_tile(player.pos)
                self._draw_text(f"{player.name}   ${player.balance}",
                                (x + 48, y + 8), self.font)
                detail = f"{tile.name}  -  {len(player.properties)} props"
                self._draw_text(detail, (x + 48, y + 36), self.small_font, MUTED)
            y += 74
        return y

    def _draw_log(self, y):
        self._draw_text("Log", (SIDE_X, y), self.title_font)
        y += 40
        for line in self.log:
            self._draw_text(line, (SIDE_X, y), self.small_font, TEXT)
            y += 24

    def _draw_prompt(self, question, options, mouse):
        """Draws the question and option buttons; returns [(rect, value), ...]."""
        height = 70 + len(options) * 52
        box = pygame.Rect(SIDE_X, HEIGHT - height - 10, SIDE_W, height)
        pygame.draw.rect(self.screen, PANEL, box, border_radius=10)
        pygame.draw.rect(self.screen, PANEL_EDGE, box, 2, border_radius=10)

        # Wrap the question to the panel width.
        y = box.y + 12
        for line in self._wrap(question, self.font, SIDE_W - 24):
            self._draw_text(line, (box.x + 12, y), self.font)
            y += 28
        y += 6

        buttons = []
        for i, (label, value) in enumerate(options):
            rect = pygame.Rect(box.x + 12, y, SIDE_W - 24, 44)
            hover = rect.collidepoint(mouse) if mouse else False
            pygame.draw.rect(self.screen, BUTTON_HOVER if hover else BUTTON,
                             rect, border_radius=8)
            label_text = f"{i + 1}. {label}"
            surf = self.font.render(label_text, True, BUTTON_TEXT)
            self.screen.blit(surf, surf.get_rect(center=rect.center))
            buttons.append((rect, value))
            y += 52
        return buttons

    def _wrap(self, text, font, max_width):
        words, lines, line = text.split(), [], ""
        for word in words:
            trial = f"{line} {word}".strip()
            if font.size(trial)[0] <= max_width:
                line = trial
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines or [""]

    def _draw_scene(self, question=None, options=None, mouse=None):
        self.screen.fill(BG)
        self._draw_board()
        next_y = self._draw_players_panel()
        if question:
            buttons = self._draw_prompt(question, options, mouse)
        else:
            self._draw_log(next_y + 4)
            buttons = []
        return buttons

    def render(self):
        """Draws a single frame (no prompt). Used for smoke tests."""
        self._draw_scene()
        pygame.display.flip()

    # ----- input ---------------------------------------------------------

    def ask(self, question, options):
        """
        Shows a modal question with clickable / numbered options and blocks
        until one is chosen, returning its value.

        Args:
            question (str): Prompt text.
            options (list): List of ``(label, value)`` pairs.
        """
        if self._auto is not None:
            return self._auto(question, options)

        while True:
            mouse = pygame.mouse.get_pos()
            buttons = self._draw_scene(question, options, mouse)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise QuitGame
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, value in buttons:
                        if rect.collidepoint(event.pos):
                            return value
                if event.type == pygame.KEYDOWN:
                    index = event.key - pygame.K_1
                    if 0 <= index < len(options):
                        return options[index][1]
            self.clock.tick(30)

    # ----- decisions -----------------------------------------------------

    def _prompt_purchase(self, player, prop):
        """Purchase hook: asks the buyer whether to buy an unowned property."""
        if player.balance < prop.price:
            self.add_log(f"{player.name} can't afford {prop.name}")
            return False
        choice = self.ask(
            f"{player.name}: buy {prop.name} for ${prop.price}? "
            f"(balance ${player.balance})",
            [("Buy", True), ("Decline", False)],
        )
        self.add_log(
            f"{player.name} {'bought' if choice else 'declined'} {prop.name}"
        )
        return choice

    def _buildable(self, player):
        return [t for t in self.game.board.tiles
                if isinstance(t, StreetProperty)
                and t.can_build_house(self.game, player)]

    def _sellable(self, player):
        return [t for t in self.game.board.tiles
                if isinstance(t, StreetProperty)
                and t.can_sell_house(self.game, player)]

    def _build_flow(self, player):
        streets = self._buildable(player)
        options = [(f"{t.name} (${t.house_cost()})", t) for t in streets]
        options.append(("Cancel", None))
        street = self.ask(f"{player.name}: build a house on which street?",
                          options)
        if street is not None:
            self.game.build_house(street, player)
            self.add_log(f"{player.name} built on {street.name} "
                         f"(now {street.houses})")

    def _sell_flow(self, player):
        streets = self._sellable(player)
        options = [(f"{t.name} ({t.houses} houses)", t) for t in streets]
        options.append(("Cancel", None))
        street = self.ask(f"{player.name}: sell a house from which street?",
                          options)
        if street is not None:
            self.game.sell_house(street, player)
            self.add_log(f"{player.name} sold a house on {street.name}")

    def _take_turn(self, player):
        """
        Runs the management + action menu for the current player, returning the
        jail/roll choice to pass into ``Game.step``. Building and selling loop
        until the player picks an action that ends the menu.
        """
        while True:
            options = []
            if player.in_jail:
                if player.balance >= 50:
                    options.append(("Pay $50 to leave jail", "pay"))
                if player.jail_cards:
                    options.append(("Use Get Out of Jail card", "card"))
                options.append(("Roll for doubles", "roll"))
            else:
                options.append(("Roll dice", "roll"))
            if self._buildable(player):
                options.append(("Build a house", "build"))
            if self._sellable(player):
                options.append(("Sell a house", "sell"))

            location = "in jail" if player.in_jail else \
                self.game.board.get_tile(player.pos).name
            choice = self.ask(
                f"{player.name}'s turn ({location}, ${player.balance})", options)

            if choice == "build":
                self._build_flow(player)
            elif choice == "sell":
                self._sell_flow(player)
            else:
                return choice

    # ----- game loop -----------------------------------------------------

    def run(self, max_turns=10000):
        """Plays the game to completion, then shows the result screen."""
        try:
            turns = 0
            while not self.game.is_over() and turns < max_turns:
                player = self.game.players[self.game.current_player]
                start_pos, start_balance = player.pos, player.balance

                jail_choice = self._take_turn(player)
                self.game.step(jail_choice)

                self._log_turn(player, start_pos, start_balance)
                self._handoff()
                turns += 1

            self._show_result()
        except QuitGame:
            pass
        finally:
            pygame.quit()

    def _log_turn(self, player, start_pos, start_balance):
        tile = self.game.board.get_tile(player.pos)
        moved = "jailed" if player.in_jail else f"to {tile.name}"
        delta = player.balance - start_balance
        money = "" if delta == 0 else f" ({'+' if delta > 0 else ''}{delta})"
        if player.bankrupt:
            self.add_log(f"{player.name} went BANKRUPT")
        else:
            self.add_log(f"{player.name} rolled {self.game.roll}, {moved}{money}")

    def _handoff(self):
        if self.game.is_over():
            return
        nxt = self.game.players[self.game.current_player]
        self.ask(f"Pass the computer to {nxt.name}.", [("Continue", "ok")])

    def _show_result(self):
        winner = self.game.winner()
        text = f"{winner.name} wins!" if winner else "Game over."
        self.add_log(text)
        self.ask(text, [("Quit", "quit")])


def main():
    """Builds a fresh game and launches the UI."""
    # Imported here so the module also works when only the layout is needed.
    from engine.game import Game
    from models.board import Board
    from models.player import Player
    from data.decks import build_chance_deck, build_community_deck
    from data.board_tiles import build_board_tiles

    players = [Player("Red"), Player("Blue"), Player("Green"), Player("Yellow")]
    game = Game(players, Board(build_board_tiles()),
                build_chance_deck(), build_community_deck())
    MonopolyApp(game).run()
    sys.exit(0)


if __name__ == "__main__":
    main()
