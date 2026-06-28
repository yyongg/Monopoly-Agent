"""Pygame hot-seat UI for the Monopoly game.

All players are controlled from this computer. On each turn the game pauses and
asks the human for every decision (build/sell, jail action, roll, and whether to
buy a property). Dice results are shown, movement slides tile by tile, and
doubles stop the turn to ask the player to roll again rather than re-rolling
automatically.

The engine is driven through the hooks it exposes:
    - per-player ``decide_purchase`` (overridden here to prompt the user),
    - ``Game.handle_jail_turn`` for a single jail attempt,
    - ``Game.roll_once`` for one dice roll at a time (so the UI controls
      re-rolls and animation),
    - ``Game.build_house`` / ``Game.sell_house`` for the management phase.
"""

import math
import os
import random
import time

import pygame

from ui.board_layout import (tile_center, tile_rect, token_offset,
                             interior_offset, TILE_FRAC)
from models.tiles.properties.street_property import StreetProperty
from models.tiles.properties.railroad import Railroad
from models.tiles.properties.utility import Utility
from models.tiles.property import Property

# Window / board geometry.
WIDTH, HEIGHT = 1500, 950
BOARD_X, BOARD_Y, BOARD_PX = 20, 20, 880
SIDE_X = 920
SIDE_W = WIDTH - SIDE_X - 20
TOKEN_PX = 40

# Palette.
BG = (11, 76, 47)
PANEL = (247, 246, 241)
PANEL_ALT = (236, 235, 228)
EDGE = (206, 205, 196)
INK = (34, 34, 34)
MUTED = (122, 122, 122)
# Light tones for text drawn directly on the dark green felt (with a shadow),
# so headings, the log and inventory stay legible against the background.
FELT_INK = (245, 243, 233)
FELT_SUB = (197, 214, 197)
ACCENT = (212, 175, 55)
BTN = (58, 110, 165)
BTN_HOVER = (78, 138, 197)
BTN_INK = (255, 255, 255)
HOUSE_GREEN = (38, 158, 70)
HOTEL_RED = (200, 62, 55)
DIE_FACE = (250, 250, 248)
PIP = (32, 32, 32)

# Per-player token / marker colors, keyed by player name.
PLAYER_COLORS = {
    "Red": (211, 47, 47),
    "Blue": (33, 99, 199),
    "Green": (39, 158, 70),
    "Yellow": (240, 196, 32),
}
DEFAULT_COLOR = (120, 120, 120)


def player_color(name):
    """Returns the marker color for a player name (gray if unknown)."""
    return PLAYER_COLORS.get(name, DEFAULT_COLOR)

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
                -> value`` for testing; when set, prompts return its value and
                animations are skipped.
        """
        self.game = game
        self._auto = auto
        self.log = []
        self.selected = None          # index of player whose inventory is open
        self.roll_display = None      # {"name", "dice"} for the dice panel
        self.vpos = {p.name: float(p.pos) for p in game.players}
        self.hop = {p.name: 0.0 for p in game.players}  # token landing bounce

        pygame.init()
        pygame.display.set_caption("Monopoly")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.f_title = self._font(30, bold=True)
        self.f_head = self._font(24, bold=True)
        self.f_body = self._font(22)
        self.f_small = self._font(19)

        self.board_img = pygame.transform.smoothscale(
            pygame.image.load(os.path.join(ASSETS, "board.jpg")).convert(),
            (BOARD_PX, BOARD_PX),
        )
        self.tokens = {}
        for player in game.players:
            path = os.path.join(ASSETS, f"{player.name.lower()}_player.png")
            img = pygame.image.load(path).convert_alpha()
            self.tokens[player.name] = pygame.transform.smoothscale(
                img, (TOKEN_PX, TOKEN_PX))

        for player in game.players:
            player.decide_purchase = (
                lambda prop, _p=player: self._prompt_purchase(_p, prop))

        # Show drawn Chance / Community Chest cards to the player.
        game.on_card = self._show_card
        # Inform the player of automatic events (rent, tax, GO salary, ...).
        game.notify = self._inform

    @staticmethod
    def _font(size, bold=False):
        return pygame.font.SysFont("dejavusans,arial,helvetica", size, bold=bold)

    # ----- small drawing helpers ----------------------------------------

    def _text(self, text, pos, font=None, color=INK, shadow=False):
        font = font or self.f_body
        if shadow:
            sh = font.render(text, True, (0, 0, 0))
            self.screen.blit(sh, (pos[0] + 1, pos[1] + 2))
        self.screen.blit(font.render(text, True, color), pos)

    def _panel(self, rect, fill=PANEL):
        pygame.draw.rect(self.screen, fill, rect, border_radius=10)
        pygame.draw.rect(self.screen, EDGE, rect, 1, border_radius=10)

    def add_log(self, message):
        self.log.append(message)
        self.log = self.log[-40:]

    # ----- board ---------------------------------------------------------

    def _token_center(self, vpos):
        base = math.floor(vpos)
        frac = vpos - base
        x0, y0 = tile_center(base % 40, BOARD_X, BOARD_Y, BOARD_PX)
        x1, y1 = tile_center((base + 1) % 40, BOARD_X, BOARD_Y, BOARD_PX)
        return (x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac)

    def _draw_house_indicators(self):
        tsize = TILE_FRAC * BOARD_PX
        for tile in self.game.board.tiles:
            if not isinstance(tile, StreetProperty) or tile.houses == 0:
                continue
            cx, cy = tile_center(tile.pos, BOARD_X, BOARD_Y, BOARD_PX)
            ox, oy = interior_offset(tile.pos)
            bx, by = cx + ox * tsize * 0.32, cy + oy * tsize * 0.32
            px, py = -oy, ox  # unit vector along the tile edge

            if tile.houses >= 5:  # hotel
                rect = pygame.Rect(0, 0, 18, 13)
                rect.center = (bx, by)
                pygame.draw.rect(self.screen, HOTEL_RED, rect, border_radius=2)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1, border_radius=2)
            else:
                size, gap = 10, 13
                start = -(tile.houses - 1) / 2
                for i in range(tile.houses):
                    off = (start + i) * gap
                    rect = pygame.Rect(0, 0, size, size)
                    rect.center = (bx + px * off, by + py * off)
                    pygame.draw.rect(self.screen, HOUSE_GREEN, rect, border_radius=2)
                    pygame.draw.rect(self.screen, (0, 0, 0), rect, 1, border_radius=2)

    def _draw_ownership(self):
        """Draws a small colored dot on each owned property in its owner's
        color, set toward the outer edge so it doesn't sit under the tokens.
        Mortgaged properties show as a hollow dot."""
        tsize = TILE_FRAC * BOARD_PX
        for tile in self.game.board.tiles:
            if not isinstance(tile, Property) or tile.owner is None:
                continue
            cx, cy = tile_center(tile.pos, BOARD_X, BOARD_Y, BOARD_PX)
            ox, oy = interior_offset(tile.pos)        # toward interior
            mx, my = int(cx - ox * tsize * 0.34), int(cy - oy * tsize * 0.34)
            color = player_color(tile.owner.name)
            pygame.draw.circle(self.screen, (255, 255, 255), (mx, my), 7)
            if tile.mortgaged:
                # Hollow ring marks a mortgaged property.
                pygame.draw.circle(self.screen, color, (mx, my), 6, 2)
            else:
                pygame.draw.circle(self.screen, color, (mx, my), 6)
            pygame.draw.circle(self.screen, (0, 0, 0), (mx, my), 7, 1)

    def _draw_current_highlight(self):
        """Pulsing gold outline around the tile the active player occupies."""
        player = self.game.players[self.game.current_player]
        if player.bankrupt:
            return
        x, y, w, h = tile_rect(player.pos, BOARD_X, BOARD_Y, BOARD_PX)
        pulse = 0.5 + 0.5 * math.sin(time.time() * 5.0)
        thick = 2 + int(round(pulse * 2))
        rect = pygame.Rect(x + 1, y + 1, w - 2, h - 2)
        pygame.draw.rect(self.screen, ACCENT, rect, thick, border_radius=4)

    def _draw_board(self):
        self.screen.blit(self.board_img, (BOARD_X, BOARD_Y))
        self._draw_ownership()
        self._draw_house_indicators()
        self._draw_current_highlight()
        current = self.game.players[self.game.current_player]
        for index, player in enumerate(self.game.players):
            if player.bankrupt:
                continue
            cx, cy = self._token_center(self.vpos[player.name])
            dx, dy = token_offset(index, BOARD_PX)
            tx, ty = int(cx + dx), int(cy + dy - self.hop.get(player.name, 0.0))
            # Glow ring under the active player's token so it's easy to find.
            if player is current:
                pygame.draw.circle(self.screen, ACCENT, (tx, ty),
                                   TOKEN_PX // 2 + 4, 3)
            token = self.tokens[player.name]
            self.screen.blit(token, token.get_rect(center=(tx, ty)))

    # ----- dice ----------------------------------------------------------

    def _draw_die(self, x, y, size, value):
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, DIE_FACE, rect, border_radius=8)
        pygame.draw.rect(self.screen, EDGE, rect, 1, border_radius=8)
        r = max(3, size // 9)
        a, b, c = x + size * 0.27, x + size * 0.5, x + size * 0.73
        d, e, f = y + size * 0.27, y + size * 0.5, y + size * 0.73
        layout = {
            1: [(b, e)],
            2: [(a, d), (c, f)],
            3: [(a, d), (b, e), (c, f)],
            4: [(a, d), (c, d), (a, f), (c, f)],
            5: [(a, d), (c, d), (b, e), (a, f), (c, f)],
            6: [(a, d), (c, d), (a, e), (c, e), (a, f), (c, f)],
        }
        for px, py in layout.get(value, []):
            pygame.draw.circle(self.screen, PIP, (int(px), int(py)), r)

    def _draw_dice_panel(self, y):
        box = pygame.Rect(SIDE_X, y, SIDE_W, 96)
        self._panel(box)
        if not self.roll_display:
            self._text("Roll the dice to begin.", (box.x + 16, box.y + 36),
                       self.f_body, MUTED)
            return box.bottom
        d1, d2 = self.roll_display["dice"]
        self._draw_die(box.x + 16, box.y + 16, 64, d1)
        self._draw_die(box.x + 92, box.y + 16, 64, d2)
        name = self.roll_display["name"]
        self._text(f"{name} rolled", (box.x + 176, box.y + 24), self.f_body, INK)
        self._text(f"{d1} + {d2} = {d1 + d2}", (box.x + 176, box.y + 52),
                   self.f_head, INK)
        return box.bottom

    # ----- players & inventory ------------------------------------------

    def _draw_players(self, y):
        self._text("Players", (SIDE_X, y), self.f_title, FELT_INK, shadow=True)
        y += 42
        current = self.game.players[self.game.current_player]
        rects = []
        for index, player in enumerate(self.game.players):
            box = pygame.Rect(SIDE_X, y, SIDE_W, 56)
            selected = self.selected == index
            fill = ACCENT if player is current and not player.bankrupt else (
                PANEL_ALT if selected else PANEL)
            self._panel(box, fill)
            token = pygame.transform.smoothscale(self.tokens[player.name], (28, 28))
            self.screen.blit(token, (box.x + 10, box.y + 14))
            if player.bankrupt:
                self._text(f"{player.name} — bankrupt", (box.x + 48, box.y + 16),
                           self.f_body, MUTED)
            else:
                jail = "  (in jail)" if player.in_jail else ""
                self._text(f"{player.name}{jail}", (box.x + 48, box.y + 8),
                           self.f_body, INK)
                meta = f"${player.balance}   ·   {len(player.properties)} properties"
                self._text(meta, (box.x + 48, box.y + 31), self.f_small, MUTED)
            rects.append((box, index))
            y += 64
        return y, rects

    def _rent_line(self, prop):
        """Returns (status_label, rent_label) for a property in an inventory."""
        if prop.mortgaged:
            return "Mortgaged", "No rent"
        if isinstance(prop, StreetProperty):
            if prop.houses >= 5:
                houses = "Hotel"
            elif prop.houses > 0:
                houses = f"{prop.houses} house" + ("s" if prop.houses > 1 else "")
            else:
                houses = "No houses"
            rent = prop.get_rent(self.game, prop.owner)
            return houses, f"Rent ${rent}"
        if isinstance(prop, Railroad):
            rent = prop.get_rent(self.game, prop.owner)
            return "Railroad", f"Rent ${rent}"
        if isinstance(prop, Utility):
            count = sum(isinstance(p, Utility) for p in prop.owner.properties)
            return "Utility", f"Rent {4 if count == 1 else 10}× dice"
        return "", ""

    def _draw_inventory(self, y, player):
        self._text(f"{player.name}'s inventory", (SIDE_X, y), self.f_head,
                   FELT_INK, shadow=True)
        self._text("(click the player again to close)", (SIDE_X, y + 28),
                   self.f_small, FELT_SUB, shadow=True)
        y += 56
        if not player.properties:
            self._text("No properties owned.", (SIDE_X, y), self.f_body,
                       FELT_SUB, shadow=True)
            return
        for prop in player.properties:
            houses, rent = self._rent_line(prop)
            self._text(prop.name, (SIDE_X, y), self.f_small, FELT_INK,
                       shadow=True)
            info = f"{houses}  ·  {rent}"
            w = self.f_small.size(info)[0]
            self._text(info, (SIDE_X + SIDE_W - w, y), self.f_small, FELT_SUB,
                       shadow=True)
            y += 30
            if y > HEIGHT - 30:
                break

    def _draw_log(self, y):
        self._text("Log", (SIDE_X, y), self.f_title, FELT_INK, shadow=True)
        y += 40
        for line in self.log[-((HEIGHT - y) // 24):]:
            self._text(line, (SIDE_X, y), self.f_small, FELT_INK, shadow=True)
            y += 24

    # ----- prompt --------------------------------------------------------

    def _draw_prompt(self, question, options, mouse):
        # Size the box from the actual wrapped question height so the buttons
        # always fit inside it (a multi-line question used to push them out the
        # bottom), and clamp it to the screen so it never runs off the top.
        lines = self._wrap(question, self.f_body, SIDE_W - 28)
        height = 14 + len(lines) * 26 + 10 + len(options) * 52 + 6
        top = max(8, HEIGHT - height - 8)
        box = pygame.Rect(SIDE_X, top, SIDE_W, height)
        self._panel(box)
        y = box.y + 14
        for line in lines:
            self._text(line, (box.x + 14, y), self.f_body)
            y += 26
        y += 10
        buttons = []
        for i, (label, value) in enumerate(options):
            rect = pygame.Rect(box.x + 14, y, SIDE_W - 28, 44)
            hover = rect.collidepoint(mouse) if mouse else False
            pygame.draw.rect(self.screen, BTN_HOVER if hover else BTN, rect,
                             border_radius=8)
            label = self._truncate(f"{i + 1}.  {label}", self.f_body,
                                   rect.width - 32)
            surf = self.f_body.render(label, True, BTN_INK)
            self.screen.blit(surf, surf.get_rect(
                midleft=(rect.x + 16, rect.centery)))
            buttons.append((rect, value))
            y += 52
        return buttons

    def _truncate(self, text, font, max_width):
        """Shortens text with an ellipsis so it fits within a button's width."""
        if font.size(text)[0] <= max_width:
            return text
        while text and font.size(text + "…")[0] > max_width:
            text = text[:-1]
        return text + "…"

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

    # ----- scene ---------------------------------------------------------

    def _draw_scene(self, question=None, options=None, mouse=None):
        self.screen.fill(BG)
        self._draw_board()
        self._text("Monopoly", (SIDE_X, 16), self.f_title, PANEL, shadow=True)
        y = self._draw_dice_panel(54)
        y, player_rects = self._draw_players(y + 14)
        y += 8
        # Inventory/log always render so a player can be clicked open even while
        # a decision is pending; the prompt simply overlays the bottom.
        if self.selected is not None:
            self._draw_inventory(y, self.game.players[self.selected])
        else:
            self._draw_log(y)
        buttons = []
        if question:
            buttons = self._draw_prompt(question, options, mouse)
        return buttons, player_rects

    def render(self):
        """Draws a single frame (no prompt). Used for smoke tests."""
        self._draw_scene()
        pygame.display.flip()

    # ----- input ---------------------------------------------------------

    def ask(self, question, options):
        """Shows a modal question and blocks until an option is chosen."""
        if self._auto is not None:
            return self._auto(question, options)
        while True:
            mouse = pygame.mouse.get_pos()
            buttons, player_rects = self._draw_scene(question, options, mouse)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise QuitGame
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, value in buttons:
                        if rect.collidepoint(event.pos):
                            return value
                    for rect, index in player_rects:
                        if rect.collidepoint(event.pos):
                            self.selected = None if self.selected == index else index
                if event.type == pygame.KEYDOWN:
                    idx = event.key - pygame.K_1
                    if 0 <= idx < len(options):
                        return options[idx][1]
            self.clock.tick(60)

    # ----- animation -----------------------------------------------------

    def _frame(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise QuitGame
        self._draw_scene()
        pygame.display.flip()
        self.clock.tick(60)

    def _snap(self, player):
        self.vpos[player.name] = float(player.pos)

    def _animate_slide(self, player, from_pos, steps):
        name = player.name
        if self._auto is not None or steps <= 0:
            self._snap(player)
            return
        duration = min(0.16 * steps, 1.2)
        hop_h = TOKEN_PX * 0.40
        start = time.time()
        while True:
            t = (time.time() - start) / duration
            if t >= 1:
                break
            ease = t * t * (3 - 2 * t)
            v = from_pos + steps * ease
            self.vpos[name] = v
            # A little hop as the token crosses each tile.
            self.hop[name] = math.sin((v % 1.0) * math.pi) * hop_h
            self._frame()
        self.hop[name] = 0.0
        self._snap(player)
        self._animate_land(player)

    def _animate_land(self, player):
        """A small settling bounce once the token reaches its tile."""
        if self._auto is not None:
            return
        name = player.name
        start = time.time()
        dur = 0.24
        while True:
            t = (time.time() - start) / dur
            if t >= 1:
                break
            self.hop[name] = math.sin(t * math.pi) * (TOKEN_PX * 0.20) * (1 - t)
            self._frame()
        self.hop[name] = 0.0

    def _animate_dice(self, player, final):
        """Tumbles the dice faces for a moment, then settles on the result."""
        if self._auto is not None:
            self._set_roll(player, final)
            return
        for _ in range(11):
            self._set_roll(
                player, (random.randint(1, 6), random.randint(1, 6)))
            self._frame()
            pygame.time.wait(38)
        self._set_roll(player, final)
        self._frame()

    # ----- decisions -----------------------------------------------------

    def _prompt_purchase(self, player, prop):
        if player.balance < prop.price:
            self.add_log(f"{player.name} can't afford {prop.name}.")
            return False
        choice = self.ask(
            f"{player.name}: buy {prop.name} for ${prop.price}? "
            f"(Balance ${player.balance})",
            [("Buy", "buy"), ("Decline", "decline")])
        bought = choice == "buy"
        self.add_log(f"{player.name} {'bought' if bought else 'declined'} "
                     f"{prop.name}.")
        return bought

    def _show_card(self, pile, card):
        """Shows a drawn Chance / Community Chest card before its effect runs."""
        self.add_log(f"{pile}: {card.text}")
        self.ask(f"{pile} card — {card.text}", [("Continue", "ok")])

    def _inform(self, message):
        """Logs an automatic game event and makes the player acknowledge it.

        Bound to ``Game.notify``. In auto / RL mode ``ask`` returns immediately,
        so the event is still logged but no prompt blocks play.
        """
        self.add_log(message)
        self.ask(message, [("Continue", "ok")])

    def _buildable(self, player):
        return [t for t in self.game.board.tiles
                if isinstance(t, StreetProperty)
                and t.can_build_house(self.game, player)]

    def _sellable(self, player):
        return [t for t in self.game.board.tiles
                if isinstance(t, StreetProperty)
                and t.can_sell_house(self.game, player)]

    def _mortgageable(self, player):
        return [t for t in player.properties
                if t.can_mortgage(self.game, player)]

    def _unmortgageable(self, player):
        return [t for t in player.properties
                if t.can_unmortgage(self.game, player)]

    def _can_manage(self, player):
        """Whether the player has any property action available right now."""
        return bool(self._buildable(player) or self._sellable(player)
                    or self._mortgageable(player)
                    or self._unmortgageable(player))

    def _build_flow(self, player):
        options = [(f"{t.name} (${t.house_cost()})", t.pos)
                   for t in self._buildable(player)]
        options.append(("Cancel", None))
        choice = self.ask(f"{player.name}: build a house on which street?",
                          options)
        if choice is not None:
            tile = self.game.board.get_tile(choice)
            self.game.build_house(tile, player)
            self.add_log(f"{player.name} built on {tile.name} "
                         f"(now {tile.houses}).")

    def _sell_flow(self, player):
        options = [(f"{t.name} ({t.houses} houses)", t.pos)
                   for t in self._sellable(player)]
        options.append(("Cancel", None))
        choice = self.ask(f"{player.name}: sell a house from which street?",
                          options)
        if choice is not None:
            tile = self.game.board.get_tile(choice)
            self.game.sell_house(tile, player)
            self.add_log(f"{player.name} sold a house on {tile.name}.")

    def _mortgage_flow(self, player):
        options = [(f"{t.name}  (+${t.mortgage_value})", t.pos)
                   for t in self._mortgageable(player)]
        options.append(("Cancel", None))
        choice = self.ask(f"{player.name}: mortgage which property?", options)
        if choice is not None:
            tile = self.game.board.get_tile(choice)
            if self.game.mortgage_property(tile, player):
                self.add_log(f"{player.name} mortgaged {tile.name} "
                             f"for ${tile.mortgage_value}.")

    def _unmortgage_flow(self, player):
        options = [(f"{t.name}  (-${t.unmortgage_cost})", t.pos)
                   for t in self._unmortgageable(player)]
        options.append(("Cancel", None))
        choice = self.ask(f"{player.name}: unmortgage which property?", options)
        if choice is not None:
            tile = self.game.board.get_tile(choice)
            cost = tile.unmortgage_cost
            if self.game.unmortgage_property(tile, player):
                self.add_log(f"{player.name} lifted the mortgage on "
                             f"{tile.name} for ${cost}.")

    def _manage_menu(self, player):
        """Property-management submenu: build/sell houses, mortgage/unmortgage.

        Reachable both before and after rolling, and loops until the player is
        done so they can make several changes in one visit.
        """
        while True:
            options = []
            if self._buildable(player):
                options.append(("Build a house", "build"))
            if self._sellable(player):
                options.append(("Sell a house", "sell"))
            if self._mortgageable(player):
                options.append(("Mortgage a property", "mortgage"))
            if self._unmortgageable(player):
                options.append(("Unmortgage a property", "unmortgage"))
            options.append(("Done managing", "done"))
            choice = self.ask(
                f"{player.name}: manage properties (${player.balance})", options)
            if choice == "build":
                self._build_flow(player)
            elif choice == "sell":
                self._sell_flow(player)
            elif choice == "mortgage":
                self._mortgage_flow(player)
            elif choice == "unmortgage":
                self._unmortgage_flow(player)
            else:
                return

    def _set_roll(self, player, dice):
        self.roll_display = {"name": player.name, "dice": dice}

    # ----- turn flow -----------------------------------------------------

    def _turn_menu(self, player):
        """Pre-roll menu: manage properties, then choose the jail/roll action."""
        while True:
            options = []
            if player.in_jail:
                if player.balance >= 50:
                    options.append(("Pay $50 to leave jail", "pay"))
                if player.jail_cards:
                    options.append(("Use Get Out of Jail Free card", "card"))
                options.append(("Roll for doubles", "roll"))
            else:
                options.append(("Roll dice", "roll"))
            if self._can_manage(player):
                options.append(("Manage properties", "manage"))

            where = "in jail" if player.in_jail else \
                self.game.board.get_tile(player.pos).name
            choice = self.ask(
                f"{player.name}'s turn — {where} (${player.balance})", options)
            if choice == "manage":
                self._manage_menu(player)
            else:
                return choice

    def _post_roll_manage(self, player):
        """After the roll resolves, offer one more management pass before the
        turn ends."""
        if not self._can_manage(player):
            return
        choice = self.ask(
            f"{player.name}: manage properties before ending your turn? "
            f"(${player.balance})",
            [("Manage properties", "manage"), ("End turn", "end")])
        if choice == "manage":
            self._manage_menu(player)

    def _resolve_and_sync(self, player):
        self.game.resolve_tile(player)
        self._snap(player)  # cards/jail teleport: land instantly, no slide

    def _play_rolls(self, player):
        """Rolls one die at a time, stopping to ask before re-rolling doubles."""
        while True:
            from_pos = player.pos
            die1, die2, is_double, to_jail = self.game.roll_once(player)
            self._animate_dice(player, (die1, die2))

            if to_jail:
                self.add_log(
                    f"{player.name} rolled a third double and was sent to jail!")
                self._snap(player)
                break

            self.add_log(f"{player.name} rolled {die1} + {die2} = {die1 + die2}.")
            self._animate_slide(player, from_pos, die1 + die2)
            self._resolve_and_sync(player)

            if player.in_jail:
                self.add_log(f"{player.name} was sent to jail.")
                break
            if player.bankrupt:
                break
            if is_double:
                self.ask(f"{player.name} rolled doubles — roll again!",
                         [("Roll again", "roll")])
                continue
            break

    def _handle_jail(self, player, choice):
        result = self.game.handle_jail_turn(player, choice)
        dice = self.game.last_dice

        if result == "released":
            if choice == "pay":
                msg = f"{player.name} paid $50 and left jail."
            else:
                msg = f"{player.name} used a Get Out of Jail Free card."
            self.add_log(msg)
            # Let the player roll for their normal turn rather than auto-rolling.
            self.ask(f"{msg} Roll the dice to take your turn.",
                     [("Roll dice", "roll")])
            self._play_rolls(player)  # takes a normal turn
            return

        if result == "moved":
            self._animate_dice(player, dice)
            self.add_log(f"{player.name} rolled doubles "
                         f"({dice[0]} + {dice[1]}) and left jail.")
            self._animate_slide(player, self.game.jail_position, sum(dice))
            self._resolve_and_sync(player)
        elif result == "freed":
            self._animate_dice(player, dice)
            self.add_log(f"{player.name} failed to roll doubles and was released.")
        else:  # jailed
            self._animate_dice(player, dice)
            self.add_log(f"{player.name} rolled {dice[0]} + {dice[1]} — "
                         f"no doubles, still in jail.")

    def _end_turn(self, player):
        player.double_count = 0
        self.game.advance_turn()

    def _handoff(self):
        nxt = self.game.players[self.game.current_player]
        self.selected = None
        self.ask(f"Pass the computer to {nxt.name}.", [("Continue", "ok")])

    def _show_result(self):
        winner = self.game.winner()
        message = f"{winner.name} wins!" if winner else "Game over."
        self.add_log(message)
        self.ask(message, [("Quit", "quit")])

    # ----- main loop -----------------------------------------------------

    def run(self, max_turns=10000):
        """Plays the game to completion, then shows the result screen."""
        try:
            turns = 0
            while not self.game.is_over() and turns < max_turns:
                player = self.game.players[self.game.current_player]
                choice = self._turn_menu(player)
                if player.in_jail:
                    self._handle_jail(player, choice)
                else:
                    self._play_rolls(player)
                # Let the player manage properties after rolling, then end turn.
                if not player.bankrupt:
                    self._post_roll_manage(player)
                self._end_turn(player)
                turns += 1
                if not self.game.is_over():
                    self._handoff()
            self._show_result()
        except QuitGame:
            pass
        finally:
            pygame.quit()


def main():
    """Builds a fresh game and launches the UI."""
    from engine.game import Game
    from models.board import Board
    from models.player import Player
    from data.decks import build_chance_deck, build_community_deck
    from data.board_tiles import build_board_tiles

    players = [Player("Red"), Player("Blue"), Player("Green"), Player("Yellow")]
    game = Game(players, Board(build_board_tiles()),
                build_chance_deck(), build_community_deck())
    MonopolyApp(game).run()


if __name__ == "__main__":
    main()
