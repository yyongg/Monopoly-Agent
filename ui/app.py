"""Pygame UI for the Monopoly game — supports human and AI players.

All decision hooks route through ``ask()`` for human players. AI players bypass
the prompt entirely: their trained model is queried synchronously and the result
applied directly. Animations (dice roll, token slide) play for all players so
human observers can follow AI turns.

The ``main()`` entry point shows a setup screen where each of the four players
can be toggled between Human and AI before the game starts. Pass ``--model
<path.zip>`` to load a specific model (defaults to ``runs/monopoly_ppo.zip``).
"""

import argparse
import math
import os
import random
import time

import pygame

from ui.board_layout import (tile_center, tile_rect, interior_offset,
                              TILE_FRAC)
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

FELT_INK = (245, 243, 233)
FELT_SUB = (197, 214, 197)
ACCENT = (212, 175, 55)
BTN = (58, 110, 165)
BTN_HOVER = (78, 138, 197)
BTN_INK = (255, 255, 255)
BTN_DIM = (90, 90, 90)

CHOICE_ICONS = {
    "build": "up",
    "sell": "down",
    "mortgage": "coin",
    "unmortgage": "ring",
    "trade": "swap",
    "manage": "menu",
    "roll": "die",
    "end": "check",
    "done": "check",
    "pay": "coin",
    "card": "star",
    "give_up": "warn",
}
ICON_COLORS = {
    "up": (120, 230, 140),
    "down": (255, 150, 150),
    "coin": (255, 210, 90),
    "ring": (255, 210, 90),
    "warn": (255, 184, 92),
}
HOUSE_GREEN = (38, 158, 70)
HOTEL_RED = (200, 62, 55)
DIE_FACE = (250, 250, 248)
PIP = (32, 32, 32)

PLAYER_COLORS = {
    "Red": (211, 47, 47),
    "Blue": (33, 99, 199),
    "Green": (39, 158, 70),
    "Yellow": (240, 196, 32),
}
DEFAULT_COLOR = (120, 120, 120)


def player_color(name):
    return PLAYER_COLORS.get(name, DEFAULT_COLOR)


GROUP_COLORS = {
    "brown": (124, 78, 51),
    "light_blue": (170, 224, 250),
    "pink": (213, 56, 145),
    "orange": (244, 149, 31),
    "red": (227, 38, 41),
    "yellow": (254, 240, 64),
    "green": (32, 165, 90),
    "dark_blue": (0, 112, 186),
}
RAILROAD_COLOR = (40, 40, 40)
UTILITY_COLOR = (200, 200, 200)


def contrast_text(color):
    r, g, b = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return INK if luminance > 140 else (255, 255, 255)


ASSETS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


class QuitGame(Exception):
    pass


# ---------------------------------------------------------------------------
# Setup screen
# ---------------------------------------------------------------------------

def _run_setup(screen, clock, model_available):
    """Shows the player-setup dialog before the game starts.

    Returns ``{"types": {name: "human"|"ai"}, "randomize": bool}`` or ``None``
    if the user closed the window.
    """
    font_title = pygame.font.SysFont("dejavusans,arial,helvetica", 44, bold=True)
    font_head  = pygame.font.SysFont("dejavusans,arial,helvetica", 28, bold=True)
    font_body  = pygame.font.SysFont("dejavusans,arial,helvetica", 24)
    font_small = pygame.font.SysFont("dejavusans,arial,helvetica", 20)

    names = ["Red", "Blue", "Green", "Yellow"]
    types = {n: "human" for n in names}
    randomize = True

    # Panel geometry — centered on the screen.
    pw, ph = 560, 530
    px = WIDTH // 2 - pw // 2
    py = HEIGHT // 2 - ph // 2 - 20

    while True:
        mouse = pygame.mouse.get_pos()
        screen.fill(BG)

        # Panel card.
        panel_rect = pygame.Rect(px, py, pw, ph)
        pygame.draw.rect(screen, PANEL, panel_rect, border_radius=14)
        pygame.draw.rect(screen, EDGE, panel_rect, 1, border_radius=14)

        # Title.
        surf = font_title.render("MONOPOLY", True, ACCENT)
        screen.blit(surf, surf.get_rect(midtop=(px + pw // 2, py + 18)))
        surf = font_head.render("Player Setup", True, INK)
        screen.blit(surf, surf.get_rect(midtop=(px + pw // 2, py + 70)))

        hot = []  # list of (rect, callback)

        # Player rows.
        row_top = py + 116
        row_h = 60
        for i, name in enumerate(names):
            ry = row_top + i * row_h
            # Separator line above each row except the first.
            if i > 0:
                pygame.draw.line(screen, EDGE,
                                 (px + 16, ry), (px + pw - 16, ry))

            # Color dot.
            pygame.draw.circle(screen, player_color(name),
                               (px + 32, ry + row_h // 2), 13)

            # Name label.
            surf = font_body.render(name, True, INK)
            screen.blit(surf, surf.get_rect(midleft=(px + 58, ry + row_h // 2)))

            # Human / AI toggle buttons (right side of row).
            for j, (label, ptype) in enumerate([("Human", "human"), ("AI", "ai")]):
                bw, bh = 86, 34
                bx = px + pw - 24 - (2 - j) * (bw + 8)
                by = ry + (row_h - bh) // 2
                btn = pygame.Rect(bx, by, bw, bh)
                is_selected = types[name] == ptype
                is_disabled = ptype == "ai" and not model_available
                if is_disabled:
                    fill = (160, 160, 160)
                    ink = (110, 110, 110)
                elif is_selected:
                    fill = BTN_HOVER if btn.collidepoint(mouse) else BTN
                    ink = BTN_INK
                else:
                    fill = PANEL_ALT if not btn.collidepoint(mouse) else (210, 210, 205)
                    ink = MUTED
                pygame.draw.rect(screen, fill, btn, border_radius=7)
                pygame.draw.rect(screen, EDGE, btn, 1, border_radius=7)
                surf = font_small.render(label, True, ink)
                screen.blit(surf, surf.get_rect(center=btn.center))
                if not is_disabled:
                    _name, _ptype = name, ptype
                    hot.append((btn, (_name, _ptype)))

        # Separator before extras.
        sep_y = row_top + len(names) * row_h + 8
        pygame.draw.line(screen, EDGE, (px + 16, sep_y), (px + pw - 16, sep_y))

        # Randomize order toggle.
        rand_y = sep_y + 16
        box = pygame.Rect(px + 22, rand_y + 5, 22, 22)
        pygame.draw.rect(screen, PANEL_ALT, box, border_radius=4)
        pygame.draw.rect(screen, EDGE, box, 1, border_radius=4)
        if randomize:
            pygame.draw.lines(screen, BTN, False,
                              [(box.x + 3, box.y + 11),
                               (box.x + 9, box.y + 17),
                               (box.x + 19, box.y + 4)], 3)
        surf = font_body.render("Randomize turn order", True, INK)
        screen.blit(surf, surf.get_rect(midleft=(px + 54, rand_y + 16)))
        rand_hitbox = pygame.Rect(px + 16, rand_y, pw - 32, 32)
        hot.append((rand_hitbox, ("randomize", None)))

        # Warning when no model is available.
        warn_y = rand_y + 42
        if not model_available:
            surf = font_small.render(
                "No model found — AI players unavailable.", True, (180, 90, 40))
            screen.blit(surf, surf.get_rect(midtop=(px + pw // 2, warn_y)))
            surf = font_small.render(
                "Train a model first: python train.py", True, MUTED)
            screen.blit(surf, surf.get_rect(midtop=(px + pw // 2, warn_y + 24)))

        # Start Game button.
        start_bw, start_bh = 200, 48
        start_btn = pygame.Rect(px + pw // 2 - start_bw // 2,
                                py + ph - start_bh - 20,
                                start_bw, start_bh)
        hover_start = start_btn.collidepoint(mouse)
        pygame.draw.rect(screen, BTN_HOVER if hover_start else BTN,
                         start_btn, border_radius=10)
        surf = font_body.render("Start Game", True, BTN_INK)
        screen.blit(surf, surf.get_rect(center=start_btn.center))
        hot.append((start_btn, ("start", None)))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return {"types": dict(types), "randomize": randomize}
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, payload in hot:
                    if rect.collidepoint(event.pos):
                        key, val = payload
                        if key == "start":
                            return {"types": dict(types), "randomize": randomize}
                        elif key == "randomize":
                            randomize = not randomize
                        else:
                            types[key] = val
        clock.tick(60)


def _show_error(screen, clock, lines):
    """Shows a blocking error panel until the user clicks or presses a key."""
    font_head = pygame.font.SysFont("dejavusans,arial,helvetica", 28, bold=True)
    font_body = pygame.font.SysFont("dejavusans,arial,helvetica", 20)
    while True:
        screen.fill(BG)
        pw, ph = 640, 60 + len(lines) * 30 + 70
        px = WIDTH // 2 - pw // 2
        py = HEIGHT // 2 - ph // 2
        panel = pygame.Rect(px, py, pw, ph)
        pygame.draw.rect(screen, PANEL, panel, border_radius=14)
        pygame.draw.rect(screen, (180, 90, 40), panel, 2, border_radius=14)
        surf = font_head.render("Heads up", True, (180, 90, 40))
        screen.blit(surf, surf.get_rect(midtop=(px + pw // 2, py + 18)))
        y = py + 62
        for line in lines:
            screen.blit(font_body.render(line, True, INK), (px + 30, y))
            y += 30
        btn = pygame.Rect(px + pw // 2 - 90, py + ph - 56, 180, 40)
        pygame.draw.rect(screen, BTN, btn, border_radius=8)
        surf = font_body.render("Continue", True, BTN_INK)
        screen.blit(surf, surf.get_rect(center=btn.center))
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN):
                return
        clock.tick(60)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class MonopolyApp:
    """Renders the board and drives a hot-seat / AI game loop."""

    def __init__(self, game, auto=None, ai_deciders=None, screen=None):
        """
        Args:
            game: A constructed game (players, board, decks).
            auto: Optional headless responder ``(question, options) -> value``.
            ai_deciders: dict of ``player_name -> GUIAIDecider`` for AI seats.
            screen: An existing pygame surface to reuse (skips display init).
        """
        self.game = game
        self._auto = auto
        self._ai_deciders = ai_deciders or {}
        self.log = []
        self.selected = None
        self.roll_display = None
        self.board_dice = None
        self.vpos = {p.name: float(p.pos) for p in game.players}
        self.hop = {p.name: 0.0 for p in game.players}

        if screen is None:
            pygame.init()
            pygame.key.set_repeat(300, 40)
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        else:
            self.screen = screen

        pygame.display.set_caption("Monopoly")
        self.clock = pygame.time.Clock()
        self.f_title = self._font(30, bold=True)
        self.f_head  = self._font(24, bold=True)
        self.f_body  = self._font(22)
        self.f_small = self._font(19)

        self.board_img = pygame.transform.smoothscale(
            pygame.image.load(
                os.path.join(ASSETS, "board_minimal.png")).convert(),
            (BOARD_PX, BOARD_PX),
        )
        self.tokens = {}
        for player in game.players:
            path = os.path.join(ASSETS, f"{player.name.lower()}_player.png")
            img = pygame.image.load(path).convert_alpha()
            self.tokens[player.name] = pygame.transform.smoothscale(
                img, (TOKEN_PX, TOKEN_PX))

        # Bind AI deciders to the live game board.
        ownable = [t for t in game.board.tiles
                   if isinstance(t, (StreetProperty, Railroad, Utility))]
        for ai in self._ai_deciders.values():
            ai.bind(game, ownable)
            ai.log = self.add_log  # so AI moves show up in the game log

        # Wire per-player purchase hooks.
        for player in game.players:
            if player.name in self._ai_deciders:
                ai = self._ai_deciders[player.name]
                player.decide_purchase = (
                    lambda prop, _ai=ai, _p=player: _ai.purchase_decision(_p, prop))
            else:
                player.decide_purchase = (
                    lambda prop, _p=player: self._prompt_purchase(_p, prop))

        game.on_card = self._show_card
        game.notify = self._inform
        game.on_shortfall = self._on_shortfall

    # -- Helpers ----------------------------------------------------------

    def _is_ai(self, player):
        return player.name in self._ai_deciders

    def _current_is_ai(self):
        return self._is_ai(self.game.players[self.game.current_player])

    def _ai_pause(self, seconds):
        """A short, skippable pause so AI events are watchable (no blocking)."""
        if self._auto is not None:
            return
        start = time.time()
        while time.time() - start < seconds:
            if self._frame():  # a click / key press skips the wait
                return

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
            px, py = -oy, ox

            if tile.houses >= 5:
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
        tsize = TILE_FRAC * BOARD_PX
        for tile in self.game.board.tiles:
            if not isinstance(tile, Property) or tile.owner is None:
                continue
            cx, cy = tile_center(tile.pos, BOARD_X, BOARD_Y, BOARD_PX)
            ox, oy = interior_offset(tile.pos)
            mx, my = int(cx - ox * tsize * 0.34), int(cy - oy * tsize * 0.34)
            color = player_color(tile.owner.name)
            pygame.draw.circle(self.screen, (255, 255, 255), (mx, my), 7)
            if tile.mortgaged:
                pygame.draw.circle(self.screen, color, (mx, my), 6, 2)
            else:
                pygame.draw.circle(self.screen, color, (mx, my), 6)
            pygame.draw.circle(self.screen, (0, 0, 0), (mx, my), 7, 1)

    def _draw_current_highlight(self):
        player = self.game.players[self.game.current_player]
        if player.bankrupt:
            return
        x, y, w, h = tile_rect(player.pos, BOARD_X, BOARD_Y, BOARD_PX)
        pulse = 0.5 + 0.5 * math.sin(time.time() * 5.0)
        thick = 2 + int(round(pulse * 2))
        rect = pygame.Rect(x + thick // 2, y + thick // 2,
                           w - thick, h - thick)
        pygame.draw.rect(self.screen, ACCENT, rect, thick)

    def _cluster_offset(self, slot, count):
        if count <= 1:
            return (0.0, 0.0)
        step = BOARD_PX * TILE_FRAC * 0.30
        cols = 2
        rows = (count + 1) // 2
        row, col = slot // cols, slot % cols
        in_row = cols if (row < rows - 1 or count % cols == 0) else count % cols
        dx = (col - (in_row - 1) / 2) * step
        dy = (row - (rows - 1) / 2) * step
        return (dx, dy)

    def _draw_board(self):
        shadow = pygame.Rect(BOARD_X + 6, BOARD_Y + 8, BOARD_PX, BOARD_PX)
        pygame.draw.rect(self.screen, (6, 48, 30), shadow, border_radius=6)
        self.screen.blit(self.board_img, (BOARD_X, BOARD_Y))
        self._draw_ownership()
        self._draw_house_indicators()
        self._draw_current_highlight()
        current = self.game.players[self.game.current_player]
        active = [p for p in self.game.players if not p.bankrupt]
        for player in active:
            cx, cy = self._token_center(self.vpos[player.name])
            sharers = [p for p in active if p.pos == player.pos]
            dx, dy = self._cluster_offset(sharers.index(player), len(sharers))
            tx = int(cx + dx)
            ty = int(cy + dy - self.hop.get(player.name, 0.0))
            if player is current:
                pygame.draw.circle(self.screen, ACCENT, (tx, ty),
                                   TOKEN_PX // 2 + 4, 3)
            token = self.tokens[player.name]
            self.screen.blit(token, token.get_rect(center=(tx, ty)))
        self._draw_board_dice()

    def _draw_board_dice(self):
        if not self.board_dice:
            return
        d1, d2 = self.board_dice
        size, gap = 88, 26
        cx = BOARD_X + BOARD_PX // 2
        cy = BOARD_Y + BOARD_PX // 2
        total_w = size * 2 + gap
        pad = 24
        back = pygame.Rect(0, 0, total_w + pad * 2, size + pad * 2)
        back.center = (cx, cy)
        backdrop = pygame.Surface(back.size, pygame.SRCALPHA)
        pygame.draw.rect(backdrop, (10, 38, 24, 180), backdrop.get_rect(),
                         border_radius=20)
        self.screen.blit(backdrop, back.topleft)
        left = cx - total_w // 2
        top = cy - size // 2
        self._draw_die(left, top, size, d1)
        self._draw_die(left + size + gap, top, size, d2)

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

    def _net_worth(self, player):
        total = player.balance
        for prop in player.properties:
            total += prop.mortgage_value if prop.mortgaged else prop.price
            if isinstance(prop, StreetProperty):
                total += prop.houses * prop.house_cost()
        return total

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
            stripe = pygame.Rect(box.x, box.y, 7, box.height)
            pygame.draw.rect(self.screen, player_color(player.name), stripe,
                             border_top_left_radius=10, border_bottom_left_radius=10)
            token = pygame.transform.smoothscale(self.tokens[player.name], (28, 28))
            self.screen.blit(token, (box.x + 18, box.y + 14))
            if player.bankrupt:
                self._text(f"{player.name} — bankrupt", (box.x + 56, box.y + 16),
                           self.f_body, MUTED)
            else:
                jail = "  (in jail)" if player.in_jail else ""
                ai_badge = " [AI]" if self._is_ai(player) else ""
                self._text(f"{player.name}{ai_badge}{jail}",
                           (box.x + 56, box.y + 8), self.f_body, INK)
                meta = (f"${player.balance}   ·   {len(player.properties)} props"
                        f"   ·   net ${self._net_worth(player)}")
                self._text(meta, (box.x + 56, box.y + 31), self.f_small, MUTED)
            rects.append((box, index))
            y += 64
        return y, rects

    def _rent_line(self, prop):
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

    def _draw_inventory(self, y, player, bottom):
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
            if y > bottom - 30:
                break
            houses, rent = self._rent_line(prop)
            self._draw_property_chip(prop, SIDE_X, y)
            info = f"{houses}  ·  {rent}"
            w = self.f_small.size(info)[0]
            self._text(info, (SIDE_X + SIDE_W - w, y), self.f_small, FELT_SUB,
                       shadow=True)
            y += 30

    def _property_color(self, prop):
        if isinstance(prop, StreetProperty):
            return GROUP_COLORS.get(prop.color, DEFAULT_COLOR)
        if isinstance(prop, Railroad):
            return RAILROAD_COLOR
        if isinstance(prop, Utility):
            return UTILITY_COLOR
        return DEFAULT_COLOR

    def _choice_icon(self, value):
        if isinstance(value, int):
            tile = self.game.board.tiles[value]
            if isinstance(tile, Property):
                return "swatch", self._property_color(tile)
        if value is None:
            return "x", None
        shape = CHOICE_ICONS.get(value, "dot")
        return shape, ICON_COLORS.get(shape)

    def _draw_icon(self, shape, color, cx, cy, s=20):
        scr = self.screen
        h = s // 2
        if shape == "swatch":
            r = pygame.Rect(0, 0, s, s)
            r.center = (cx, cy)
            pygame.draw.rect(scr, color, r, border_radius=4)
            pygame.draw.rect(scr, (0, 0, 0), r, 1, border_radius=4)
        elif shape == "up":
            pygame.draw.polygon(scr, color,
                                [(cx, cy - h), (cx + h, cy + h), (cx - h, cy + h)])
        elif shape == "down":
            pygame.draw.polygon(scr, color,
                                [(cx, cy + h), (cx + h, cy - h), (cx - h, cy - h)])
        elif shape == "coin":
            pygame.draw.circle(scr, color, (cx, cy), h)
            pygame.draw.circle(scr, (0, 0, 0), (cx, cy), h, 1)
            pygame.draw.line(scr, (0, 0, 0), (cx, cy - 5), (cx, cy + 5), 2)
        elif shape == "ring":
            pygame.draw.circle(scr, color, (cx, cy), h, 2)
            pygame.draw.line(scr, color, (cx, cy - 4), (cx, cy + 4), 2)
        elif shape == "swap":
            pygame.draw.line(scr, color, (cx - h, cy - 4), (cx + h - 3, cy - 4), 2)
            pygame.draw.polygon(scr, color, [(cx + h, cy - 4),
                                (cx + h - 6, cy - 8), (cx + h - 6, cy)])
            pygame.draw.line(scr, color, (cx - h + 3, cy + 4), (cx + h, cy + 4), 2)
            pygame.draw.polygon(scr, color, [(cx - h, cy + 4),
                                (cx - h + 6, cy), (cx - h + 6, cy + 8)])
        elif shape == "menu":
            for dy in (-6, 0, 6):
                pygame.draw.line(scr, color, (cx - h, cy + dy), (cx + h, cy + dy), 2)
        elif shape == "die":
            r = pygame.Rect(0, 0, s, s)
            r.center = (cx, cy)
            pygame.draw.rect(scr, color, r, 2, border_radius=4)
            for px, py in [(cx - 5, cy - 5), (cx + 5, cy - 5), (cx, cy),
                           (cx - 5, cy + 5), (cx + 5, cy + 5)]:
                pygame.draw.circle(scr, color, (px, py), 2)
        elif shape == "check":
            pygame.draw.lines(scr, color, False,
                              [(cx - h + 1, cy + 1), (cx - 2, cy + h - 2),
                               (cx + h, cy - h + 1)], 3)
        elif shape == "star":
            pts = []
            for k in range(10):
                ang = -math.pi / 2 + k * math.pi / 5
                rad = h if k % 2 == 0 else h * 0.45
                pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
            pygame.draw.polygon(scr, color, pts)
        elif shape == "warn":
            pygame.draw.polygon(scr, color,
                                [(cx, cy - h), (cx + h, cy + h), (cx - h, cy + h)], 2)
            pygame.draw.line(scr, color, (cx, cy - 3), (cx, cy + 4), 2)
            pygame.draw.circle(scr, color, (cx, cy + 8), 1)
        elif shape == "x":
            pygame.draw.line(scr, color, (cx - h, cy - h), (cx + h, cy + h), 3)
            pygame.draw.line(scr, color, (cx + h, cy - h), (cx - h, cy + h), 3)
        else:
            pygame.draw.circle(scr, color, (cx, cy), 4)

    def _draw_property_chip(self, prop, x, y):
        color = self._property_color(prop)
        tw = self.f_small.size(prop.name)[0]
        chip = pygame.Rect(x - 4, y - 2, tw + 12, self.f_small.get_height() + 4)
        pygame.draw.rect(self.screen, color, chip, border_radius=4)
        pygame.draw.rect(self.screen, (0, 0, 0), chip, 1, border_radius=4)
        self.screen.blit(
            self.f_small.render(prop.name, True, contrast_text(color)),
            (x + 2, y))

    def _draw_log(self, y, bottom):
        self._text("Log", (SIDE_X, y), self.f_title, FELT_INK, shadow=True)
        y += 40
        rows = max(0, (bottom - y) // 24)
        for line in self.log[-rows:] if rows else []:
            self._text(line, (SIDE_X, y), self.f_small, FELT_INK, shadow=True)
            y += 24

    # ----- prompt --------------------------------------------------------

    def _prompt_geometry(self, question, options):
        lines = self._wrap(question, self.f_body, SIDE_W - 28)
        height = 14 + len(lines) * 26 + 10 + len(options) * 52 + 6
        top = max(8, HEIGHT - height - 8)
        return pygame.Rect(SIDE_X, top, SIDE_W, height), lines

    def _draw_prompt(self, box, lines, options, mouse):
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
            x = rect.x + 14
            num = self.f_small.render(f"{i + 1}", True, BTN_INK)
            self.screen.blit(num, num.get_rect(midleft=(x, rect.centery)))
            x += 24
            shape, color = self._choice_icon(value)
            self._draw_icon(shape, color or BTN_INK, x + 10, rect.centery)
            x += 32
            text = self._truncate(label, self.f_body, rect.right - 14 - x)
            surf = self.f_body.render(text, True, BTN_INK)
            self.screen.blit(surf, surf.get_rect(midleft=(x, rect.centery)))
            buttons.append((rect, value))
            y += 52
        return buttons

    def _truncate(self, text, font, max_width):
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
        box, lines = (None, None)
        content_bottom = HEIGHT - 10
        if question:
            box, lines = self._prompt_geometry(question, options)
            content_bottom = box.top - 10
        if self.selected is not None:
            self._draw_inventory(y, self.game.players[self.selected],
                                 content_bottom)
        else:
            self._draw_log(y, content_bottom)
        buttons = []
        if question:
            buttons = self._draw_prompt(box, lines, options, mouse)
        return buttons, player_rects

    def render(self):
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
        skip = False
        board_rect = pygame.Rect(BOARD_X, BOARD_Y, BOARD_PX, BOARD_PX)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise QuitGame
            # Only a click *on the board* skips animations -- clicks elsewhere
            # (e.g. on player inventory panels) must not, so they stay usable
            # while AI turns animate. Any key press still skips.
            if event.type == pygame.KEYDOWN:
                skip = True
            elif (event.type == pygame.MOUSEBUTTONDOWN
                  and board_rect.collidepoint(event.pos)):
                skip = True
        self._draw_scene()
        pygame.display.flip()
        self.clock.tick(60)
        return skip

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
        skipped = False
        while True:
            t = (time.time() - start) / duration
            if t >= 1:
                break
            ease = t * t * (3 - 2 * t)
            v = from_pos + steps * ease
            self.vpos[name] = v
            self.hop[name] = math.sin((v % 1.0) * math.pi) * hop_h
            if self._frame():
                skipped = True
                break
        self.hop[name] = 0.0
        self._snap(player)
        if not skipped:
            self._animate_land(player)

    def _animate_land(self, player):
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
            if self._frame():
                break
        self.hop[name] = 0.0

    def _animate_dice(self, player, final):
        if self._auto is not None:
            self._set_roll(player, final)
            return
        skipped = False
        for _ in range(11):
            dice = (random.randint(1, 6), random.randint(1, 6))
            self._set_roll(player, dice)
            self.board_dice = dice
            if self._frame():
                skipped = True
                break
            pygame.time.wait(38)
        self._set_roll(player, final)
        self.board_dice = final
        for _ in range(0 if skipped else 14):
            if self._frame():
                break
            pygame.time.wait(28)
        self.board_dice = None
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
        self.add_log(f"{pile}: {card.text}")
        # AI turns don't block: log the card and pause briefly so it's watchable.
        if self._current_is_ai():
            self._ai_pause(0.9)
            return
        self.ask(f"{pile} card — {card.text}", [("Continue", "ok")])

    def _inform(self, message):
        self.add_log(message)
        # AI turns don't block on event acknowledgements; the log records them.
        if self._current_is_ai():
            return
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

    def _tradeable(self, owner):
        return [t for t in owner.properties
                if self.game.can_trade_property(t)]

    def _can_trade(self, player):
        others = [p for p in self.game.active_players() if p is not player]
        if not others:
            return False
        return any(self._tradeable(p) for p in [player, *others])

    def _can_manage(self, player):
        return bool(self._buildable(player) or self._sellable(player)
                    or self._mortgageable(player)
                    or self._unmortgageable(player)
                    or self._can_trade(player))

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

    def _manage_menu(self, player, jail_exit=False):
        while True:
            options = []
            if jail_exit and player.balance >= 50:
                options.append(("Pay $50 to leave jail", "pay"))
            if jail_exit and player.jail_cards:
                options.append(("Use Get Out of Jail Free card", "card"))
            if self._buildable(player):
                options.append(("Build a house", "build"))
            if self._sellable(player):
                options.append(("Sell a house", "sell"))
            if self._mortgageable(player):
                options.append(("Mortgage a property", "mortgage"))
            if self._unmortgageable(player):
                options.append(("Unmortgage a property", "unmortgage"))
            if self._can_trade(player):
                options.append(("Propose a trade", "trade"))
            options.append(("Done managing", "done"))
            choice = self.ask(
                f"{player.name}: manage properties (${player.balance})", options)
            if choice in ("pay", "card"):
                return choice
            if choice == "build":
                self._build_flow(player)
            elif choice == "sell":
                self._sell_flow(player)
            elif choice == "mortgage":
                self._mortgage_flow(player)
            elif choice == "unmortgage":
                self._unmortgage_flow(player)
            elif choice == "trade":
                self._trade_flow(player)
            else:
                return None

    # ----- trading -------------------------------------------------------

    def _cash_label(self, initiator, partner, cash):
        if cash > 0:
            return f"{initiator.name} pays ${cash}"
        if cash < 0:
            return f"{partner.name} pays ${-cash}"
        return "no cash"

    def _trade_flow(self, player):
        partners = [p for p in self.game.active_players() if p is not player]
        if not partners or self._auto is not None:
            return
        state = {
            "partner": partners[0],
            "give": set(),
            "receive": set(),
            "cash_you": "",
            "cash_them": "",
            "active": None,
        }
        while True:
            mouse = pygame.mouse.get_pos()
            hot = self._draw_trade_dialog(player, partners, state, mouse)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise QuitGame
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self._handle_trade_click(player, state, hot, event.pos):
                        return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    self._handle_trade_key(state, event)
            self.clock.tick(60)

    def _handle_trade_click(self, player, state, hot, pos):
        for rect, partner in hot["partners"]:
            if rect.collidepoint(pos) and state["partner"] is not partner:
                state["partner"] = partner
                state["receive"] = set()
                state["cash_them"] = ""
                return False
        for rect, tile in hot["give"] + hot["receive"]:
            if rect.collidepoint(pos):
                target = state["give"] if (rect, tile) in hot["give"] \
                    else state["receive"]
                target.discard(tile) if tile in target else target.add(tile)
                return False
        if hot["cash_you"].collidepoint(pos):
            state["active"] = "you"
        elif hot["cash_them"].collidepoint(pos):
            state["active"] = "them"
        elif hot["propose"].collidepoint(pos):
            return self._finalize_trade(player, state)
        elif hot["cancel"].collidepoint(pos):
            return True
        else:
            state["active"] = None
        return False

    def _handle_trade_key(self, state, event):
        if state["active"] is None:
            return
        key = "cash_you" if state["active"] == "you" else "cash_them"
        if event.key == pygame.K_BACKSPACE:
            state[key] = state[key][:-1]
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            state["active"] = None
        elif event.unicode.isdigit() and len(state[key]) < 6:
            state[key] = (state[key] + event.unicode).lstrip("0") or ""

    def _finalize_trade(self, player, state):
        partner = state["partner"]
        give = [t for t in player.properties if t in state["give"]]
        receive = [t for t in partner.properties if t in state["receive"]]
        cash = int(state["cash_you"] or 0) - int(state["cash_them"] or 0)
        if not give and not receive and cash == 0:
            return False

        gnames = ", ".join(t.name for t in give) or "nothing"
        rnames = ", ".join(t.name for t in receive) or "nothing"

        if self._is_ai(partner):
            # The AI evaluates the offer with its valuation formula rather than
            # being asked. From the AI's perspective it gains the properties the
            # proposer gives up, loses the ones it gives away, and its cash
            # changes by +cash (the proposer pays it ``cash``).
            decider = self._ai_deciders[partner.name]
            accepted, value = decider.evaluate_trade(
                partner, player, give, receive, cash)
            if not accepted:
                self.add_log(f"{partner.name} [AI] declined {player.name}'s "
                             f"trade (value {value:+.0f}).")
                return True
            if self.game.execute_trade(player, partner, give, receive, cash):
                self.add_log(f"{partner.name} [AI] accepted {player.name}'s "
                             f"trade (value {value:+.0f}).")
                return True
            self.ask("Trade couldn't be completed (the cash payer can't "
                     "afford it).", [("OK", "ok")])
            return False

        accept = self.ask(
            f"{partner.name}: {player.name} proposes a trade. You give "
            f"{rnames}; you get {gnames}; "
            f"{self._cash_label(player, partner, cash)}. Accept?",
            [("Accept", "yes"), ("Decline", "no")])
        if accept != "yes":
            self.add_log(f"{partner.name} declined {player.name}'s trade.")
            return True
        if self.game.execute_trade(player, partner, give, receive, cash):
            self.add_log(f"{player.name} and {partner.name} made a trade.")
            return True
        self.ask("Trade couldn't be completed (the cash payer can't afford it).",
                 [("OK", "ok")])
        return False

    # ----- trade dialog rendering ----------------------------------------

    def _draw_trade_dialog(self, player, partners, state, mouse):
        self._draw_scene()
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 40, 26, 190))
        self.screen.blit(overlay, (0, 0))

        dlg = pygame.Rect(BOARD_X + 6, BOARD_Y + 6, BOARD_PX - 12, BOARD_PX - 12)
        self._panel(dlg, PANEL)
        partner = state["partner"]
        pad = 22

        title = self.f_title.render("Propose a Trade", True, INK)
        self.screen.blit(title, title.get_rect(midtop=(dlg.centerx, dlg.y + 14)))

        hot = {"partners": [], "give": [], "receive": []}
        px = dlg.x + pad
        py = dlg.y + 58
        self._text("With:", (px, py + 6), self.f_small, MUTED)
        px += 56
        for p in partners:
            fill = player_color(p.name) if p is partner else PANEL_ALT
            ink = contrast_text(fill) if p is partner else INK
            label = self.f_body.render(p.name, True, ink)
            rect = pygame.Rect(px, py, label.get_width() + 28, 36)
            pygame.draw.rect(self.screen, fill, rect, border_radius=8)
            pygame.draw.rect(self.screen, EDGE, rect, 1, border_radius=8)
            self.screen.blit(label, label.get_rect(center=rect.center))
            hot["partners"].append((rect, p))
            px += rect.width + 10

        colw = (dlg.w - 3 * pad) // 2
        col_top = dlg.y + 112
        cash_y = dlg.bottom - 132
        chips_bottom = cash_y - 28
        left_x = dlg.x + pad
        right_x = dlg.x + pad + colw + pad

        hot["give"] = self._draw_trade_column(
            player, state["give"], left_x, col_top, colw, chips_bottom,
            f"You give — {player.name}", player.balance)
        hot["receive"] = self._draw_trade_column(
            partner, state["receive"], right_x, col_top, colw, chips_bottom,
            f"You receive — {partner.name}", partner.balance)

        hot["cash_you"] = self._draw_cash_box(
            "Your cash to add", state["cash_you"], left_x, cash_y, colw,
            state["active"] == "you")
        hot["cash_them"] = self._draw_cash_box(
            f"{partner.name}'s cash to add", state["cash_them"], right_x, cash_y,
            colw, state["active"] == "them")

        net = int(state["cash_you"] or 0) - int(state["cash_them"] or 0)
        summary = "Net: " + self._cash_label(player, partner, net)
        self._text(summary, (dlg.x + pad, dlg.bottom - 50), self.f_body, INK)
        hot["cancel"] = self._draw_dialog_button(
            "Cancel", dlg.right - pad - 150, dlg.bottom - 58, 150, mouse,
            (150, 70, 70))
        hot["propose"] = self._draw_dialog_button(
            "Propose", dlg.right - pad - 312, dlg.bottom - 58, 150, mouse, BTN)
        return hot

    def _draw_trade_column(self, owner, selected, x, y, w, bottom, header, bal):
        self._text(header, (x, y), self.f_head, INK)
        self._text(f"Balance ${bal}", (x, y + 28), self.f_small, MUTED)
        cy = y + 56
        rows = []
        tradeable = self._tradeable(owner)
        if not tradeable:
            self._text("Nothing tradeable.", (x, cy), self.f_small, MUTED)
            return rows
        for prop in tradeable:
            if cy > bottom - 30:
                self._text("…more", (x, cy), self.f_small, MUTED)
                break
            rect = pygame.Rect(x, cy, w, 30)
            color = self._property_color(prop)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            if prop in selected:
                pygame.draw.rect(self.screen, ACCENT, rect, 3, border_radius=5)
            else:
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1, border_radius=5)
            name = prop.name + (" (mortgaged)" if prop.mortgaged else "")
            name = self._truncate(name, self.f_small, w - 16)
            self.screen.blit(
                self.f_small.render(name, True, contrast_text(color)),
                (x + 8, cy + 5))
            rows.append((rect, prop))
            cy += 34
        return rows

    def _draw_cash_box(self, label, value, x, y, w, active):
        self._text(label, (x, y - 22), self.f_small, INK)
        box = pygame.Rect(x, y, w, 40)
        pygame.draw.rect(self.screen, (255, 255, 255), box, border_radius=6)
        pygame.draw.rect(self.screen, ACCENT if active else EDGE, box,
                         3 if active else 1, border_radius=6)
        shown = "$" + (value or "0") + ("|" if active else "")
        self.screen.blit(self.f_body.render(shown, True, INK),
                         (box.x + 12, box.y + 8))
        return box

    def _draw_dialog_button(self, label, x, y, w, mouse, color):
        rect = pygame.Rect(x, y, w, 44)
        hover = rect.collidepoint(mouse) if mouse else False
        shade = tuple(min(255, c + 24) for c in color) if hover else color
        pygame.draw.rect(self.screen, shade, rect, border_radius=8)
        surf = self.f_body.render(label, True, BTN_INK)
        self.screen.blit(surf, surf.get_rect(center=rect.center))
        return rect

    # ----- shortfall / fund-raising --------------------------------------

    def _on_shortfall(self, player, amount):
        """Dispatches a shortfall to the AI liquidator or the human raise-funds UI."""
        if self._is_ai(player):
            self._ai_deciders[player.name].liquidate_loop(player, amount)
        else:
            self._raise_funds(player, amount)

    def _raise_funds(self, player, amount):
        while player.balance < amount:
            options = []
            if self._sellable(player):
                options.append(("Sell a house", "sell"))
            if self._mortgageable(player):
                options.append(("Mortgage a property", "mortgage"))
            if not options:
                self.ask(
                    f"{player.name} owes ${amount} but has only "
                    f"${player.balance} and nothing left to sell — bankrupt!",
                    [("Continue", "ok")])
                return
            options.append(("Give up (declare bankruptcy)", "give_up"))
            choice = self.ask(
                f"{player.name} owes ${amount} but has only ${player.balance}. "
                f"Raise cash to avoid bankruptcy.", options)
            if choice == "sell":
                self._sell_flow(player)
            elif choice == "mortgage":
                self._mortgage_flow(player)
            else:
                return

    def _set_roll(self, player, dice):
        self.roll_display = {"name": player.name, "dice": dice}

    # ----- turn flow -----------------------------------------------------

    def _turn_menu(self, player):
        """Returns the jail/roll choice. AI players skip the UI prompt."""
        if self._is_ai(player):
            return self._ai_turn_start(player)

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
                jail_action = self._manage_menu(player, jail_exit=player.in_jail)
                if jail_action is not None:
                    return jail_action
            else:
                return choice

    def _ai_turn_start(self, player):
        """Runs an AI player's pre-roll phase and returns their jail/roll choice."""
        ai = self._ai_deciders[player.name]
        if player.in_jail:
            return ai.jail_choice(player)
        ai.manage_loop(player)
        return "roll"

    def _post_roll_manage(self, player):
        """After the roll, offer one more management pass before ending the turn."""
        if self._is_ai(player):
            self._ai_deciders[player.name].manage_loop(player)
            return
        if not self._can_manage(player):
            return
        choice = self.ask(
            f"{player.name}: manage properties before ending your turn? "
            f"(${player.balance})",
            [("End turn", "end"), ("Manage properties", "manage")])
        if choice == "manage":
            self._manage_menu(player)

    def _resolve_and_sync(self, player):
        self.game.resolve_tile(player)
        self._snap(player)

    def _play_rolls(self, player):
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
                if not self._is_ai(player):
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
            if not self._is_ai(player):
                self.ask(f"{msg} Roll the dice to take your turn.",
                         [("Roll dice", "roll")])
            self._play_rolls(player)
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
        else:
            self._animate_dice(player, dice)
            self.add_log(f"{player.name} rolled {dice[0]} + {dice[1]} — "
                         f"no doubles, still in jail.")

    def _end_turn(self, player):
        player.double_count = 0
        self.game.advance_turn()

    def _show_result(self):
        winner = self.game.winner()
        message = f"{winner.name} wins!" if winner else "Game over."
        self.add_log(message)
        self.ask(message, [("Quit", "quit")])

    # ----- main loop -----------------------------------------------------

    def run(self, max_turns=10000):
        try:
            turns = 0
            while not self.game.is_over() and turns < max_turns:
                player = self.game.players[self.game.current_player]
                choice = self._turn_menu(player)
                if player.in_jail:
                    self._handle_jail(player, choice)
                else:
                    self._play_rolls(player)
                if not player.bankrupt:
                    self._post_roll_manage(player)
                self._end_turn(player)
                self.selected = None
                turns += 1
            self._show_result()
        except QuitGame:
            pass
        finally:
            pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Shows the setup screen, then launches the game."""
    parser = argparse.ArgumentParser(description="Monopoly — pygame UI")
    parser.add_argument(
        "--model", default=None,
        help="path to a trained MaskablePPO model .zip "
             "(default: runs/monopoly_ppo.zip if it exists)")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="seed for dice, card shuffles, turn order, and AI sampling "
             "(default: a fresh random seed each game; printed so you can "
             "replay a game with --seed)")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="make AI players act greedily (same move for the same state). "
             "By default the AI samples its policy so games vary.")
    args, _ = parser.parse_known_args()

    # Find a model to offer AI players.
    model_path = args.model
    if model_path is None:
        for candidate in ("runs/monopoly_ppo.zip",
                          "runs/sp_checkpoints/monopoly_selfplay_400000_steps.zip"):
            if os.path.exists(candidate):
                model_path = candidate
                break
    model_available = model_path is not None

    pygame.init()
    pygame.key.set_repeat(300, 40)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    config = _run_setup(screen, clock, model_available)
    if config is None:
        pygame.quit()
        return

    # Determine which players are AI.
    ai_names = {name for name, t in config["types"].items() if t == "ai"}

    # Load the model once if any AI player was chosen.
    model = None
    if ai_names and model_path:
        try:
            from sb3_contrib import MaskablePPO
            print(f"Loading model from {model_path} …")
            model = MaskablePPO.load(model_path, device="cpu")
        except Exception as exc:
            print(f"ERROR: could not load model ({exc}).")
            _show_error(screen, clock, [
                "Could not load the AI model.",
                str(exc)[:60],
                "",
                "Make sure you run inside the project venv:",
                "  source .venv/bin/activate && python play_gui.py",
                "",
                "Starting a human-only game instead.",
            ])
            ai_names = set()

    # Seed every source of randomness for this game: dice and card shuffles
    # (both draw from the stdlib ``random`` module), the turn-order shuffle, and
    # the AI's policy sampling (torch). A fresh entropy-based seed each game
    # gives variety; ``--seed`` makes a game reproducible. We print it so any
    # game can be replayed.
    #
    # The seed is drawn from ``SystemRandom`` (OS entropy), NOT ``random``,
    # because loading the MaskablePPO model reseeds the global ``random`` module
    # to its training seed -- so ``random.randrange`` here would return the same
    # value on every launch and every AI game would play out identically.
    seed = args.seed if args.seed is not None \
        else random.SystemRandom().randrange(2 ** 31)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    print(f"Game seed: {seed}  (replay with --seed {seed})")

    # Build players and (optionally) shuffle turn order.
    from engine.game import Game
    from models.board import Board
    from models.player import Player
    from data.decks import build_chance_deck, build_community_deck
    from data.board_tiles import build_board_tiles

    players = [Player(n) for n in ("Red", "Blue", "Green", "Yellow")]
    if config["randomize"]:
        random.shuffle(players)

    game = Game(players, Board(build_board_tiles()),
                build_chance_deck(), build_community_deck())

    # Create one GUIAIDecider per AI seat (all share the same model).
    ai_deciders = {}
    if model is not None:
        from ui.ai_player import GUIAIDecider
        for name in ai_names:
            ai_deciders[name] = GUIAIDecider(
                num_players=4, model=model, deterministic=args.deterministic)

    MonopolyApp(game, ai_deciders=ai_deciders, screen=screen).run()


if __name__ == "__main__":
    main()
