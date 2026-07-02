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

# Affirmative / negative action buttons (buy / auction, accept / reject).
POS_GREEN = (39, 158, 70)
NEG_RED = (200, 62, 55)
CARD_BG = (252, 251, 247)
MONEY_BG = (222, 246, 224)
MONEY_INK = (22, 110, 52)

# Log-line colours by action type -- tuned to read on the dark-green felt so
# similar events are easy to pick out at a glance.
LOG_BUY = (124, 214, 138)       # buying property / income (money in)
LOG_RENT = (240, 138, 128)      # rent, fees, tax (money out)
LOG_BUILD = (128, 194, 255)     # building houses / hotels
LOG_SELL = (240, 186, 96)       # selling houses back to the bank
LOG_MORTGAGE = (226, 198, 108)  # mortgage / unmortgage
LOG_TRADE = (204, 164, 236)     # trades
LOG_AUCTION = (116, 214, 208)   # auctions & bids
LOG_JAIL = (226, 158, 112)      # jail events
LOG_BANKRUPT = (240, 96, 86)    # bankruptcy / elimination

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
        # Row offset for the open inventory panel, so a player owning more
        # properties than fit on screen can be scrolled through (mouse wheel).
        self._inv_scroll = 0
        # Lines the game log is scrolled back from its newest entry (0 = pinned
        # to the bottom, following new events); the wheel scrolls it when no
        # inventory panel is open. Clamped to the log length in ``_draw_log``.
        self._log_scroll = 0
        # Player-panel click targets from the last drawn scene, so animation
        # frames (the only input path during all-AI turns) can toggle the
        # inventory view too.
        self._player_rects = []
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
            ai.trade_arbiter = self._ai_trade_arbiter  # resolve AI-proposed trades

        # Wire per-player purchase and auction-bid hooks.
        for player in game.players:
            if player.name in self._ai_deciders:
                ai = self._ai_deciders[player.name]
                player.decide_purchase = (
                    lambda prop, _ai=ai, _p=player: _ai.purchase_decision(_p, prop))
                player.decide_bid = (
                    lambda prop, min_bid=0, _ai=ai, _p=player:
                    _ai.bid_choice(_p, prop, min_bid))
            else:
                player.decide_purchase = (
                    lambda prop, _p=player: self._prompt_purchase(_p, prop))
                player.decide_bid = (
                    lambda prop, min_bid=0, _p=player:
                    self._prompt_bid(_p, prop, min_bid))

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
        self.log.append((message, self._log_color(message)))
        self.log = self.log[-200:]
        # Keep the same lines in view when scrolled back into history; when
        # pinned to the bottom (scroll 0) stay following the newest events.
        if self._log_scroll:
            self._log_scroll = min(self._log_scroll + 1, len(self.log) - 1)

    def _log_color(self, message):
        """Maps a log line to a colour by the kind of action it describes, so
        similar events (rent, builds, trades, ...) share a hue in the log."""
        m = message.lower()
        if "bankrupt" in m or "eliminated" in m:
            return LOG_BANKRUPT
        if "auction" in m or "bid" in m:
            return LOG_AUCTION
        if "trade" in m:
            return LOG_TRADE
        if "built" in m or "build" in m:
            return LOG_BUILD
        if "sold a house" in m or "sell" in m:
            return LOG_SELL
        if "mortgage" in m:  # covers "mortgaged" and "lifted the mortgage"
            return LOG_MORTGAGE
        if "jail" in m:
            return LOG_JAIL
        if ("rent" in m or "pays" in m or "paid" in m or "tax" in m
                or " fee" in m or "owes" in m):
            return LOG_RENT
        if ("bought" in m or "buy" in m or "collect" in m or "salary" in m
                or "passed go" in m or "received" in m or "won" in m):
            return LOG_BUY
        return FELT_INK

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

    def _property_worth(self, prop):
        """Cash value of one property: mortgage value if mortgaged, else price,
        plus any house investment. Matches the per-property term in
        ``_net_worth`` so the inventory values sum to the player's property net
        worth."""
        worth = prop.mortgage_value if prop.mortgaged else prop.price
        if isinstance(prop, StreetProperty):
            worth += prop.houses * prop.house_cost()
        return worth

    def _property_sort_key(self, prop):
        """Sort key that groups properties by colour set (board order: browns
        through dark blue, then railroads, then utilities) and orders within a
        group by price -- used everywhere properties are listed."""
        order = list(GROUP_COLORS)
        if isinstance(prop, StreetProperty):
            rank = order.index(prop.color) if prop.color in order else len(order)
        elif isinstance(prop, Railroad):
            rank = len(order)
        else:  # Utility
            rank = len(order) + 1
        return (rank, prop.price, prop.name)

    def _sorted_props(self, props):
        """Properties grouped by colour then price (see ``_property_sort_key``)."""
        return sorted(props, key=self._property_sort_key)

    ROW_H = 34  # pixel height of one inventory row

    def _draw_inventory(self, y, player, bottom):
        self._text(f"{player.name}'s inventory", (SIDE_X, y), self.f_head,
                   FELT_INK, shadow=True)
        props = self._sorted_props(player.properties)

        # How many rows fit, and how far we may scroll, given the space below
        # the header. Clamp the stored scroll here so the wheel handler can
        # never push it past the last page.
        list_top = y + 56
        visible = max(1, (bottom - list_top) // self.ROW_H)
        max_scroll = max(0, len(props) - visible)
        self._inv_scroll = max(0, min(self._inv_scroll, max_scroll))
        scroll = self._inv_scroll

        if props:
            total = sum(self._property_worth(p) for p in props)
            subline = f"${total} in property  ·  click the player again to close"
            if max_scroll:
                first, last = scroll + 1, min(scroll + visible, len(props))
                subline = (f"${total} in property  ·  showing {first}-{last} of "
                           f"{len(props)}  ·  scroll to see more")
        else:
            subline = "(click the player again to close)"
        self._text(subline, (SIDE_X, y + 28), self.f_small, FELT_SUB,
                   shadow=True)
        y = list_top
        if not props:
            self._text("No properties owned.", (SIDE_X, y), self.f_body,
                       FELT_SUB, shadow=True)
            return

        # A scrollbar on the right edge indicates position when overflowing;
        # nudge the row text left of it so they don't overlap.
        text_right = SIDE_X + SIDE_W - (12 if max_scroll else 0)
        for prop in props[scroll:scroll + visible]:
            self._draw_property_chip(prop, SIDE_X, y)
            # Houses/hotel/mortgage shown as icons just right of the name chip.
            chip_w = self.f_small.size(prop.name)[0] + 8
            cy = y + self.f_small.get_height() // 2 + 1
            self._draw_buildings(prop, SIDE_X + chip_w + 6, cy)
            _, rent = self._rent_line(prop)
            info = f"${self._property_worth(prop)}  ·  {rent}"
            w = self.f_small.size(info)[0]
            self._text(info, (text_right - w, y), self.f_small, FELT_SUB,
                       shadow=True)
            y += self.ROW_H

        if max_scroll:
            self._draw_inventory_scrollbar(list_top, visible, len(props), scroll)

    def _draw_inventory_scrollbar(self, top, visible, total, scroll, row_h=None):
        """Draws a slim scrollbar for a scrolling side list along the panel edge
        (the inventory, or the log when ``row_h`` overrides the row height)."""
        track_h = visible * (row_h or self.ROW_H)
        x = SIDE_X + SIDE_W - 5
        pygame.draw.rect(self.screen, FELT_SUB,
                         pygame.Rect(x, top, 3, track_h), border_radius=2)
        thumb_h = max(18, int(track_h * visible / total))
        thumb_y = top + int((track_h - thumb_h) * scroll / max(1, total - visible))
        pygame.draw.rect(self.screen, FELT_INK,
                         pygame.Rect(x, thumb_y, 3, thumb_h), border_radius=2)

    def _scroll_inventory(self, dy):
        """Scrolls the open inventory panel by ``dy`` wheel notches (up = +)."""
        if self.selected is not None:
            self._inv_scroll = max(0, self._inv_scroll - dy)

    def _property_color(self, prop):
        if isinstance(prop, StreetProperty):
            return GROUP_COLORS.get(prop.color, DEFAULT_COLOR)
        if isinstance(prop, Railroad):
            return RAILROAD_COLOR
        if isinstance(prop, Utility):
            return UTILITY_COLOR
        return DEFAULT_COLOR

    def _choice_icon(self, value):
        # Some prompts (auction bids) use integer *amounts* as option values,
        # which must NOT be read as board indices -- only treat an int as a tile
        # position when it is actually in range and points at a property.
        if isinstance(value, int) and 0 <= value < len(self.game.board.tiles):
            tile = self.game.board.tiles[value]
            if isinstance(tile, Property):
                return "swatch", self._property_color(tile)
        if value is None:
            return "x", None
        if isinstance(value, int):
            return "coin", ICON_COLORS.get("coin")
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
        elif shape == "mortgage":
            pygame.draw.circle(scr, color, (cx, cy), h, 2)
            pygame.draw.line(scr, color, (cx - h + 2, cy + h - 2),
                             (cx + h - 2, cy - h + 2), 2)
        else:
            pygame.draw.circle(scr, color, (cx, cy), 4)

    def _draw_house(self, cx, cy, color, s=10):
        """A small house glyph (roofed square) centred at ``(cx, cy)``."""
        h = s // 2
        body = pygame.Rect(cx - h, cy - h + 2, s, s - 2)
        pygame.draw.polygon(self.screen, color,
                            [(cx - h - 1, cy - h + 2),
                             (cx + h + 1, cy - h + 2), (cx, cy - h - 3)])
        pygame.draw.rect(self.screen, color, body)
        pygame.draw.rect(self.screen, INK, body, 1)

    HOUSE_GREEN = (46, 160, 80)
    HOTEL_RED = (204, 58, 58)
    MTG_TAN = (150, 120, 60)

    def _draw_buildings(self, prop, x, cy):
        """Draws status icons for ``prop`` at vertical centre ``cy`` starting at
        ``x``: a mortgage badge, a red hotel, or up to four green houses.
        Returns the x cursor after the icons (== ``x`` when there's nothing)."""
        if getattr(prop, "mortgaged", False):
            self._draw_icon("mortgage", self.MTG_TAN, x + 8, cy, s=15)
            return x + 18
        if not isinstance(prop, StreetProperty) or prop.houses == 0:
            return x
        if prop.houses >= 5:  # hotel
            self._draw_house(x + 8, cy, self.HOTEL_RED, s=14)
            return x + 18
        cx = x + 6
        for _ in range(prop.houses):
            self._draw_house(cx, cy, self.HOUSE_GREEN, s=10)
            cx += 11
        return cx + 1

    def _draw_property_chip(self, prop, x, y):
        color = self._property_color(prop)
        tw = self.f_small.size(prop.name)[0]
        chip = pygame.Rect(x - 4, y - 2, tw + 12, self.f_small.get_height() + 4)
        pygame.draw.rect(self.screen, color, chip, border_radius=4)
        pygame.draw.rect(self.screen, (0, 0, 0), chip, 1, border_radius=4)
        self.screen.blit(
            self.f_small.render(prop.name, True, contrast_text(color)),
            (x + 2, y))

    # ----- card popups & modals ------------------------------------------

    def _fit_font(self, text, max_w, size, bold=True):
        """Returns the largest bold font (down to 12px) that fits ``text``."""
        while size > 12:
            font = self._font(size, bold=bold)
            if font.size(text)[0] <= max_w:
                return font
            size -= 2
        return self._font(12, bold=bold)

    def _dim_scene(self, alpha=190):
        """Draws the live board/panels, then a dark scrim, ready for a modal."""
        self._draw_scene()
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((8, 40, 26, alpha))
        self.screen.blit(overlay, (0, 0))

    def _run_overlay_modal(self, draw):
        """Blocks on a modal drawn over a dimmed board.

        ``draw(mouse)`` renders one full frame (it should call ``_dim_scene``
        first) and returns a list of ``(rect, value)`` hot buttons; the chosen
        value is returned. Number keys ``1..n`` map to the buttons in order.
        Mouse-wheel notches are forwarded to ``draw`` via ``self._modal_wheel``
        so scrollable modals can respond.
        """
        self._modal_wheel = 0
        while True:
            mouse = pygame.mouse.get_pos()
            buttons = draw(mouse)
            pygame.display.flip()
            self._modal_wheel = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise QuitGame
                if event.type == pygame.MOUSEWHEEL:
                    self._modal_wheel = event.y
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, value in buttons:
                        if rect.collidepoint(event.pos):
                            return value
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "__cancel__"
                    idx = event.key - pygame.K_1
                    if 0 <= idx < len(buttons):
                        return buttons[idx][1]
            self.clock.tick(60)

    def _draw_title_deed(self, rect, prop):
        """Renders a classic Monopoly title-deed card for ``prop`` in ``rect``."""
        pygame.draw.rect(self.screen, CARD_BG, rect, border_radius=12)
        pygame.draw.rect(self.screen, INK, rect, 2, border_radius=12)
        pad = 18
        inner_w = rect.w - 2 * pad
        color = self._property_color(prop)
        band = pygame.Rect(rect.x + pad, rect.y + pad, inner_w, 82)
        pygame.draw.rect(self.screen, color, band)
        pygame.draw.rect(self.screen, INK, band, 2)
        bink = contrast_text(color)
        td = self.f_small.render("TITLE DEED", True, bink)
        self.screen.blit(td, td.get_rect(midtop=(band.centerx, band.y + 8)))
        nfont = self._fit_font(prop.name.upper(), inner_w - 16, 26)
        nm = nfont.render(prop.name.upper(), True, bink)
        self.screen.blit(nm, nm.get_rect(center=(band.centerx, band.y + 52)))

        x0, x1 = rect.x + pad, rect.right - pad
        y = [band.bottom + 16]

        def row(label, value, bold=False, gap=25):
            font = self._font(19, bold=True) if bold else self.f_small
            self.screen.blit(font.render(label, True, INK), (x0, y[0]))
            if value is not None:
                vs = font.render(value, True, INK)
                self.screen.blit(vs, vs.get_rect(topright=(x1, y[0])))
            y[0] += gap

        def rule():
            pygame.draw.line(self.screen, EDGE, (x0, y[0]), (x1, y[0]))
            y[0] += 10

        if isinstance(prop, StreetProperty):
            rt = prop.rent_table
            row("RENT", f"${rt[0]}", bold=True)
            row("With Color Set", f"${rt[0] * 2}")
            for i in range(1, 5):
                row(f"With {i} House" + ("s" if i > 1 else ""), f"${rt[i]}")
            row("With HOTEL", f"${rt[5]}", bold=True)
            rule()
            hc = prop.house_cost()
            row("Houses cost", f"${hc} ea.")
            row("Hotel (4 houses +)", f"${hc}")
            rule()
            row("Mortgage Value", f"${prop.mortgage_value}")
        elif isinstance(prop, Railroad):
            row("RAILROAD", None, bold=True)
            y[0] += 4
            for k in range(1, 5):
                row(f"If {k} R.R. owned", f"${25 * 2 ** k}")
            rule()
            row("Mortgage Value", f"${prop.mortgage_value}")
        else:  # Utility
            row("UTILITY", None, bold=True)
            y[0] += 4
            for line in self._wrap(
                    "If one Utility is owned, rent is 4 times the dice roll.",
                    self.f_small, inner_w):
                row(line, None, gap=22)
            y[0] += 6
            for line in self._wrap(
                    "If both Utilities are owned, rent is 10 times the roll.",
                    self.f_small, inner_w):
                row(line, None, gap=22)
            rule()
            row("Mortgage Value", f"${prop.mortgage_value}")

        pygame.draw.line(self.screen, EDGE, (x0, rect.bottom - 42),
                         (x1, rect.bottom - 42))
        pf = self._font(22, bold=True)
        self.screen.blit(pf.render("PRICE", True, MUTED), (x0, rect.bottom - 34))
        ps = pf.render(f"${prop.price}", True, INK)
        self.screen.blit(ps, ps.get_rect(topright=(x1, rect.bottom - 34)))

    def _draw_mini_card(self, rect, prop, selected=False):
        """A compact property card for lists: colour stripe + name on top, and a
        bottom row with building/mortgage icons (left) and the price (right), so
        no text sits over the coloured stripe. A gold ring marks selection."""
        pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=6)
        stripe = pygame.Rect(rect.x, rect.y, rect.w, 12)
        color = self._property_color(prop)
        pygame.draw.rect(self.screen, color, stripe,
                         border_top_left_radius=6, border_top_right_radius=6)
        namef = self._font(14)
        name = self._truncate(prop.name, namef, rect.w - 14)
        self.screen.blit(namef.render(name, True, INK), (rect.x + 7, rect.y + 14))
        # Bottom row (only when the card is tall enough): icons + price, hugging
        # the bottom edge so they never collide with the name.
        if rect.h >= 40:
            by = rect.bottom - 9
            self._draw_buildings(prop, rect.x + 3, by)
            pricef = self._font(13, bold=True)
            ps = pricef.render(f"${prop.price}", True, MUTED)
            self.screen.blit(ps, ps.get_rect(midright=(rect.right - 7, by)))
        pygame.draw.rect(self.screen, ACCENT if selected else INK, rect,
                         3 if selected else 1, border_radius=6)

    def _draw_money_card(self, rect, amount):
        """A green 'bill' chip showing a cash amount in a trade."""
        pygame.draw.rect(self.screen, MONEY_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, POS_GREEN, rect, 2, border_radius=6)
        s = self._font(22, bold=True).render(f"$ {amount}", True, MONEY_INK)
        self.screen.blit(s, s.get_rect(center=rect.center))

    # ----- purchase card -------------------------------------------------

    def _draw_purchase_card(self, player, prop, mouse):
        self._dim_scene()
        board_cx = BOARD_X + BOARD_PX // 2
        board_cy = BOARD_Y + BOARD_PX // 2
        card_w = 360
        card_h = 500 if isinstance(prop, StreetProperty) else 320
        card = pygame.Rect(board_cx - card_w // 2,
                           board_cy - card_h // 2 - 22, card_w, card_h)
        head = self.f_head.render(f"{player.name}, buy {prop.name}?", True,
                                  FELT_INK)
        self.screen.blit(head, head.get_rect(midbottom=(board_cx, card.y - 30)))
        bal = self.f_small.render(f"Balance ${player.balance}", True, FELT_SUB)
        self.screen.blit(bal, bal.get_rect(midbottom=(board_cx, card.y - 8)))
        self._draw_title_deed(card, prop)
        bw, gap = 168, 16
        by = card.bottom + 22
        if player.balance >= prop.price:
            buy = self._draw_dialog_button(f"Buy  ${prop.price}",
                                           board_cx - bw - gap // 2, by, bw,
                                           mouse, POS_GREEN)
            auc = self._draw_dialog_button("Auction", board_cx + gap // 2, by,
                                           bw, mouse, NEG_RED)
            return [(buy, "buy"), (auc, "decline")]
        # Short on cash: don't auto-auction -- offer to raise cash by
        # mortgaging / selling houses, or send the property to auction.
        short = prop.price - player.balance
        note = self.f_small.render(f"Short ${short} — raise cash to buy", True,
                                   NEG_RED)
        self.screen.blit(note, note.get_rect(midtop=(board_cx, by)))
        by += 26
        raise_btn = self._draw_dialog_button("Raise Cash",
                                             board_cx - bw - gap // 2, by, bw,
                                             mouse, POS_GREEN)
        auc = self._draw_dialog_button("Auction", board_cx + gap // 2, by, bw,
                                       mouse, NEG_RED)
        return [(raise_btn, "raise"), (auc, "decline")]

    # ----- trade offer (accept / reject) ---------------------------------

    def _confirm_trade(self, proposer, responder, give, receive, cash):
        """Shows ``responder`` a visual trade offer; returns True on accept.

        ``give`` are tiles ``proposer`` hands over, ``receive`` tiles the
        responder hands over, ``cash`` the amount the proposer pays the
        responder (negative means the responder pays).
        """
        if self._auto is not None:
            return False
        # Each side shows the player's *full* inventory so the responder can
        # weigh the deal in context; the tiles actually changing hands are
        # highlighted. Each column scrolls independently under the mouse.
        self._offer_scroll_l = 0
        self._offer_scroll_r = 0
        choice = self._run_overlay_modal(
            lambda mouse: self._draw_trade_offer(
                proposer, responder, give, receive, cash, mouse))
        return choice == "accept"

    def _draw_trade_offer(self, proposer, responder, give, receive, cash, mouse):
        self._dim_scene()
        dlg = pygame.Rect(BOARD_X + 24, BOARD_Y + 34, BOARD_PX - 48,
                          BOARD_PX - 68)
        self._panel(dlg, PANEL)
        title = self.f_title.render("Trade Offer", True, INK)
        self.screen.blit(title, title.get_rect(midtop=(dlg.centerx, dlg.y + 12)))
        sub = self.f_small.render(
            f"{responder.name}, do you accept?   ·   highlighted cards change "
            f"hands", True, MUTED)
        self.screen.blit(sub, sub.get_rect(midtop=(dlg.centerx, dlg.y + 46)))

        pad = 24
        colw = (dlg.w - 3 * pad) // 2
        col_top = dlg.y + 76
        col_bottom = dlg.bottom - 88
        left = pygame.Rect(dlg.x + pad, col_top, colw, col_bottom - col_top)
        right = pygame.Rect(dlg.right - pad - colw, col_top, colw,
                            col_bottom - col_top)

        # Route wheel notches (from the modal loop) to the hovered column.
        wheel = getattr(self, "_modal_wheel", 0)
        if wheel:
            if left.collidepoint(mouse):
                self._offer_scroll_l -= wheel
            elif right.collidepoint(mouse):
                self._offer_scroll_r -= wheel

        lmax = self._draw_trade_side(left, proposer, set(give),
                                     cash if cash > 0 else 0, "gives",
                                     self._offer_scroll_l)
        rmax = self._draw_trade_side(right, responder, set(receive),
                                     -cash if cash < 0 else 0, "you give",
                                     self._offer_scroll_r)
        self._offer_scroll_l = max(0, min(self._offer_scroll_l, lmax))
        self._offer_scroll_r = max(0, min(self._offer_scroll_r, rmax))

        mid = (dlg.centerx, (col_top + col_bottom) // 2)
        pygame.draw.circle(self.screen, PANEL, mid, 24)
        pygame.draw.circle(self.screen, EDGE, mid, 24, 1)
        self._draw_icon("swap", INK, mid[0], mid[1], s=26)

        bw, gap = 190, 20
        by = dlg.bottom - 62
        accept = self._draw_dialog_button("Accept Trade",
                                          dlg.centerx - bw - gap // 2, by, bw,
                                          mouse, POS_GREEN)
        reject = self._draw_dialog_button("Reject Trade", dlg.centerx + gap // 2,
                                          by, bw, mouse, NEG_RED)
        return [(accept, "accept"), (reject, "reject")]

    def _draw_trade_side(self, rect, player, offered, cash, verb, scroll):
        """One party's panel in the offer dialog: header, their *whole*
        inventory as mini-cards (tiles in ``offered`` ringed with the accent
        colour), and a cash chip pinned at the bottom. Returns the max scroll."""
        self._panel(rect, PANEL_ALT)
        color = player_color(player.name)
        hdr = pygame.Rect(rect.x, rect.y, rect.w, 40)
        pygame.draw.rect(self.screen, color, hdr,
                         border_top_left_radius=10, border_top_right_radius=10)
        token = pygame.transform.smoothscale(self.tokens[player.name], (26, 26))
        self.screen.blit(token, (hdr.x + 8, hdr.y + 7))
        nm = self.f_body.render(player.name, True, contrast_text(color))
        self.screen.blit(nm, nm.get_rect(midleft=(hdr.x + 42, hdr.centery)))
        vb = self.f_small.render(verb, True, contrast_text(color))
        self.screen.blit(vb, vb.get_rect(midright=(hdr.right - 10, hdr.centery)))

        x, cw = rect.x + 12, rect.w - 24
        list_top = hdr.bottom + 10
        cash_h = 46 if cash > 0 else 0
        list_bottom = rect.bottom - 12 - (cash_h + 10 if cash_h else 0)

        inventory = self._sorted_props(player.properties)
        row_h = 54
        slots = max(1, (list_bottom - list_top) // row_h)
        max_scroll = max(0, len(inventory) - slots)
        scroll = max(0, min(scroll, max_scroll))

        if not inventory:
            self._text("no properties", (x, list_top), self.f_small, MUTED)
        y = list_top
        for prop in inventory[scroll:scroll + slots]:
            card = pygame.Rect(x, y, cw, 48)
            self._draw_mini_card(card, prop, selected=prop in offered)
            y += row_h
        if max_scroll:
            lo, hi = scroll + 1, min(scroll + slots, len(inventory))
            self._text(f"scroll · {lo}-{hi}/{len(inventory)}",
                       (x, list_bottom + 2), self.f_small, MUTED)
        if cash_h:
            self._draw_money_card(
                pygame.Rect(x, rect.bottom - 12 - cash_h, cw, cash_h), cash)
        return max_scroll

    LOG_ROW_H = 24  # pixel height of one log line

    def _draw_log(self, y, bottom):
        self._text("Log", (SIDE_X, y), self.f_title, FELT_INK, shadow=True)
        list_top = y + 40
        rows = max(0, (bottom - list_top) // self.LOG_ROW_H)
        # Clamp the stored scroll (lines back from the newest) to the history
        # above the last page, so the wheel handler can never overshoot.
        max_scroll = max(0, len(self.log) - rows)
        self._log_scroll = max(0, min(self._log_scroll, max_scroll))
        if not rows or not self.log:
            return

        end = len(self.log) - self._log_scroll
        start = max(0, end - rows)
        # Leave room for a scrollbar on the right edge when the log overflows.
        text_right = SIDE_X + SIDE_W - (12 if max_scroll else 0)
        y = list_top
        for line, color in self.log[start:end]:
            text = self._truncate(line, self.f_small, text_right - SIDE_X)
            self._text(text, (SIDE_X, y), self.f_small, color, shadow=True)
            y += self.LOG_ROW_H
        if max_scroll:
            self._draw_inventory_scrollbar(list_top, end - start, len(self.log),
                                           start, row_h=self.LOG_ROW_H)

    def _scroll_log(self, dy):
        """Scrolls the game log by ``dy`` wheel notches (up = +, back into
        history); only when no inventory panel has taken over the side area."""
        if self.selected is None:
            self._log_scroll = max(0, self._log_scroll + dy)

    def _scroll_side_panel(self, dy):
        """Routes a wheel notch to whichever side panel is showing: the open
        inventory, or the game log when none is open."""
        if self.selected is not None:
            self._scroll_inventory(dy)
        else:
            self._scroll_log(dy)

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
        self._player_rects = player_rects
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
                if event.type == pygame.MOUSEWHEEL:
                    self._scroll_side_panel(event.y)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, value in buttons:
                        if rect.collidepoint(event.pos):
                            return value
                    for rect, index in player_rects:
                        if rect.collidepoint(event.pos):
                            self.selected = None if self.selected == index else index
                            self._inv_scroll = 0
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
            elif event.type == pygame.MOUSEWHEEL:
                # Scrolling the inventory or log must not skip the animation.
                self._scroll_side_panel(event.y)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if board_rect.collidepoint(event.pos):
                    skip = True
                else:
                    # A click off the board toggles the inventory panel (the
                    # only way to open one during all-AI turns, which never
                    # reach ``ask``); it must not skip the animation.
                    for rect, index in self._player_rects:
                        if rect.collidepoint(event.pos):
                            self.selected = (None if self.selected == index
                                             else index)
                            self._inv_scroll = 0
                            break
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
        if self._auto is not None:
            if player.balance < prop.price:
                return False
            choice = self._auto(f"{player.name}: buy {prop.name}?",
                                [("Buy", "buy"), ("Auction", "decline")])
            return choice == "buy"
        # A Monopoly-style title-deed card over the board with green Buy / red
        # Auction buttons. If the human can't afford the price we don't auto-
        # auction: a "Raise Cash" button opens the manage panel so they can
        # mortgage / sell houses and then buy. Declining sends it to auction.
        while True:
            choice = self._run_overlay_modal(
                lambda mouse: self._draw_purchase_card(player, prop, mouse))
            if choice == "raise":
                self._manage_menu_cards(player)
                continue
            bought = choice == "buy" and player.balance >= prop.price
            self.add_log(
                f"{player.name} {'bought' if bought else 'sent to auction'}"
                f" {prop.name}.")
            return bought

    def _prompt_bid(self, player, prop, min_bid=0):
        """Asks a human for one round of an ascending auction on ``prop``.

        Called each round by ``Game.run_auction``: ``min_bid`` is the smallest
        bid that beats the current standing bid. Shows the property's title-deed
        card with several green bid amounts (the minimum, larger jumps, and an
        all-in) plus a red "Pass"; returns the chosen amount to raise or 0 to
        drop out. The engine keeps calling this as the price climbs until the
        human passes or wins, so tied bids keep escalating.
        """
        if self._auto is not None:
            return 0
        if min_bid <= 0:
            min_bid = max(1, round(prop.price * 0.1))
        # No forced forfeit: even when the human is short on cash they can open
        # the manage panel to mortgage / sell and raise the money to keep
        # bidding, then come back to the auction card.
        while True:
            options = self._bid_options(player, prop, min_bid)
            choice = self._run_overlay_modal(
                lambda mouse: self._draw_auction_card(player, prop, min_bid,
                                                      options, mouse))
            if choice == "raise":
                self._manage_menu_cards(player)
                continue
            bid = int(choice) if isinstance(choice, int) else 0
            if bid > 0:
                self.add_log(f"{player.name} bids ${bid} for {prop.name}.")
            return bid

    def _bid_options(self, player, prop, min_bid):
        """Distinct affordable bid amounts to offer the human this round: the
        minimum raise, a couple of larger jumps, and an all-in. Empty when the
        player can't currently afford the minimum bid (they can still raise
        cash from the auction card)."""
        increment = max(1, round(prop.price * 0.1))
        options = []
        for amt in (min_bid, min_bid + increment, min_bid + 3 * increment):
            if amt <= player.balance and amt not in options:
                options.append(amt)
        if options and player.balance > options[-1]:
            options.append(player.balance)  # all-in
        return options

    def _draw_auction_card(self, player, prop, min_bid, options, mouse):
        self._dim_scene()
        board_cx = BOARD_X + BOARD_PX // 2
        board_cy = BOARD_Y + BOARD_PX // 2
        card_w = 360
        card_h = 500 if isinstance(prop, StreetProperty) else 320
        card = pygame.Rect(board_cx - card_w // 2,
                           board_cy - card_h // 2 - 22, card_w, card_h)
        head = self.f_head.render(f"Auction — {prop.name}", True, FELT_INK)
        self.screen.blit(head, head.get_rect(midbottom=(board_cx, card.y - 30)))
        can_raise = bool(self._mortgageable(player) or self._sellable(player))
        note = (f"{player.name}   ·   next bid ${min_bid}   ·   balance "
                f"${player.balance}")
        if not options:
            note += "   ·   short — raise cash to bid" if can_raise \
                else "   ·   can't cover the next bid"
        sub = self.f_small.render(note, True, FELT_SUB)
        self.screen.blit(sub, sub.get_rect(midbottom=(board_cx, card.y - 8)))
        self._draw_title_deed(card, prop)
        # One button per affordable bid amount, an optional Raise Cash button
        # (mortgage / sell to bid beyond the current balance), then Pass -- all
        # in a single row spanning the board width.
        specs = [(f"All in ${amt}" if amt >= player.balance else f"Bid ${amt}",
                  amt, POS_GREEN) for amt in options]
        if can_raise:
            specs.append(("Raise Cash", "raise", BTN))
        specs.append(("Pass", "pass", NEG_RED))

        buttons = []
        nbtn = len(specs)
        gap = 12
        avail = min(BOARD_PX - 48, nbtn * 160)
        bw = (avail - (nbtn - 1) * gap) // nbtn
        x = board_cx - avail // 2
        by = card.bottom + 22
        for label, value, color in specs:
            r = self._draw_dialog_button(label, x, by, bw, mouse, color)
            buttons.append((r, value))
            x += bw + gap
        return buttons

    def _ai_trade_arbiter(self, initiator, partner, give, receive, cash):
        """Resolves a trade an AI seat proposes to ``partner``.

        ``give`` / ``receive`` are from the initiator's view: it hands over
        ``give`` (plus ``cash``) and receives ``receive``. A human partner is
        asked with a modal; an AI partner judges with its valuation formula.
        Returns True to accept.
        """
        if self._is_ai(partner):
            accepted, _ = self._ai_deciders[partner.name].evaluate_trade(
                partner, initiator, give, receive, cash)
            return accepted
        if self._auto is not None:
            return False
        return self._confirm_trade(initiator, partner, give, receive, cash)

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
        # Routine, high-frequency events the player is already watching -- the
        # GO salary and live auction bids -- are logged but never force a
        # "Continue" click.
        m = message.lower()
        if ("passed go" in m or "auction" in m or " bids $" in m
                or "no one bid" in m):
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
        # Headless/auto games use the plain option list; interactive players get
        # the card-based panel drawn over the board.
        if self._auto is not None:
            return self._manage_menu_text(player, jail_exit)
        return self._manage_menu_cards(player, jail_exit)

    def _manage_menu_text(self, player, jail_exit=False):
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

    # ----- manage-properties card panel ----------------------------------

    def _small_button(self, label, rect, mouse, color):
        """A compact filled button (f_small text) used inside the card grid."""
        hover = rect.collidepoint(mouse) if mouse else False
        shade = tuple(min(255, c + 24) for c in color) if hover else color
        pygame.draw.rect(self.screen, shade, rect, border_radius=6)
        surf = self.f_small.render(label, True, BTN_INK)
        self.screen.blit(surf, surf.get_rect(center=rect.center))
        return rect

    def _manage_menu_cards(self, player, jail_exit=False):
        """Card-grid management panel: build/sell/mortgage per property, plus
        trade and (in jail) the pay/card exits. Returns ``"pay"``/``"card"`` for
        a jail exit or ``None`` when the player is done managing."""
        scroll = 0
        while True:
            mouse = pygame.mouse.get_pos()
            hot, max_scroll = self._draw_manage_panel(
                player, jail_exit, scroll, mouse)
            pygame.display.flip()
            scroll = max(0, min(scroll, max_scroll))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise QuitGame
                if event.type == pygame.MOUSEWHEEL:
                    scroll = max(0, min(scroll - event.y, max_scroll))
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    action = self._handle_manage_click(player, hot, event.pos)
                    if action == "done":
                        return None
                    if action in ("pay", "card"):
                        return action
            self.clock.tick(60)

    def _handle_manage_click(self, player, hot, pos):
        """Applies a clicked manage action. Returns ``"done"``/``"pay"``/
        ``"card"`` for terminal actions, else ``None`` (panel stays open)."""
        for rect, (action, tile) in hot:
            if not rect.collidepoint(pos):
                continue
            if action == "inspect":
                self._manage_detail(player, tile)
            elif action == "trade":
                self._trade_flow(player)
            elif action in ("done", "pay", "card"):
                return action
            else:
                self._apply_manage_action(player, action, tile)
            return None
        return None

    def _apply_manage_action(self, player, action, tile):
        """Carries out a build / sell / mortgage / unmortgage on ``tile`` and
        logs the result. Shared by the card grid and the per-property detail."""
        if action == "build":
            if self.game.build_house(tile, player):
                self.add_log(f"{player.name} built on {tile.name} "
                             f"(now {tile.houses}).")
        elif action == "sell":
            yield_ = tile.house_cost() // 2
            if self.game.sell_house(tile, player):
                self.add_log(f"{player.name} sold a house on {tile.name} "
                             f"for ${yield_}.")
        elif action == "mortgage":
            if self.game.mortgage_property(tile, player):
                self.add_log(f"{player.name} mortgaged {tile.name} for "
                             f"${tile.mortgage_value}.")
        elif action == "unmortgage":
            cost = tile.unmortgage_cost
            if self.game.unmortgage_property(tile, player):
                self.add_log(f"{player.name} lifted the mortgage on "
                             f"{tile.name} for ${cost}.")

    def _manage_detail(self, player, prop):
        """Full-card view of one owned property: the title deed (every rent
        tier + house cost), a live rent/house/sell-yield readout, and the
        build/sell/mortgage buttons. Stays open so the player can build several
        houses and watch the rent climb; Esc or Close returns to the grid."""
        while True:
            mouse = pygame.mouse.get_pos()
            hot = self._draw_manage_detail(player, prop, mouse)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise QuitGame
                if event.type == pygame.KEYDOWN \
                        and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, action in hot:
                        if not rect.collidepoint(event.pos):
                            continue
                        if action == "close":
                            return
                        self._apply_manage_action(player, action, prop)
                        break
            self.clock.tick(60)

    def _manage_detail_line(self, prop):
        """A one-line live readout: current rent, house count, and what the
        next house / a sale would do."""
        if not isinstance(prop, StreetProperty):
            return f"Mortgage value ${prop.mortgage_value}"
        if prop.mortgaged:
            state = "mortgaged — collects no rent"
        elif prop.houses >= 5:
            state = "hotel"
        elif prop.houses:
            state = f"{prop.houses} house" + ("s" if prop.houses > 1 else "")
        else:
            state = "no houses"
        rent = 0 if prop.mortgaged else prop.get_rent(self.game, prop.owner)
        parts = [f"Rent now ${rent}", state]
        if not prop.mortgaged and prop.houses < 5 \
                and prop.has_monopoly(self.game):
            nxt = prop.rent_table[prop.houses + 1]
            parts.append(f"+house → ${nxt} (cost ${prop.house_cost()})")
        if prop.houses > 0:
            parts.append(f"sell → +${prop.house_cost() // 2}")
        return "    ·    ".join(parts)

    def _draw_manage_detail(self, player, prop, mouse):
        self._dim_scene()
        board_cx = BOARD_X + BOARD_PX // 2
        board_cy = BOARD_Y + BOARD_PX // 2
        card_w = 360
        card_h = 500 if isinstance(prop, StreetProperty) else 320
        card = pygame.Rect(board_cx - card_w // 2,
                           board_cy - card_h // 2 - 30, card_w, card_h)
        head = self.f_head.render(prop.name, True, FELT_INK)
        self.screen.blit(head, head.get_rect(midbottom=(board_cx, card.y - 32)))
        sub = self.f_small.render(self._manage_detail_line(prop), True, FELT_SUB)
        self.screen.blit(sub, sub.get_rect(midbottom=(board_cx, card.y - 10)))
        self._draw_title_deed(card, prop)

        # Action buttons (only those legal right now) plus Close, in one row.
        actions = self._manage_card_actions(player, prop)
        actions.append(("Close", "close", BTN))
        buttons = []
        nbtn = len(actions)
        gap = 12
        avail = min(BOARD_PX - 48, nbtn * 170)
        bw = (avail - (nbtn - 1) * gap) // nbtn
        x = board_cx - avail // 2
        by = card.bottom + 20
        for label, action, color in actions:
            r = self._draw_dialog_button(label, x, by, bw, mouse, color)
            buttons.append((r, action))
            x += bw + gap
        return buttons

    def _manage_card_actions(self, player, prop):
        """The (label, action, color) buttons available for ``prop`` right now."""
        actions = []
        if isinstance(prop, StreetProperty):
            if prop.can_build_house(self.game, player):
                actions.append((f"Build  ${prop.house_cost()}", "build",
                                POS_GREEN))
            if prop.can_sell_house(self.game, player):
                actions.append(("Sell house", "sell", (204, 133, 42)))
        if prop.can_mortgage(self.game, player):
            actions.append((f"Mortgage  +${prop.mortgage_value}", "mortgage",
                            BTN))
        if prop.can_unmortgage(self.game, player):
            actions.append((f"Unmortgage  -${prop.unmortgage_cost}",
                            "unmortgage", BTN))
        return actions

    def _draw_manage_panel(self, player, jail_exit, scroll, mouse):
        self._dim_scene()
        dlg = pygame.Rect(BOARD_X + 16, BOARD_Y + 16, BOARD_PX - 32,
                          BOARD_PX - 32)
        self._panel(dlg, PANEL)
        pad = 18
        hot = []

        title = self.f_title.render("Manage Properties", True, INK)
        self.screen.blit(title, title.get_rect(midtop=(dlg.centerx, dlg.y + 12)))
        sub = self.f_small.render(
            f"{player.name}   ·   ${player.balance}   ·   click a card for full "
            f"rents & options", True, MUTED)
        self.screen.blit(sub, sub.get_rect(midtop=(dlg.centerx, dlg.y + 48)))

        content_top = dlg.y + 78
        if jail_exit:
            bx = dlg.x + pad
            if player.balance >= 50:
                r = self._draw_dialog_button("Pay $50 to leave jail", bx,
                                             content_top, 236, mouse, BTN)
                hot.append((r, ("pay", None)))
                bx += 248
            if player.jail_cards:
                r = self._draw_dialog_button("Use Jail Free card", bx,
                                             content_top, 236, mouse, BTN)
                hot.append((r, ("card", None)))
            content_top += 56

        bottom_bar = dlg.bottom - 60
        grid_top = content_top
        grid_bottom = bottom_bar - 12
        props = self._sorted_props(player.properties)

        cols = 3
        cell_pad = 14
        cell_w = (dlg.w - 2 * pad - (cols - 1) * cell_pad) // cols
        cell_h = 142
        rows_fit = max(1, (grid_bottom - grid_top + cell_pad)
                       // (cell_h + cell_pad))
        total_rows = (len(props) + cols - 1) // cols
        max_scroll = max(0, total_rows - rows_fit)

        if not props:
            self._text("You don't own any properties yet.",
                       (dlg.x + pad, grid_top), self.f_body, MUTED)
        else:
            start = scroll * cols
            visible = props[start:start + rows_fit * cols]
            for i, prop in enumerate(visible):
                r, c = divmod(i, cols)
                cx = dlg.x + pad + c * (cell_w + cell_pad)
                cy = grid_top + r * (cell_h + cell_pad)
                card_rect = pygame.Rect(cx, cy, cell_w, 62)
                self._draw_mini_card(card_rect, prop)
                # Clicking the card (not its quick-action buttons) opens the
                # full title-deed detail with every rent tier.
                hot.append((card_rect, ("inspect", prop)))
                by = cy + 68
                for label, act, color in self._manage_card_actions(
                        player, prop)[:2]:
                    r_btn = self._small_button(
                        label, pygame.Rect(cx, by, cell_w, 30), mouse, color)
                    hot.append((r_btn, (act, prop)))
                    by += 36
            if max_scroll:
                hint = f"scroll for more  ·  {scroll + 1}/{max_scroll + 1}"
                self._text(hint, (dlg.x + pad, grid_bottom + 2), self.f_small,
                           MUTED)

        done = self._draw_dialog_button("Done", dlg.right - pad - 150,
                                        bottom_bar, 150, mouse, POS_GREEN)
        hot.append((done, ("done", None)))
        if self._can_trade(player):
            tr = self._draw_dialog_button("Propose Trade",
                                          dlg.right - pad - 150 - 12 - 190,
                                          bottom_bar, 190, mouse, BTN)
            hot.append((tr, ("trade", None)))
        return hot, max_scroll

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
            "scroll_give": 0,
            "scroll_recv": 0,
        }
        while True:
            mouse = pygame.mouse.get_pos()
            hot = self._draw_trade_dialog(player, partners, state, mouse)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise QuitGame
                if event.type == pygame.MOUSEWHEEL:
                    self._scroll_trade_columns(state, pygame.mouse.get_pos(),
                                               event.y)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self._handle_trade_click(player, state, hot, event.pos):
                        return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    self._handle_trade_key(state, event)
            self.clock.tick(60)

    def _scroll_trade_columns(self, state, pos, dy):
        """Wheel-scrolls whichever trade column the mouse is over (clamped in
        the draw pass)."""
        if state.get("_left_region") and state["_left_region"].collidepoint(pos):
            state["scroll_give"] = max(0, state["scroll_give"] - dy)
        elif state.get("_right_region") \
                and state["_right_region"].collidepoint(pos):
            state["scroll_recv"] = max(0, state["scroll_recv"] - dy)

    def _handle_trade_click(self, player, state, hot, pos):
        for rect, partner in hot["partners"]:
            if rect.collidepoint(pos) and state["partner"] is not partner:
                state["partner"] = partner
                state["receive"] = set()
                state["cash_them"] = ""
                state["scroll_recv"] = 0
                return False
        for rect, tile in hot["give"] + hot["receive"]:
            if rect.collidepoint(pos):
                target = state["give"] if (rect, tile) in hot["give"] \
                    else state["receive"]
                # Pop the full title-deed card so the player can read the
                # property's value and rents before adding/removing it.
                selected = tile in target
                choice = self._run_overlay_modal(
                    lambda mouse, _t=tile, _s=selected:
                    self._draw_trade_property_card(_t, _s, mouse))
                if choice == "toggle":
                    target.discard(tile) if selected else target.add(tile)
                return False
        if hot["view_mine"].collidepoint(pos):
            self._browse_inventory(player, state["give"],
                                   f"Your inventory — {player.name}")
            return False
        if hot["view_theirs"].collidepoint(pos):
            partner = state["partner"]
            self._browse_inventory(partner, state["receive"],
                                   f"{partner.name}'s inventory")
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

        if not self._confirm_trade(player, partner, give, receive, cash):
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
        cash_y = dlg.bottom - 116
        # Leave room for the cash-box label, which _draw_cash_box renders 22px
        # above the box top — a smaller gap makes it overlap the view button.
        view_y = cash_y - 82
        col_bottom = view_y - 14
        left_x = dlg.x + pad
        right_x = dlg.x + pad + colw + pad

        # The two columns now show the cards *currently in the trade*; picking
        # cards happens in the per-player inventory browsers below.
        give_tiles = [t for t in player.properties if t in state["give"]]
        recv_tiles = [t for t in partner.properties if t in state["receive"]]
        hot["give"], gmax = self._draw_selected_column(
            give_tiles, left_x, col_top, colw, col_bottom,
            f"You give — {player.name}", state["scroll_give"])
        hot["receive"], rmax = self._draw_selected_column(
            recv_tiles, right_x, col_top, colw, col_bottom,
            f"You receive — {partner.name}", state["scroll_recv"])
        state["scroll_give"] = max(0, min(state["scroll_give"], gmax))
        state["scroll_recv"] = max(0, min(state["scroll_recv"], rmax))
        state["_left_region"] = pygame.Rect(left_x, col_top, colw,
                                            col_bottom - col_top)
        state["_right_region"] = pygame.Rect(right_x, col_top, colw,
                                             col_bottom - col_top)

        hot["view_mine"] = self._draw_dialog_button(
            "View Your Inventory", left_x, view_y, colw, mouse, BTN)
        hot["view_theirs"] = self._draw_dialog_button(
            f"View {partner.name}'s Inventory", right_x, view_y, colw, mouse,
            BTN)

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

    def _draw_selected_column(self, tiles, x, y, w, bottom, header, scroll):
        """A trade column showing the property cards *currently in the trade*
        (``tiles``) as mini-cards; clicking one opens its full card to view or
        remove it. Returns ``(rows, max_scroll)`` of ``(rect, prop)`` hit-boxes.
        """
        self._text(header, (x, y), self.f_head, INK)
        self._text(f"{len(tiles)} card" + ("s" if len(tiles) != 1 else "")
                   + " in trade", (x, y + 28), self.f_small, MUTED)
        list_top = y + 54
        rows = []
        if not tiles:
            self._text("No cards yet — use the", (x, list_top), self.f_small,
                       MUTED)
            self._text("inventory button below.", (x, list_top + 20),
                       self.f_small, MUTED)
            return rows, 0
        tiles = self._sorted_props(tiles)
        row_h = 54
        slots = max(1, (bottom - list_top) // row_h)
        max_scroll = max(0, len(tiles) - slots)
        scroll = max(0, min(scroll, max_scroll))
        for prop in tiles[scroll:scroll + slots]:
            card = pygame.Rect(x, list_top, w, 48)
            self._draw_mini_card(card, prop, selected=True)
            rows.append((card, prop))
            list_top += row_h
        if max_scroll:
            self._text(f"scroll · {len(tiles)} in trade", (x, bottom + 2),
                       self.f_small, MUTED)
        return rows, max_scroll

    def _browse_inventory(self, owner, target_set, title):
        """Full-screen inventory browser for ``owner``: a scrollable grid of
        their property cards. Clicking a card opens its full title deed to add
        or remove it from ``target_set`` (a set of tiles in the trade)."""
        scroll = 0
        while True:
            mouse = pygame.mouse.get_pos()
            hot, max_scroll = self._draw_inventory_browser(
                owner, target_set, title, scroll, mouse)
            pygame.display.flip()
            scroll = max(0, min(scroll, max_scroll))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise QuitGame
                if event.type == pygame.MOUSEWHEEL:
                    scroll = max(0, min(scroll - event.y, max_scroll))
                if event.type == pygame.KEYDOWN \
                        and event.key == pygame.K_ESCAPE:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for rect, item in hot:
                        if not rect.collidepoint(event.pos):
                            continue
                        if item == "__done__":
                            return
                        prop = item
                        can = self.game.can_trade_property(prop)
                        selected = prop in target_set
                        choice = self._run_overlay_modal(
                            lambda m, _p=prop, _s=selected, _c=can:
                            self._draw_trade_property_card(_p, _s, m, _c))
                        if choice == "toggle" and can:
                            (target_set.discard(prop) if selected
                             else target_set.add(prop))
                        break
            self.clock.tick(60)

    def _draw_inventory_browser(self, owner, target_set, title, scroll, mouse):
        self._dim_scene()
        dlg = pygame.Rect(BOARD_X + 16, BOARD_Y + 16, BOARD_PX - 32,
                          BOARD_PX - 32)
        self._panel(dlg, PANEL)
        pad = 18
        hot = []
        t = self.f_title.render(title, True, INK)
        self.screen.blit(t, t.get_rect(midtop=(dlg.centerx, dlg.y + 12)))
        sub = self.f_small.render(
            "click a card to view its rents and add / remove it from the trade",
            True, MUTED)
        self.screen.blit(sub, sub.get_rect(midtop=(dlg.centerx, dlg.y + 48)))

        bottom_bar = dlg.bottom - 60
        grid_top = dlg.y + 78
        grid_bottom = bottom_bar - 12
        props = self._sorted_props(owner.properties)
        cols = 3
        cell_pad = 14
        cell_w = (dlg.w - 2 * pad - (cols - 1) * cell_pad) // cols
        cell_h = 66
        rows_fit = max(1, (grid_bottom - grid_top + cell_pad)
                       // (cell_h + cell_pad))
        total_rows = (len(props) + cols - 1) // cols
        max_scroll = max(0, total_rows - rows_fit)

        if not props:
            self._text("No properties owned.", (dlg.x + pad, grid_top),
                       self.f_body, MUTED)
        else:
            start = scroll * cols
            for i, prop in enumerate(props[start:start + rows_fit * cols]):
                r, c = divmod(i, cols)
                cx = dlg.x + pad + c * (cell_w + cell_pad)
                cy = grid_top + r * (cell_h + cell_pad)
                card = pygame.Rect(cx, cy, cell_w, cell_h)
                self._draw_mini_card(card, prop, selected=prop in target_set)
                hot.append((card, prop))
            if max_scroll:
                hint = f"scroll for more  ·  {scroll + 1}/{max_scroll + 1}"
                self._text(hint, (dlg.x + pad, grid_bottom + 2), self.f_small,
                           MUTED)

        done = self._draw_dialog_button("Done", dlg.right - pad - 150,
                                        bottom_bar, 150, mouse, POS_GREEN)
        hot.append((done, "__done__"))
        return hot, max_scroll

    def _draw_trade_property_card(self, prop, selected, mouse, can_toggle=True):
        """Full title-deed card popup for a property in the trade builder, with
        an add/remove toggle (green/red) and a Close button. When
        ``can_toggle`` is False the property can't be traded (it has houses in
        its group), so only a Close button and a note are shown."""
        self._dim_scene()
        board_cx = BOARD_X + BOARD_PX // 2
        board_cy = BOARD_Y + BOARD_PX // 2
        card_w = 360
        card_h = 500 if isinstance(prop, StreetProperty) else 320
        card = pygame.Rect(board_cx - card_w // 2,
                           board_cy - card_h // 2 - 22, card_w, card_h)
        self._draw_title_deed(card, prop)
        bw, gap = 190, 16
        by = card.bottom + 22
        if not can_toggle:
            note = self.f_small.render(
                "Sell the houses in this color group before trading it.", True,
                FELT_SUB)
            self.screen.blit(note, note.get_rect(midtop=(board_cx, by)))
            close = self._draw_dialog_button("Close", board_cx - bw // 2, by + 24,
                                             bw, mouse, BTN)
            return [(close, "close")]
        if selected:
            act = self._draw_dialog_button("Remove from Trade",
                                           board_cx - bw - gap // 2, by, bw,
                                           mouse, NEG_RED)
        else:
            act = self._draw_dialog_button("Add to Trade",
                                           board_cx - bw - gap // 2, by, bw,
                                           mouse, POS_GREEN)
        close = self._draw_dialog_button("Close", board_cx + gap // 2, by, bw,
                                         mouse, BTN)
        return [(act, "toggle"), (close, "close")]

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
        """Announces the outcome and lets the player start another game or quit.
        Returns ``"restart"`` or ``"quit"``."""
        winner = self.game.winner()
        message = f"{winner.name} wins!" if winner else "Game over."
        self.add_log(message)
        return self.ask(message, [("Play Again", "restart"), ("Quit", "quit")])

    # ----- main loop -----------------------------------------------------

    def run(self, max_turns=10000):
        """Plays one game to its end. Returns ``"restart"`` if the player chose
        Play Again on the result screen, otherwise ``"quit"`` (including closing
        the window mid-game). Leaves pygame running so the caller can start a new
        game on the same window."""
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
                turns += 1
            return self._show_result()
        except QuitGame:
            return "quit"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _new_match(config, model, ai_names, deterministic, seed):
    """Seeds every RNG for one match and builds a fresh ``Game`` plus a
    ``GUIAIDecider`` for each AI seat (all sharing ``model``).

    Seeds dice and card shuffles (stdlib ``random``), the turn-order shuffle,
    and the AI's policy sampling (torch), then prints the seed so the game can
    be replayed with ``--seed``.
    """
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    print(f"Game seed: {seed}  (replay with --seed {seed})")

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

    ai_deciders = {}
    if model is not None:
        from ui.ai_player import GUIAIDecider
        for name in ai_names:
            ai_deciders[name] = GUIAIDecider(
                num_players=4, model=model, deterministic=deterministic)
    return game, ai_deciders


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

    # Play matches back-to-back: "Play Again" on the result screen loops here
    # for a fresh game on the same window, without re-running setup or reloading
    # the model. ``--seed`` fixes the *first* game (for reproducibility); every
    # restart draws a fresh game.
    #
    # A fresh seed is drawn from ``SystemRandom`` (OS entropy), NOT ``random``,
    # because loading the MaskablePPO model reseeds the global ``random`` module
    # to its training seed -- so ``random.randrange`` here would return the same
    # value on every launch and every AI game would play out identically.
    next_seed = args.seed
    while True:
        seed = (next_seed if next_seed is not None
                else random.SystemRandom().randrange(2 ** 31))
        next_seed = None  # any restart is a fresh, differently-seeded game
        game, ai_deciders = _new_match(
            config, model, ai_names, args.deterministic, seed)
        result = MonopolyApp(game, ai_deciders=ai_deciders, screen=screen).run()
        if result != "restart":
            break

    pygame.quit()


if __name__ == "__main__":
    main()
