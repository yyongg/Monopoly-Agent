"""Maps board positions (0-39) to pixel coordinates on the board image.

The standard board is an 11x11 grid of tiles. Position 0 (GO) sits in the
bottom-right corner and positions increase counter-clockwise: left along the
bottom edge to Jail (10, bottom-left), up the left edge to Free Parking
(20, top-left), right along the top edge to Go To Jail (30, top-right), then
down the right edge back toward GO.
"""

# The board is 11 tiles per side.
GRID = 11


def tile_grid(pos):
    """
    Returns the (row, col) cell of a board position in the 11x11 grid, with
    row 0 at the top and col 0 at the left.

    Args:
        pos (int): Board position 0-39.
    """
    pos %= 40
    if pos <= 10:
        # Bottom edge: GO at (10, 10) leftward to Jail at (10, 0).
        return (10, 10 - pos)
    if pos <= 20:
        # Left edge: upward from (9, 0) to Free Parking at (0, 0).
        return (10 - (pos - 10), 0)
    if pos <= 30:
        # Top edge: rightward from (0, 1) to Go To Jail at (0, 10).
        return (0, pos - 20)
    # Right edge: downward from (1, 10) to (9, 10).
    return (pos - 30, 10)


def tile_center(pos, board_x, board_y, board_px):
    """
    Returns the (x, y) pixel center of a board position's tile.

    Args:
        pos (int): Board position 0-39.
        board_x (int): Left pixel of the board image on screen.
        board_y (int): Top pixel of the board image on screen.
        board_px (int): On-screen pixel size of the (square) board.
    """
    row, col = tile_grid(pos)
    cell = board_px / GRID
    cx = board_x + (col + 0.5) * cell
    cy = board_y + (row + 0.5) * cell
    return (int(cx), int(cy))


def interior_offset(pos):
    """
    Returns a unit (dx, dy) pointing from a perimeter tile toward the board's
    interior. Used to place house markers just inside the edge a tile sits on.
    Streets never occupy the four corners, so only the four edges are handled.

    Args:
        pos (int): Board position 0-39.
    """
    pos %= 40
    if pos < 10:
        return (0, -1)   # bottom edge -> up
    if pos < 20:
        return (1, 0)    # left edge -> right
    if pos < 30:
        return (0, 1)    # top edge -> down
    return (-1, 0)       # right edge -> left


def token_offset(index, board_px):
    """
    Returns a small (dx, dy) so up to four tokens on one tile don't fully
    overlap, arranged in a 2x2 cluster by player index.

    Args:
        index (int): Player index (0-3).
        board_px (int): On-screen pixel size of the board (for scaling).
    """
    step = board_px / GRID * 0.2
    corners = [(-step, -step), (step, -step), (-step, step), (step, step)]
    dx, dy = corners[index % len(corners)]
    return (int(dx), int(dy))
