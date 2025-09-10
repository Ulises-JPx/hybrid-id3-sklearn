
from typing import Tuple
import matplotlib.pyplot as plt

_DEFAULTS = {
    "max_dim_px": 64000,
    "font_size": 12,
    "dpi": 200,
    "padding_px": 24,
}

def _estimate_tree_dimensions(tree_text: str, font_size: int, dpi: int, padding_px: int) -> Tuple[int, int]:
    lines = tree_text.splitlines() if tree_text else [""]
    pixels_per_point = dpi / 72.0
    char_width_px = font_size * 0.6 * pixels_per_point
    line_height_px = max(1.0, font_size * 1.4 * pixels_per_point)
    max_characters = max((len(line) for line in lines), default=1)
    width_px = int(padding_px * 2 + max(1, max_characters) * char_width_px)
    height_px = int(padding_px * 2 + max(1, len(lines)) * line_height_px)
    return width_px, height_px

def save_tree_png_safe(tree_text: str, filename: str, max_dim_px: int = None, font_size: int = None, dpi: int = None, padding_px: int = None):
    max_dim_px = _DEFAULTS["max_dim_px"] if max_dim_px is None else max_dim_px
    font_size = _DEFAULTS["font_size"] if font_size is None else font_size
    dpi = _DEFAULTS["dpi"] if dpi is None else dpi
    padding_px = _DEFAULTS["padding_px"] if padding_px is None else padding_px

    width_px, height_px = _estimate_tree_dimensions(tree_text, font_size, dpi, padding_px)
    if width_px > max_dim_px or height_px > max_dim_px:
        print("Cannot generate tree PNG: dimensions exceed the allowed maximum.")
        return

    try:
        figure_width_in = max(2.0, width_px / dpi)
        figure_height_in = max(2.0, height_px / dpi)

        fig = plt.figure(figsize=(figure_width_in, figure_height_in), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        ax.text(0.01, 0.99, tree_text, fontfamily="monospace", fontsize=font_size, va="top", ha="left")

        plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    except Exception:
        print("Cannot generate tree PNG: an error occurred during rendering or saving.")
