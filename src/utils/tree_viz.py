"""
@author Ulises Jaramillo Portilla | A01798380 | Ulises-JPx

This module provides utility functions for visualizing decision trees as PNG images.
It estimates the required image dimensions based on the tree's textual representation,
and safely saves the visualization to a PNG file using matplotlib.
"""

from typing import Tuple
import matplotlib.pyplot as plt

# Default configuration values for image generation
_DEFAULTS = {
    "max_dim_px": 64000,   # Maximum allowed dimension (width or height) in pixels
    "font_size": 12,       # Default font size for tree text
    "dpi": 200,            # Default dots per inch for image resolution
    "padding_px": 24,      # Default padding around the tree text in pixels
}

def _estimate_tree_dimensions(
    tree_text: str,
    font_size: int,
    dpi: int,
    padding_px: int
) -> Tuple[int, int]:
    """
    Estimate the required width and height in pixels to render the tree text.

    Parameters:
        tree_text (str): The textual representation of the decision tree.
        font_size (int): Font size in points for rendering the text.
        dpi (int): Dots per inch for image resolution.
        padding_px (int): Padding in pixels around the tree text.

    Returns:
        Tuple[int, int]: Estimated (width_px, height_px) for the image.
    """
    # Split the tree text into lines for dimension calculations
    lines = tree_text.splitlines() if tree_text else [""]
    # Calculate the number of pixels per point based on DPI
    pixels_per_point = dpi / 72.0
    # Estimate the width of a single character in pixels
    char_width_px = font_size * 0.6 * pixels_per_point
    # Estimate the height of a single line in pixels
    line_height_px = max(1.0, font_size * 1.4 * pixels_per_point)
    # Find the maximum number of characters in any line for width calculation
    max_characters = max((len(line) for line in lines), default=1)
    # Calculate total width including padding
    width_px = int(padding_px * 2 + max(1, max_characters) * char_width_px)
    # Calculate total height including padding
    height_px = int(padding_px * 2 + max(1, len(lines)) * line_height_px)
    return width_px, height_px

def save_tree_png_safe(
    tree_text: str,
    filename: str,
    max_dim_px: int = None,
    font_size: int = None,
    dpi: int = None,
    padding_px: int = None
):
    """
    Safely save a decision tree's textual representation as a PNG image.

    This function estimates the required image size, checks if it exceeds the
    allowed maximum dimension, and renders the tree text using matplotlib.
    If the image is too large or an error occurs, it prints an informative
    message and does not save the file.

    Parameters:
        tree_text (str): The textual representation of the decision tree.
        filename (str): Path to save the PNG image.
        max_dim_px (int, optional): Maximum allowed dimension (width or height) in pixels.
                                    Defaults to _DEFAULTS["max_dim_px"] if None.
        font_size (int, optional): Font size in points for rendering the text.
                                   Defaults to _DEFAULTS["font_size"] if None.
        dpi (int, optional): Dots per inch for image resolution.
                             Defaults to _DEFAULTS["dpi"] if None.
        padding_px (int, optional): Padding in pixels around the tree text.
                                    Defaults to _DEFAULTS["padding_px"] if None.
    """
    # Use default values if parameters are not provided
    max_dim_px = _DEFAULTS["max_dim_px"] if max_dim_px is None else max_dim_px
    font_size = _DEFAULTS["font_size"] if font_size is None else font_size
    dpi = _DEFAULTS["dpi"] if dpi is None else dpi
    padding_px = _DEFAULTS["padding_px"] if padding_px is None else padding_px

    # Estimate the required image dimensions for the tree text
    width_px, height_px = _estimate_tree_dimensions(tree_text, font_size, dpi, padding_px)

    # Check if the estimated dimensions exceed the allowed maximum
    if width_px > max_dim_px or height_px > max_dim_px:
        print("Cannot generate tree PNG: dimensions exceed the allowed maximum.")
        return

    try:
        # Calculate figure size in inches for matplotlib
        figure_width_in = max(2.0, width_px / dpi)
        figure_height_in = max(2.0, height_px / dpi)

        # Create a new matplotlib figure with the calculated size and DPI
        fig = plt.figure(figsize=(figure_width_in, figure_height_in), dpi=dpi)
        # Add axes that fill the entire figure and remove axis lines
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        # Render the tree text in monospace font, aligned to top-left
        ax.text(
            0.01, 0.99, tree_text,
            fontfamily="monospace",
            fontsize=font_size,
            va="top",
            ha="left"
        )

        # Save the figure as a PNG file with tight bounding box and minimal padding
        plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)
        # Close the figure to free resources
        plt.close(fig)
    except Exception:
        # Handle any errors during rendering or saving
        print("Cannot generate tree PNG: an error occurred during rendering or saving.")
