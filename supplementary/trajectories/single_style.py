"""85 mm figure primitives using the accepted HOVER results palette."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

WIDTH = 85
INK = "#1C2A34"
MUTED = "#526574"
GRID = "#E7EDF4"
ZERO = "#788896"
BLUE = "#5C85C2"
BLUE_TEXT = "#436A9F"
PALE_BLUE = "#C7D9EE"
CORAL = "#D96769"
CORAL_TEXT = "#A74751"
PINK = "#F2BCC2"
PEACH = "#FBE5D6"
GOLD = "#FFEBB2"


def style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = next(name for name in ("Times New Roman", "Liberation Serif", "DejaVu Serif") if name in available)
    plt.rcParams.update({
        "font.family": family,
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "axes.linewidth": .55,
        "axes.edgecolor": ZERO,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.1,
        "xtick.major.width": .48,
        "ytick.major.width": .48,
        "xtick.major.size": 1.9,
        "ytick.major.size": 1.9,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
        "svg.hashsalt": "hover-single-column",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def figure(height):
    style()
    return plt.figure(figsize=(WIDTH / 25.4, height / 25.4), dpi=180)


def axis(fig, x, y, w, h):
    height = fig.get_figheight() * 25.4
    return fig.add_axes([x / WIDTH, y / height, w / WIDTH, h / height])


def text_at(fig, x, y, label, **kwargs):
    height = fig.get_figheight() * 25.4
    return fig.text(x / WIDTH, y / height, label, va="center",
                    ha=kwargs.pop("ha", "center"), **kwargs)


def title(fig, y, label):
    text_at(fig, 2.5, y, label, ha="left", fontweight="bold", fontsize=8.5)


def signed(value, decimals=2):
    return f"{value:+.{decimals}f}".replace("-", "−")


def export(fig, name, output=ROOT):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{ext}", dpi=450)
    plt.close(fig)
