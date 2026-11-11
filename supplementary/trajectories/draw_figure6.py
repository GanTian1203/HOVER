"""Draw the 85 mm case figure exclusively from verified DEV trace CSV rows."""
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

from single_style import BLUE, CORAL, GRID, INK, MUTED, ZERO, axis, figure, text_at

HEIGHT = 95
LABELS = {"ECL": "OLinear", "PEMS03": "NLinear"}
SERIES = (
    ("y_true", "Observed", INK, "-", 1.0),
    ("y_base", "Frozen", BLUE, (0, (3.5, 1.7)), .96),
    ("y_hover", "HOVER", CORAL, "-", 1.05),
)


def format_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linewidth=.44, color=GRID)
    ax.tick_params(pad=1.45)
    ax.yaxis.set_major_locator(MaxNLocator(3))


def draw(evidence):
    cases, rows = evidence
    fig = figure(HEIGHT)
    keys = [Line2D([], [], label=label, color=color, ls=ls, lw=width)
            for _, label, color, ls, width in SERIES]
    fig.legend(handles=keys, loc="center", bbox_to_anchor=(.53, 91.5 / HEIGHT),
               ncol=3, frameon=False, fontsize=7.7, handlelength=1.85,
               handletextpad=.38, columnspacing=1.3)
    for i, case in enumerate(cases):
        name = case["dataset"]
        points = [r for r in rows if r["dataset"] == name]
        offsets = np.array([int(r["relative_step"]) for r in points])
        np.testing.assert_array_equal(offsets, np.arange(-336, 96))
        assert len({(r["seed"], r["origin"], r["channel_0_based"]) for r in points}) == 1
        past = np.array([float(r["history"]) for r in points if int(r["relative_step"]) < 0])
        observed, frozen, refined = [
            np.array([float(r[key]) for r in points if int(r["relative_step"]) >= 0])
            for key, *_ in SERIES
        ]
        x = np.arange(96)
        j, k = case["zoom_steps"]
        bottom = 51 - 39.1 * i if len(cases) == 2 else 18
        height = 29 if len(cases) == 2 else 55
        main = axis(fig, 13.1, bottom, 38.9, height)
        zoom = axis(fig, 58.1, bottom, 23.7, height)
        main.axvspan(j, k, color="#FFEBB2", alpha=.47, linewidth=0)
        main.axvline(0, color=ZERO, linewidth=.55, linestyle=(0, (2, 2)))
        main.plot(np.arange(-48, 96), np.r_[past[-48:], observed], color=INK, lw=1.0, zorder=2)
        main.set_xlim(-48, 95)
        main.set_xticks([-48, 0, 48, 95])
        for y, (_, _, color, ls, width) in zip((observed, frozen, refined), SERIES):
            if color != INK:
                main.plot(x, y, color=color, ls=ls, lw=width)
            zoom.plot(x[j:k + 1], y[j:k + 1], color=color, ls=ls, lw=width)
        zoom.set_xlim(j, k)
        zoom.set_xticks([j, (j + k) // 2, k])
        for ax in (main, zoom):
            format_axis(ax)
        text_at(fig, 13.1, bottom + height + 4.35, f"({chr(97+i)}) {name} · {LABELS[name]}",
                ha="left", fontsize=8.2, fontweight="bold")
        text_at(fig, 69.9, bottom + height + 4.35, "Zoom", color=MUTED, fontsize=7.6)
    text_at(fig, 2.9, 44, "TRAIN-standardized value", rotation=90, fontsize=7.5)
    text_at(fig, 47.5, 3.7, "Step relative to forecast origin", fontsize=7.7)
    return fig
