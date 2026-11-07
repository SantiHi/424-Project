"""Charts for the initial analysis notebook. Each function draws one figure from summary tables.

Colour is used sparingly: one accent hue for the data, a second hue only where two series must
be told apart, and neutral greys for context. The blue/orange pair was checked for colour-blind
separation. Text is never drawn in a data colour.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .detect import KINDS
from .prepare import POSITION_GROUPS

SURFACE = "#fcfcfb"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CONTEXT = "#c3c2b7"  # marks that give context rather than carry the point
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

DRILL_LABELS = {"forty": "40-yard dash", "bench": "Bench press", "vertical": "Vertical jump",
                "broad_jump": "Broad jump", "cone": "3-cone drill", "shuttle": "20-yard shuttle"}


def use_style() -> None:
    """Apply the shared look: quiet axes and grid, so the data is the only loud thing."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9.5, "text.color": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 9.5, "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK_MUTED, "ytick.labelcolor": INK_MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK_SECONDARY,
    })


def _percent_axis(axis) -> None:
    axis.set_major_formatter(lambda value, _: f"{value:.0f}%")


def coverage_by_class(per_class: pd.DataFrame, missing_forty: pd.DataFrame) -> None:
    """Left: prospects per combine class, split by draft outcome. Right: share with no 40 time."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.5), constrained_layout=True)
    years = per_class.index

    left.bar(years, per_class["drafted"], width=0.74, color=BLUE, edgecolor=SURFACE, linewidth=1.2, label="Drafted")
    left.bar(years, per_class["undrafted"], bottom=per_class["drafted"], width=0.74, color=CONTEXT,
             edgecolor=SURFACE, linewidth=1.2, label="Not drafted")
    left.annotate("2021: no combine was held,\nso these are pro-day results", xy=(2021, per_class.loc[2021, "prospects"]),
                  xytext=(2008.5, per_class["prospects"].max() * 1.02), fontsize=8.5, color=INK_SECONDARY, va="top",
                  arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8})
    left.set_title("Position players at each combine")
    left.set_ylabel("Prospects")
    left.legend(loc="upper left", ncol=2, handlelength=1.0, columnspacing=1.2, borderaxespad=0.2)
    left.set_ylim(0, per_class["prospects"].max() * 1.22)

    share = missing_forty["share_no_forty"] * 100
    right.bar(years, share.reindex(years), width=0.74, color=BLUE)
    latest = years.max()
    right.text(latest, share[latest] + 1.2, f"{share[latest]:.0f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
    right.set_title("Share with no 40-yard dash on record")
    _percent_axis(right.yaxis)
    right.set_ylim(0, max(share.max() * 1.18, 10))

    for ax in (left, right):
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("Combine class")
        ax.set_xticks(range(2000, int(latest) + 1, 5))
        ax.set_xlim(years.min() - 0.8, latest + 0.8)


def drill_coverage(coverage: pd.DataFrame) -> None:
    """Heatmap: share of prospects with each drill on record, by position group."""
    drills = [name for name in DRILL_LABELS if name in coverage.columns and name != "forty"]
    values = coverage[drills] * 100
    fig, ax = plt.subplots(figsize=(6.6, 4.6), constrained_layout=True)
    ax.imshow(values.to_numpy(), cmap=BLUE_RAMP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(drills)), [DRILL_LABELS[name] for name in drills])
    ax.set_yticks(range(len(values)), values.index)
    ax.tick_params(length=0, labelcolor=INK_SECONDARY)
    ax.xaxis.tick_top()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # white rules between cells keep neighbouring shades apart
    ax.set_xticks(np.arange(-0.5, len(drills)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(values)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for row, group in enumerate(values.index):
        for col, drill in enumerate(drills):
            value = values.loc[group, drill]
            ax.text(col, row, f"{value:.0f}%", ha="center", va="center", fontsize=9,
                    color="#ffffff" if value >= 55 else INK)
    ax.set_title("Share of prospects with each drill on record", pad=28)


def size_and_speed(prospects: pd.DataFrame, correlation: pd.DataFrame) -> None:
    """Weight against 40 time for every sampled prospect, with each position's centre labelled."""
    sample = prospects[prospects["in_sample"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.scatter(sample["wt"], sample["forty"], s=5, color=BLUE, alpha=0.16, linewidths=0, rasterized=True)

    centres = sample.groupby("grp")[["wt", "forty"]].median().reindex(POSITION_GROUPS)
    ax.scatter(centres["wt"], centres["forty"], s=46, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    # label offsets (points) chosen so neighbouring positions do not collide
    offsets = {"CB": (-9, 7), "WR": (0, 11), "S": (-3, -13), "RB": (10, -3), "QB": (0, -14), "LB": (-3, 11),
               "TE": (-2, -13), "EDGE": (10, 3), "DT": (-12, 8), "OT": (12, -1), "IOL": (2, -14)}
    for group, centre in centres.iterrows():
        dx, dy = offsets.get(group, (8, 8))
        ax.annotate(group, (centre["wt"], centre["forty"]), xytext=(dx, dy), textcoords="offset points",
                    ha="center" if dx == 0 else ("left" if dx > 0 else "right"), va="center",
                    fontsize=9, fontweight="bold", color=INK)

    overall = correlation.loc["All positions together", "corr_weight_forty"]
    within = correlation.drop(index="All positions together")["corr_weight_forty"]
    ax.text(0.98, 0.04, f"Correlation of weight and 40 time\nall positions together: {overall:.2f}\n"
                        f"inside one position: {within.min():.2f} to {within.max():.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK_SECONDARY, linespacing=1.4)
    ax.invert_yaxis()
    ax.set_xlabel("Weight (lb)")
    ax.set_ylabel("40-yard dash (seconds; faster is up)")
    ax.set_title(f"Weight and 40 time for {len(sample):,} prospects; dots mark each position's median")


def spread_by_position(prospects: pd.DataFrame) -> None:
    """Box plots of height, weight and 40 time inside each position group."""
    sample = prospects[prospects["in_sample"]]
    panels = [("ht", "Height (in)"), ("wt", "Weight (lb)"), ("forty", "40-yard dash (s)")]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 6.6), sharex=True, constrained_layout=True)
    for ax, (column, label) in zip(axes, panels):
        data = [sample.loc[sample["grp"] == group, column] for group in POSITION_GROUPS]
        ax.boxplot(
            data, widths=0.5, patch_artist=True,
            boxprops={"facecolor": "#cde2fb", "edgecolor": BLUE, "linewidth": 0.9},
            medianprops={"color": "#184f95", "linewidth": 1.8},
            whiskerprops={"color": INK_MUTED, "linewidth": 0.9}, capprops={"color": INK_MUTED, "linewidth": 0.9},
            flierprops={"marker": "o", "markersize": 2.2, "markerfacecolor": INK_MUTED, "markeredgewidth": 0, "alpha": 0.45})
        ax.set_xticks(range(1, len(POSITION_GROUPS) + 1), POSITION_GROUPS)
        ax.tick_params(axis="x", length=0, labelcolor=INK_SECONDARY)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(label)
    axes[0].set_title("Spread inside each position group (box = middle half, dots = beyond 1.5 box-lengths)")


def testing_by_class(trend: pd.DataFrame) -> None:
    """Line chart: how each class tested against the position's all-years average."""
    fig, ax = plt.subplots(figsize=(7.6, 3.7), constrained_layout=True)
    colours = [BLUE, ORANGE]
    ax.axhline(0, color=AXIS, linewidth=1.0)
    for drill, colour in zip(trend.columns, colours):
        series = trend[drill].dropna()
        ax.plot(series.index, series, color=colour, label=DRILL_LABELS[drill])
        ax.scatter(series.index[-1], series.iloc[-1], s=42, color=colour, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(DRILL_LABELS[drill], (series.index[-1], series.iloc[-1]), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xlim(trend.index.min() - 0.5, trend.index.max() + 5.2)
    ax.set_xticks(range(2000, int(trend.index.max()) + 1, 5))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Combine class")
    ax.set_ylabel("Class average, in standard deviations\nfrom the position's all-years mean")
    ax.set_title("Recent classes test better (up = faster time, longer jump)")
    ax.legend(loc="upper left", ncol=2, handlelength=1.4, borderaxespad=0.2)


def drafted_by_speed(by_speed: pd.DataFrame) -> None:
    """Two bar charts: share drafted, and median pick, by 40-time fifth within position."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)
    positions = np.arange(len(by_speed))
    labels = [label.replace(" (", "\n(") for label in by_speed.index]

    share = by_speed["share_drafted"] * 100
    left.bar(positions, share, width=0.34, color=BLUE)
    for x, value in zip(positions, share):
        left.text(x, value + 1.5, f"{value:.0f}%", ha="center", va="bottom", fontsize=9, color=INK)
    left.set_ylim(0, 100)
    _percent_axis(left.yaxis)
    left.set_title("Share drafted")

    pick = by_speed["median_pick_if_drafted"]
    right.bar(positions, pick, width=0.34, color=BLUE)
    for x, value in zip(positions, pick):
        right.text(x, value + 3, f"{value:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    right.set_ylim(0, pick.max() * 1.18)
    right.set_title("Median overall pick, among those drafted")

    for ax in (left, right):
        ax.set_xticks(positions, labels)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("40-yard dash, fifths within position group")


def career_value(picks: pd.DataFrame, by_pick: pd.DataFrame, by_class: pd.DataFrame, last_class: int = 2015) -> None:
    """Left: career value against overall pick for finished classes. Right: average by draft class."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.7), constrained_layout=True,
                                      gridspec_kw={"width_ratios": [1.25, 1]})
    finished = picks[picks["season"] <= last_class]
    left.scatter(finished["pick"], finished["w_av"], s=5, color=CONTEXT, alpha=0.55, linewidths=0, rasterized=True)
    left.plot(by_pick.index, by_pick["smoothed_mean"], color=BLUE, linewidth=2.0)
    label_at = 40
    left.annotate("average for the slot", (label_at, by_pick.loc[label_at, "smoothed_mean"]), xytext=(58, 74),
                  fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.8})
    left.set_xlim(0, 262)
    left.set_ylim(0, finished["w_av"].max() * 1.04)
    left.set_xlabel("Overall pick")
    left.set_ylabel("Career value (weighted AV)")
    left.set_title(f"Career value by pick, 2000-{last_class} draft classes")

    classes = by_class.index
    right.bar(classes, by_class["mean_career_value"], width=0.74, color=BLUE)
    right.annotate("careers still\nin progress", xy=(classes.max() - 2.5, by_class["mean_career_value"].iloc[-6] + 1.5),
                   ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    right.set_xticks(range(2000, int(classes.max()) + 1, 5))
    right.set_xlim(classes.min() - 0.8, classes.max() + 0.8)
    right.grid(axis="x", visible=False)
    right.set_xlabel("Draft class")
    right.set_title("Average career value per pick, by draft class")


def unusual_examples(scored: pd.DataFrame, groups: tuple[str, ...] = ("WR", "EDGE", "CB")) -> None:
    """Weight against 40 time for a few position groups, with the flagged prospects highlighted."""
    fig, axes = plt.subplots(1, len(groups), figsize=(10.8, 3.9), constrained_layout=True)
    fast, other_unusual = KINDS[0], [KINDS[1], KINDS[2]]
    for ax, group in zip(axes, groups):
        rows = scored[scored["grp"] == group]
        typical = rows[~rows["unusual"]]
        ax.scatter(typical["wt"], typical["forty"], s=6, color=CONTEXT, alpha=0.6, linewidths=0, rasterized=True)
        others = rows[rows["kind"].isin(other_unusual)]
        ax.scatter(others["wt"], others["forty"], s=34, color=ORANGE, edgecolor=SURFACE, linewidth=1.0,
                   label="Unusual, not fast for size", zorder=3)
        standouts = rows[rows["kind"] == fast]
        ax.scatter(standouts["wt"], standouts["forty"], s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.0,
                   label="Unusual, fast for size", zorder=4)
        ax.invert_yaxis()
        ax.set_title(group)
        ax.set_xlabel("Weight (lb)")
    axes[0].set_ylabel("40-yard dash (seconds; faster is up)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="outside lower center", ncol=2, handletextpad=0.3, columnspacing=2.0)


def fast_flags_by_era(by_era: pd.DataFrame) -> None:
    """Grouped bars: share flagged as unusual and fast, by period, under two comparisons."""
    fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    positions = np.arange(len(by_era))
    series = [("compared_with_own_era", "Compared with his own era", BLUE, -0.075),
              ("compared_with_all_classes", "Compared with all classes", ORANGE, 0.075)]
    for column, label, colour, offset in series:
        share = by_era[column] * 100
        ax.bar(positions + offset, share, width=0.11, color=colour, label=label)
        for x, value in zip(positions + offset, share):
            ax.text(x, value + 0.06, f"{value:.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(positions, by_era.index)
    ax.set_xlim(-0.5, len(by_era) - 0.5)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, np.ceil(by_era.to_numpy().max() * 100 * 1.25))
    ax.set_yticks(range(0, int(ax.get_ylim()[1]) + 1))
    _percent_axis(ax.yaxis)
    ax.set_xlabel("Combine classes")
    ax.set_title("Share flagged as unusual and fast for size")
    ax.legend(loc="upper left", ncol=1, handlelength=1.0, borderaxespad=0.2)
