"""Descriptive tables for the initial analysis notebook. Every function returns a DataFrame.

`prospects` is the table from prepare.build_prospects(); `picks` is prepare.clean_draft_picks().
Unless a function says otherwise it uses only the rows marked `in_sample`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .detect import DETECTORS, KINDS
from .prepare import DRILLS, HIGHER_IS_BETTER, POSITION_GROUPS

MEASUREMENTS = ["ht", "wt"] + DRILLS
LABELS = {
    "ht": "Height (in)", "wt": "Weight (lb)", "forty": "40-yard dash (s)", "bench": "Bench press (reps)",
    "vertical": "Vertical jump (in)", "broad_jump": "Broad jump (in)", "cone": "3-cone drill (s)",
    "shuttle": "20-yard shuttle (s)",
}
ROUND_LABELS = ["Round 1", "Round 2", "Round 3", "Round 4", "Round 5", "Round 6", "Round 7", "Undrafted"]


def raw_overview(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Size and coverage of the three raw files."""
    combine, draft, players = raw["combine"], raw["draft_picks"], raw["players"]
    rows = [
        ("combine.csv", "a prospect at one combine", combine, f"{combine.season.min()}-{combine.season.max()}"),
        ("draft_picks.csv", "a draft pick", draft, f"{draft.season.min()}-{draft.season.max()}"),
        ("players.csv", "a player on an NFL roster", players, "all years"),
    ]
    return pd.DataFrame(
        [{"file": name, "one row is": unit, "rows": len(table), "columns": table.shape[1], "years": years}
         for name, unit, table, years in rows]).set_index("file")


def prospects_per_class(prospects: pd.DataFrame) -> pd.DataFrame:
    """Position players per combine class and how many were drafted (all rows, not only the sample)."""
    per_class = prospects.groupby("season").agg(prospects=("drafted", "size"), drafted=("drafted", "sum"))
    per_class["undrafted"] = per_class["prospects"] - per_class["drafted"]
    per_class["share_drafted"] = per_class["drafted"] / per_class["prospects"]
    return per_class


def sample_by_position(prospects: pd.DataFrame) -> pd.DataFrame:
    """Sample size, draft rate and typical size and speed for each position group."""
    sample = prospects[prospects["in_sample"]]
    table = pd.DataFrame({
        "position_players": prospects.groupby("grp").size(),
        "in_sample": sample.groupby("grp").size(),
        "drafted": sample.groupby("grp")["drafted"].sum(),
        "share_drafted": sample.groupby("grp")["drafted"].mean(),
        "median_height_in": sample.groupby("grp")["ht"].median(),
        "median_weight_lb": sample.groupby("grp")["wt"].median(),
        "median_forty_s": sample.groupby("grp")["forty"].median(),
    }).reindex(POSITION_GROUPS)
    total = pd.DataFrame({
        "position_players": len(prospects), "in_sample": len(sample), "drafted": sample["drafted"].sum(),
        "share_drafted": sample["drafted"].mean(), "median_height_in": sample["ht"].median(),
        "median_weight_lb": sample["wt"].median(), "median_forty_s": sample["forty"].median(),
    }, index=["All"])
    return pd.concat([table, total])


def measurement_summary(prospects: pd.DataFrame) -> pd.DataFrame:
    """Count, centre, spread and range of each measurement in the sample."""
    sample = prospects.loc[prospects["in_sample"], MEASUREMENTS]
    table = sample.describe().T[["count", "mean", "std", "min", "50%", "max"]]
    table = table.rename(columns={"count": "recorded", "50%": "median"})
    table.insert(1, "missing_share", 1 - table["recorded"] / len(sample))
    table.index = [LABELS[name] for name in table.index]
    return table.astype({"recorded": int})


def drill_coverage(prospects: pd.DataFrame) -> pd.DataFrame:
    """Share of sampled prospects with each drill on record, by position group."""
    sample = prospects[prospects["in_sample"]]
    recorded = sample[DRILLS].notna()
    table = recorded.groupby(sample["grp"]).mean().reindex(POSITION_GROUPS)
    table.loc["All"] = recorded.mean()
    table["mean_drills_done"] = sample.groupby("grp")["n_drills"].mean().reindex(POSITION_GROUPS)
    table.loc["All", "mean_drills_done"] = sample["n_drills"].mean()
    return table


def missing_forty_by_class(prospects: pd.DataFrame) -> pd.DataFrame:
    """Position players with no 40-yard dash on record, per combine class (all rows)."""
    table = prospects.groupby("season").agg(
        prospects=("forty", "size"), no_forty=("forty", lambda s: s.isna().sum()))
    table["share_no_forty"] = table["no_forty"] / table["prospects"]
    return table


def missing_forty_by_round(prospects: pd.DataFrame, first_class: int = 2017, last_class: int = 2025) -> pd.DataFrame:
    """Share with no 40-yard dash by where the prospect was drafted, for the classes given."""
    players = prospects[prospects["season"].between(first_class, last_class)]
    drafted_in = players["draft_round"].map(lambda r: "Undrafted" if pd.isna(r) else f"Round {int(r)}")
    table = players.groupby(drafted_in).agg(
        prospects=("forty", "size"), no_forty=("forty", lambda s: s.isna().sum()))
    table["share_no_forty"] = table["no_forty"] / table["prospects"]
    table.index.name = f"drafted in ({first_class}-{last_class} classes)"
    return table.reindex(ROUND_LABELS)


def drill_coverage_by_era(prospects: pd.DataFrame) -> pd.DataFrame:
    """Share of sampled prospects with each drill on record, by period."""
    sample = prospects[prospects["in_sample"]]
    era = pd.cut(sample["season"], [1999, 2010, 2016, 2021, 2026],
                 labels=["2000-2010", "2011-2016", "2017-2021", "2022-2026"])
    table = sample[DRILLS[1:]].notna().groupby(era, observed=True).mean()
    table.index.name = "combine classes"
    return table


def size_speed_correlation(prospects: pd.DataFrame) -> pd.DataFrame:
    """Correlation between weight and 40 time: across all positions, and inside each one."""
    sample = prospects[prospects["in_sample"]]
    within = pd.Series({group: rows["wt"].corr(rows["forty"]) for group, rows in sample.groupby("grp")})
    table = within.reindex(POSITION_GROUPS).rename("corr_weight_forty").to_frame()
    table.loc["All positions together"] = sample["wt"].corr(sample["forty"])
    return table


def testing_by_class(prospects: pd.DataFrame, drills: tuple[str, ...] = ("forty", "broad_jump")) -> pd.DataFrame:
    """Average result per combine class for each drill, against the position's all-years average.

    Results are standardised inside each position group among prospects who did the drill, and
    signed so that a positive number always means a better result (a faster time, a longer jump).
    """
    sample = prospects[prospects["in_sample"]]
    columns = {}
    for drill in drills:
        tested = sample[sample[drill].notna()]
        by_group = tested.groupby("grp")[drill]
        z = (tested[drill] - by_group.transform("mean")) / by_group.transform("std")
        if not HIGHER_IS_BETTER[drill]:
            z = -z
        columns[drill] = z.groupby(tested["season"]).mean()
    table = pd.DataFrame(columns)
    table.index.name = "combine class"
    return table


def drafted_by_speed(prospects: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Draft outcomes by how fast a prospect ran for his position (bin 1 = fastest fifth)."""
    sample = prospects[prospects["in_sample"] & (prospects["season"] <= 2025)].copy()
    rank_in_position = sample.groupby("grp")["forty"].rank(pct=True, method="first")
    sample["speed_bin"] = np.ceil(rank_in_position * bins).astype(int)
    table = sample.groupby("speed_bin").agg(
        prospects=("drafted", "size"), share_drafted=("drafted", "mean"),
        median_pick_if_drafted=("draft_ovr", "median"))
    table.index = [f"{i} (fastest)" if i == 1 else f"{i} (slowest)" if i == bins else str(i) for i in table.index]
    table.index.name = "40-time fifth within position"
    return table


def career_value_by_round(picks: pd.DataFrame, last_class: int = 2015) -> pd.DataFrame:
    """Career value by draft round for classes old enough to have finished careers."""
    finished = picks[picks["season"] <= last_class]
    by_round = finished.groupby("round")
    table = by_round["w_av"].agg(picks="size", mean="mean", median="median",
                                 p90=lambda s: s.quantile(0.9), max="max")
    table["never_played"] = by_round["games"].apply(lambda s: (s == 0).mean())
    table["zero_career_value"] = by_round["w_av"].apply(lambda s: (s == 0).mean())
    table.index = [f"Round {r}" for r in table.index]
    return table


def career_value_by_class(picks: pd.DataFrame) -> pd.DataFrame:
    """Average career value per pick for each draft class."""
    table = picks.groupby("season")["w_av"].agg(picks="size", mean_career_value="mean")
    table.index.name = "draft class"
    return table


def career_value_by_pick(picks: pd.DataFrame, last_class: int = 2015, window: int = 12) -> pd.DataFrame:
    """Average career value at each overall pick, smoothed over neighbouring picks.

    Only slots used in at least half of the classes are kept, so the last few compensatory
    picks of the longest drafts do not distort the end of the curve.
    """
    finished = picks[picks["season"] <= last_class]
    by_pick = finished.groupby("pick")["w_av"].agg(picks="size", mean="mean")
    by_pick = by_pick[by_pick["picks"] >= finished["season"].nunique() / 2].copy()
    by_pick["smoothed_mean"] = by_pick["mean"].rolling(window, center=True, min_periods=window // 2).mean()
    return by_pick


# --------------------------------------------------------------------------- first model
# These take the table returned by detect.score_prospects().

def unusual_by_position(scored: pd.DataFrame) -> pd.DataFrame:
    """How many prospects each position group has of each kind."""
    table = pd.crosstab(scored["grp"], scored["kind"]).reindex(index=POSITION_GROUPS, columns=KINDS, fill_value=0)
    table.loc["All"] = table.sum()
    table.insert(0, "prospects", table.sum(axis=1))
    table.index.name = table.columns.name = None
    return table


def detector_agreement(scored: pd.DataFrame) -> pd.DataFrame:
    """Correlation between the detectors' within-position ranks (1 = identical ordering)."""
    return scored[DETECTORS].corr()


def method_comparison(scored: pd.DataFrame, flag_share: float = 0.05) -> pd.DataFrame:
    """Each detector on its own, measured against the average of all three."""
    average_rank = scored.groupby("grp")["average"].rank(pct=True)
    average_flags = average_rank > 1 - flag_share
    rows = {}
    for detector in DETECTORS:
        own_flags = scored[detector] > 1 - flag_share
        rows[detector] = {
            "correlation_with_average": scored[detector].corr(scored["average"]),
            "flags": int(own_flags.sum()),
            "also_flagged_by_average": int((own_flags & average_flags).sum()),
            "share_of_average_flags": (own_flags & average_flags).sum() / average_flags.sum(),
        }
    return pd.DataFrame(rows).T.astype({"flags": int, "also_flagged_by_average": int})


def flag_overlap(scored: pd.DataFrame, flag_share: float = 0.05) -> pd.DataFrame:
    """How many prospects are in the top 5% of one, two or all three detectors."""
    n_detectors = (scored[DETECTORS] > 1 - flag_share).sum(axis=1)
    table = pd.DataFrame({
        "prospects": n_detectors.value_counts().reindex([3, 2, 1, 0], fill_value=0),
        "flagged_by_our_method": n_detectors[scored["unusual"]].value_counts().reindex([3, 2, 1, 0], fill_value=0),
    })
    table.index = ["all three detectors", "two detectors", "one detector", "none"]
    table.index.name = "in the top 5% of"
    return table


def most_unusual(scored: pd.DataFrame, kind: str | None = None, n: int = 15) -> pd.DataFrame:
    """The prospects with the highest score, optionally of one kind."""
    rows = scored if kind is None else scored[scored["kind"] == kind]
    columns = ["player_name", "season", "grp", "ht", "wt", "forty", "height_z", "weight_z", "speed_z",
               "score", "kind", "draft_ovr"]
    return rows.sort_values(["score", "mahalanobis"], ascending=False)[columns].head(n).reset_index(drop=True)


def fast_flags_by_era(own_era: pd.DataFrame, all_classes: pd.DataFrame) -> pd.DataFrame:
    """Share of prospects flagged as unusual and fast for size, by period, under two comparisons.

    `own_era` compares each prospect with classes near his own; `all_classes` with every class.
    """
    def share_by_period(scored: pd.DataFrame) -> pd.Series:
        period = pd.cut(scored["season"], [1999, 2016, 2021, 2026], labels=["2000-2016", "2017-2021", "2022-2026"])
        return (scored["kind"] == KINDS[0]).groupby(period, observed=True).mean()

    table = pd.DataFrame({"compared_with_own_era": share_by_period(own_era),
                          "compared_with_all_classes": share_by_period(all_classes)})
    table.index.name = "combine classes"
    return table


def flag_changes(own_era: pd.DataFrame, all_classes: pd.DataFrame) -> pd.DataFrame:
    """How many prospects are flagged as unusual under both comparisons, or under only one."""
    a, b = own_era["unusual"], all_classes["unusual"]
    counts = {"under both comparisons": (a & b).sum(),
              "only when compared with his own era": (a & ~b).sum(),
              "only when compared with all classes": (~a & b).sum()}
    table = pd.Series(counts, name="prospects").to_frame()
    table.index.name = "flagged as unusual"
    return table


def draft_outcomes_by_kind(scored: pd.DataFrame, last_class: int = 2025) -> pd.DataFrame:
    """Draft outcomes for each kind of prospect, for classes whose draft has happened."""
    rows = scored[scored["season"] <= last_class]
    table = rows.groupby("kind").agg(
        prospects=("drafted", "size"), share_drafted=("drafted", "mean"),
        median_pick_if_drafted=("draft_ovr", "median"),
        round_1_share_of_drafted=("draft_round", lambda rounds: (rounds.dropna() == 1).mean()))
    return table.reindex(KINDS)


def headline_numbers(prospects: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """The main numbers from this notebook in one table, computed from the data."""
    sample = prospects[prospects["in_sample"]]
    no_forty = prospects["forty"].isna()
    correlation = size_speed_correlation(prospects)["corr_weight_forty"]
    within = correlation.drop(index="All positions together")
    by_speed = drafted_by_speed(prospects)
    by_round = career_value_by_round(picks)
    testing = testing_by_class(prospects)
    recent, earlier = testing.index >= 2022, testing.index <= 2016

    rows = [
        ("Combine classes", f"{prospects.season.min()}-{prospects.season.max()} ({prospects.season.nunique()} classes)"),
        ("Position players in the cleaned table", f"{len(prospects):,}"),
        ("Prospects in the analysis sample", f"{len(sample):,}"),
        ("  of whom drafted", f"{sample.drafted.sum():,} ({sample.drafted.mean():.0%})"),
        ("Drill results missing in the sample", f"{1 - sample[DRILLS].notna().to_numpy().mean():.0%}"),
        ("Average drills completed, of 6", f"{sample.n_drills.mean():.1f}"),
        ("No 40-yard dash on record: 2000-2016 classes", f"{no_forty[prospects.season <= 2016].mean():.1%}"),
        ("No 40-yard dash on record: 2017-2021 classes", f"{no_forty[prospects.season.between(2017, 2021)].mean():.0%}"),
        ("No 40-yard dash on record: 2022-2026 classes", f"{no_forty[prospects.season >= 2022].mean():.0%}"),
        ("Correlation of weight and 40 time, all positions", f"{correlation['All positions together']:.2f}"),
        ("Correlation of weight and 40 time, inside a position", f"{within.min():.2f} to {within.max():.2f}"),
        ("40 time of 2022-2026 classes vs 2000-2016 classes", f"{testing.forty[recent].mean() - testing.forty[earlier].mean():+.2f} SD (faster)"),
        ("Share drafted: fastest fifth vs slowest fifth at a position", f"{by_speed.share_drafted.iloc[0]:.0%} vs {by_speed.share_drafted.iloc[-1]:.0%}"),
        ("Median pick: fastest fifth vs slowest fifth", f"{by_speed.median_pick_if_drafted.iloc[0]:.0f} vs {by_speed.median_pick_if_drafted.iloc[-1]:.0f}"),
        ("Mean career value: round 1 vs round 7 (2000-2015 classes)", f"{by_round['mean'].iloc[0]:.1f} vs {by_round['mean'].iloc[-1]:.1f}"),
        ("Career value of zero: round 1 vs round 7", f"{by_round.zero_career_value.iloc[0]:.0%} vs {by_round.zero_career_value.iloc[-1]:.0%}"),
    ]
    return pd.DataFrame(rows, columns=["what", "value"]).set_index("what")


def show(table: pd.DataFrame, percent: tuple[str, ...] | list[str] = (), decimals: int = 2) -> pd.DataFrame:
    """Format a table for display: the `percent` columns as 12%, other decimals rounded."""
    formatted = table.copy()
    for column in formatted.columns:
        if column in percent:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.0%}")
        elif pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].round(decimals)
    return formatted
