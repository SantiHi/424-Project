"""Clean the raw nflverse files into one table of draft prospects.

Inputs (data/raw/, downloaded from https://github.com/nflverse/nflverse-data in October 2026):
    combine.csv      one row per combine participant, 2000-2026: size, six drills, draft slot
    draft_picks.csv  one row per draft pick since 1980: pick number, team, career value
    players.csv      NFL roster records, used only for the position a player held in the NFL

Output (data/processed/prospects.csv): one row per position player, with measurements in
consistent units, a position group, the draft outcome and career value.

`build_prospects()` also returns a log of what each cleaning step changed, so the notebook
can show the preprocessing rather than describe it.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

DRILLS = ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
CORE_MEASUREMENTS = ["ht", "wt", "forty"]  # a prospect needs all three to be analysed
# Timed drills are better when lower; counted and measured drills are better when higher.
HIGHER_IS_BETTER = {"forty": False, "bench": True, "vertical": True, "broad_jump": True,
                    "cone": False, "shuttle": False}
FIRST_DRAFT_CLASS = 2000
LAST_DRAFT_CLASS = 2025  # newest class with a completed season when the data was downloaded
POSITION_GROUPS = ["QB", "RB", "WR", "TE", "OT", "IOL", "DT", "EDGE", "LB", "CB", "S"]

# Combine labels that map to exactly one position group.
DIRECT_GROUP = {
    "QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE",
    "OT": "OT", "OG": "IOL", "G": "IOL", "C": "IOL",
    "DT": "DT", "DE": "EDGE", "EDGE": "EDGE",
    "ILB": "LB", "LB": "LB",
    "CB": "CB", "S": "S", "SAF": "S",
}
# Labels that could be two groups; resolved in assign_position_group().
AMBIGUOUS_LABELS = ["OLB", "OL", "DL", "DB"]
# Specialists and fullbacks are not compared on athletic testing, and "CB/WR" fits no group.
EXCLUDED_LABELS = ["K", "P", "LS", "FB", "CB/WR"]

# Size cut-offs used only when a label is ambiguous and no NFL position is known.
EDGE_MIN_WEIGHT = 245      # an "OLB" this heavy is an edge rusher, lighter is an off-ball linebacker
TACKLE_MIN_HEIGHT = 77     # inches; a generic "OL" this tall is a tackle
INTERIOR_DL_MIN_WEIGHT = 285
SAFETY_MIN_WEIGHT = 200

# A weight outside these bounds means the row's position label is wrong, not that the player
# is a rare athlete (for example a 195-lb "offensive lineman"). Bounds are (min, max) in lb.
PLAUSIBLE_WEIGHT = {"OT": (270, np.inf), "IOL": (270, np.inf), "LB": (210, 275)}

# Broad families, used to decide which of two same-named prospects a draft pick belongs to.
POSITION_FAMILY = {
    **dict.fromkeys(["QB", "RB", "FB", "WR", "TE"], "skill"),
    **dict.fromkeys(["T", "OT", "G", "OG", "C", "OL"], "offensive line"),
    **dict.fromkeys(["DT", "NT", "DE", "DL", "EDGE", "LB", "OLB", "ILB"], "front seven"),
    **dict.fromkeys(["CB", "DB", "S", "SAF", "FS", "SS"], "secondary"),
}

# Franchises that moved keep one identity (Pro-Football-Reference team codes).
RELOCATED = {"STL": "LAR", "SDG": "LAC", "OAK": "LVR"}

DRAFT_COLUMNS = ["season", "round", "pick", "team", "position", "w_av", "games",
                 "seasons_started", "probowls", "allpro"]
OUTPUT_COLUMNS = (
    ["player_name", "season", "school", "pos", "grp", "ht", "wt"] + DRILLS
    + ["n_drills", "drafted", "draft_year", "draft_round", "draft_ovr", "team", "franchise",
       "draft_pos", "nfl_pos", "pfr_id", "w_av", "games", "seasons_started", "probowls",
       "allpro", "size_error", "in_sample"]
)

_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


# --------------------------------------------------------------------------- loading

def load_raw(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Read the three raw files exactly as downloaded."""
    return {
        "combine": pd.read_csv(raw_dir / "combine.csv"),
        "draft_picks": pd.read_csv(raw_dir / "draft_picks.csv"),
        "players": pd.read_csv(raw_dir / "players.csv", low_memory=False),
    }


def load_prospects(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Read the cleaned table written by build_prospects()."""
    return pd.read_csv(processed_dir / "prospects.csv")


# --------------------------------------------------------------------------- small helpers

def height_to_inches(height: object) -> float:
    """Convert a feet-inches string such as "6-4" to inches (76); anything else becomes NaN."""
    try:
        feet, inches = str(height).split("-")
        return int(feet) * 12 + int(inches)
    except ValueError:
        return np.nan


def name_key(name: object) -> str:
    """Lower-case a name and drop punctuation and suffixes, so "Ted Ginn Jr." matches "Ted Ginn"."""
    words = re.sub(r"[^a-z ]", "", str(name).lower()).split()
    return " ".join(word for word in words if word not in _NAME_SUFFIXES)


# --------------------------------------------------------------------------- cleaning steps

def link_unlinked_classes(combine: pd.DataFrame, draft: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill in draft slots for classes the combine file has not been linked to yet.

    nflverse adds the draft columns to the combine file some time after each draft. A class that
    has picks in the draft file but none in the combine file (2026 at download time) is matched
    on player name within that class. Names that are not unique on either side are left alone.
    """
    combine = combine.copy()
    linked_per_class = combine.groupby("season")["draft_ovr"].count()
    unlinked = [season for season, n_linked in linked_per_class.items()
                if n_linked == 0 and (draft["season"] == season).any()]

    n_linked = 0
    for season in unlinked:
        picks = draft.loc[draft["season"] == season, ["pfr_player_name", "round", "pick"]].copy()
        picks["key"] = picks["pfr_player_name"].map(name_key)
        picks = picks.drop_duplicates("key", keep=False).set_index("key")

        in_class = combine["season"] == season
        keys = combine.loc[in_class, "player_name"].map(name_key)
        keys = keys[~keys.duplicated(keep=False) & keys.isin(picks.index)]

        combine.loc[keys.index, "draft_year"] = season
        combine.loc[keys.index, "draft_round"] = keys.map(picks["round"]).astype(float)
        combine.loc[keys.index, "draft_ovr"] = keys.map(picks["pick"]).astype(float)
        n_linked += len(keys)
    return combine, n_linked


def clear_other_year_picks(combine: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove draft slots that belong to a different year than the prospect's combine.

    A prospect is drafted in the year he attends the combine. Where the raw file says otherwise,
    two players with the same name have been given one player ID, and the pick belongs to the
    other one. The row keeps its measurements and is treated as undrafted.
    """
    combine = combine.copy()
    wrong_year = combine["draft_year"].notna() & (combine["draft_year"] != combine["season"])
    combine.loc[wrong_year, ["draft_year", "draft_round", "draft_ovr"]] = np.nan
    combine.loc[wrong_year, ["draft_team", "pfr_id"]] = None
    return combine, int(wrong_year.sum())


def attach_draft_outcomes(combine: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """Add the pick's team, drafted position and career value from the draft file."""
    picks = draft.loc[draft["season"] >= FIRST_DRAFT_CLASS, DRAFT_COLUMNS].rename(
        columns={"season": "draft_year", "pick": "draft_ovr", "position": "draft_pos"})
    merged = combine.merge(picks, on=["draft_year", "draft_ovr"], how="left", validate="many_to_one")
    if (merged["draft_ovr"].notna() & merged["round"].isna()).any():
        raise ValueError("a combine row points at a draft pick that is not in the draft file")
    return merged.drop(columns="round")


def drop_shared_pick_namesakes(prospects: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Resolve draft picks that two same-named prospects from the same class both claim.

    The pick stays with the prospect whose combine position is in the same family as the drafted
    position. The other row is dropped because its real draft outcome cannot be recovered. If
    position does not single out one row, every row claiming the pick is dropped.
    """
    pick = ["draft_year", "draft_ovr"]
    shared = prospects["draft_ovr"].notna() & prospects.duplicated(pick, keep=False)
    same_family = (prospects["pos"].map(POSITION_FAMILY) == prospects["draft_pos"].map(POSITION_FAMILY))
    matches_per_pick = same_family.where(shared, False).groupby(
        [prospects["draft_year"], prospects["draft_ovr"]]).transform("sum")
    keeps_pick = same_family & (matches_per_pick == 1)
    drop = shared & ~keeps_pick
    return prospects[~drop].copy(), int(drop.sum())


def attach_nfl_position(prospects: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Add the position a player held on an NFL roster (missing for players who never signed)."""
    nfl = (players[["pfr_id", "position"]].dropna().drop_duplicates("pfr_id")
           .rename(columns={"position": "nfl_pos"}))
    return prospects.merge(nfl, on="pfr_id", how="left", validate="many_to_one")


def assign_position_group(prospects: pd.DataFrame) -> pd.Series:
    """Map each combine position label to one of the eleven position groups.

    Most labels map directly. Four are ambiguous (OLB, OL, DL, DB); they are resolved with the
    position the player went on to hold in the NFL, or the drafted position, and by body size
    when neither is known. Specialists and fullbacks get no group.
    """
    pos, height, weight = prospects["pos"], prospects["ht"], prospects["wt"]
    pro_pos = prospects["nfl_pos"].fillna(prospects["draft_pos"])
    group = pos.map(DIRECT_GROUP).astype(object)

    olb = pos == "OLB"
    group[olb] = np.where(weight[olb] >= EDGE_MIN_WEIGHT, "EDGE", "LB")

    ol = pos == "OL"
    by_size = np.where(height[ol] >= TACKLE_MIN_HEIGHT, "OT", "IOL")
    group[ol] = np.select([pro_pos[ol].isin(["T", "OT"]), pro_pos[ol].isin(["G", "OG", "C"])],
                          ["OT", "IOL"], default=by_size)

    dl = pos == "DL"
    by_size = np.where(weight[dl] >= INTERIOR_DL_MIN_WEIGHT, "DT", "EDGE")
    group[dl] = np.select([pro_pos[dl].isin(["DT", "NT"]), pro_pos[dl].isin(["DE", "OLB", "LB"])],
                          ["DT", "EDGE"], default=by_size)

    db = pos == "DB"
    by_size = np.where(weight[db] >= SAFETY_MIN_WEIGHT, "S", "CB")
    group[db] = np.select([pro_pos[db] == "CB", pro_pos[db].isin(["S", "SAF", "FS", "SS"])],
                          ["CB", "S"], default=by_size)

    unknown = set(pos[group.isna()]) - set(EXCLUDED_LABELS)
    if unknown:
        raise ValueError(f"combine position labels with no rule: {sorted(unknown)}")
    return group


def implausible_size(prospects: pd.DataFrame) -> pd.Series:
    """Flag rows whose weight is impossible for the listed position (a mislabelled row)."""
    flagged = pd.Series(False, index=prospects.index)
    for group, (low, high) in PLAUSIBLE_WEIGHT.items():
        in_group = prospects["grp"] == group
        flagged |= in_group & ((prospects["wt"] < low) | (prospects["wt"] > high))
    return flagged


def clean_draft_picks(draft: pd.DataFrame) -> pd.DataFrame:
    """Every pick of a position player in the 2000-2025 drafts, with career value filled in.

    This table includes players who never attended the combine, so it is the right basis for
    "what does a pick at this slot usually return". A missing career value means the player
    never appeared in an NFL game and is set to 0.
    """
    in_range = draft["season"].between(FIRST_DRAFT_CLASS, LAST_DRAFT_CLASS)
    specialist = draft["position"].isin(["K", "P", "LS"])
    picks = draft.loc[in_range & ~specialist, DRAFT_COLUMNS].copy()
    picks[["w_av", "games"]] = picks[["w_av", "games"]].fillna(0)
    picks["franchise"] = picks["team"].replace(RELOCATED)
    return picks.reset_index(drop=True)


# --------------------------------------------------------------------------- pipeline

def build_prospects(raw_dir: Path = RAW_DIR, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every cleaning step and return (prospects, log).

    `prospects` holds every position player. The column `in_sample` marks the rows the analysis
    uses: height, weight and a 40-yard dash on record, and a plausible size for the position.
    `log` has one row per step with the number of rows it changed and the rows remaining.
    """
    raw = load_raw(raw_dir)
    draft, players = raw["draft_picks"], raw["players"]
    log = []

    def record(step: str, changed: int, remaining: int, effect: str) -> None:
        log.append({"step": step, "rows_changed": changed, "rows_remaining": remaining, "effect": effect})

    df = raw["combine"].copy()
    record("Raw combine file", 0, len(df), "starting point")

    df["ht"] = df["ht"].map(height_to_inches)

    df, n = link_unlinked_classes(df, draft)
    record("Link newest class to its draft picks by name", n, len(df), "draft slot filled in")

    df, n = clear_other_year_picks(df)
    record("Clear picks from a different year than the combine", n, len(df), "treated as undrafted")

    df = attach_draft_outcomes(df, draft)
    df, n = drop_shared_pick_namesakes(df)
    record("Two same-class namesakes claim one pick", n, len(df), "row dropped")

    df = attach_nfl_position(df, players)
    df["drafted"] = df["draft_ovr"].notna()

    never_played = df["drafted"] & df["w_av"].isna()
    df.loc[never_played, ["w_av", "games"]] = 0
    record("Drafted but no career value on record", int(never_played.sum()), len(df), "career value set to 0")

    df["grp"] = assign_position_group(df)
    n = int(df["grp"].isna().sum())
    df = df[df["grp"].notna()].copy()
    record("Specialists and fullbacks (K, P, LS, FB)", n, len(df), "row dropped")

    df["n_drills"] = df[DRILLS].notna().sum(axis=1)
    df["franchise"] = df["team"].replace(RELOCATED)

    has_core = df[CORE_MEASUREMENTS].notna().all(axis=1)
    record("No height, weight or 40-yard dash", int((~has_core).sum()), int(has_core.sum()), "left out of the sample")

    df["size_error"] = implausible_size(df)
    df["in_sample"] = has_core & ~df["size_error"]
    record("Weight impossible for the listed position", int((has_core & df["size_error"]).sum()),
           int(df["in_sample"].sum()), "left out of the sample")

    df = df[OUTPUT_COLUMNS].sort_values(["season", "player_name"]).reset_index(drop=True)
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DIR / "prospects.csv", index=False)
    return df, pd.DataFrame(log)


if __name__ == "__main__":
    prospects, cleaning_log = build_prospects()
    print(cleaning_log.to_string(index=False))
    print(f"\nwrote {len(prospects):,} rows to {PROCESSED_DIR / 'prospects.csv'}")
