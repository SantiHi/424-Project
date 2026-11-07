"""Score how unusual each prospect's size and speed are for his position.

First version of the detection step. It uses the three measurements every sampled prospect
has (height, weight, 40-yard dash), so no skipped drill has to be filled in yet.

For each position group:
  1. Features. Height, weight and "speed for size": the 40 time compared with what the
     prospect's height and weight predict. Each one is measured against prospects at the same
     position from classes within two years of his own, so nobody looks unusual only because
     of when he was tested.
  2. Score. The Mahalanobis distance: how far a prospect is from the centre of his position,
     allowing for how the three features move together.
  3. Flag. The top 5% of each position are flagged as unusual.

Two other detectors, Isolation Forest and Local Outlier Factor, are computed alongside so the
choice of method can be checked (see METHOD below and settings_sensitivity()).

Nothing here sees draft position or career outcomes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

FEATURES = ["height_z", "weight_z", "speed_z"]
DETECTORS = ["isolation_forest", "local_outlier_factor", "mahalanobis"]
KINDS = ["Unusual, fast for size", "Unusual build", "Unusual, slow for size", "Typical"]
# The one detector the project uses. Of the three it agrees best with the average of all three,
# and it is the only one with no setting to choose and no randomness, so it gives the same
# flags on every run. Pass method="average" to score_prospects() for the three-detector average.
METHOD = "mahalanobis"
ERA_WINDOW = 2       # classes on each side of a prospect's own that count as his era
FLAG_SHARE = 0.05    # share of each position group flagged as unusual
SPEED_CUT = 1.0      # speed_z beyond this (either way) counts as clearly fast or slow for size
N_TREES = 500
N_NEIGHBOURS = 35
SEED = 0


# --------------------------------------------------------------------------- features

def era_centre(values: pd.Series, seasons: pd.Series, window: int | None) -> pd.Series:
    """The median among prospects from classes within `window` years of each prospect's own.

    window=None ignores the era and uses the median over every class.
    """
    if window is None:
        return pd.Series(values.median(), index=values.index)
    centre_by_class = {season: values[(seasons - season).abs() <= window].median()
                       for season in seasons.unique()}
    return seasons.map(centre_by_class)


def robust_spread(values: pd.Series) -> float:
    """Spread from the median absolute deviation, scaled to match a standard deviation.

    Unlike the standard deviation it is not stretched by the outliers we are looking for.
    """
    spread = 1.4826 * (values - values.median()).abs().median()
    return spread if spread > 0 else values.std()


def forty_vs_size(group: pd.DataFrame) -> pd.Series:
    """Seconds slower (+) or faster (-) than a prospect's height and weight predict.

    Fitted inside one position group: 40 time on height, weight and weight squared.
    """
    design = np.column_stack([np.ones(len(group)), group["ht"], group["wt"], group["wt"] ** 2])
    coefficients, *_ = np.linalg.lstsq(design, group["forty"].to_numpy(), rcond=None)
    return group["forty"] - design @ coefficients


def position_features(group: pd.DataFrame, era_window: int | None = ERA_WINDOW) -> pd.DataFrame:
    """Height, weight and speed-for-size for one position group, relative to each prospect's era.

    All three are in robust standard deviations. A positive speed_z means faster than expected.
    """
    measurements = {"height_z": group["ht"], "weight_z": group["wt"], "speed_z": -forty_vs_size(group)}
    features = {}
    for name, values in measurements.items():
        centred = values - era_centre(values, group["season"], era_window)
        features[name] = centred / robust_spread(centred)
    return pd.DataFrame(features, index=group.index)


# --------------------------------------------------------------------------- detectors
# Each takes the feature matrix of one position group and returns one score per prospect,
# where a higher score means more unusual.

def isolation_forest_score(features: np.ndarray, seed: int = SEED) -> np.ndarray:
    """How easily random splits separate a prospect from everyone else."""
    forest = IsolationForest(n_estimators=N_TREES, random_state=seed).fit(features)
    return -forest.score_samples(features)


def local_outlier_factor_score(features: np.ndarray, neighbours: int = N_NEIGHBOURS) -> np.ndarray:
    """How sparse a prospect's neighbourhood is compared with his neighbours' own."""
    neighbours = min(neighbours, len(features) // 10)
    return -LocalOutlierFactor(n_neighbors=neighbours).fit(features).negative_outlier_factor_


def mahalanobis_score(features: np.ndarray) -> np.ndarray:
    """Distance from the position's centre, allowing for how the features move together."""
    centred = features - features.mean(axis=0)
    inverse_covariance = np.linalg.inv(np.cov(features, rowvar=False))
    return np.sqrt(np.einsum("ij,jk,ik->i", centred, inverse_covariance, centred))


# --------------------------------------------------------------------------- pipeline

def score_prospects(prospects: pd.DataFrame, method: str = METHOD, era_window: int | None = ERA_WINDOW,
                    flag_share: float = FLAG_SHARE, seed: int = SEED) -> pd.DataFrame:
    """Score every sampled prospect inside his position group and flag the most unusual.

    Returns the sampled prospects with these columns added:
      height_z, weight_z, speed_z                           the features
      isolation_forest, local_outlier_factor, mahalanobis   each detector's rank in the position (0-1)
      average                                               the average of the three ranks
      score                                                 the rank from `method`, which decides the flag
      unusual                                               True for the top `flag_share` by score
      kind                                                  Typical, or which way an unusual prospect stands out
    """
    sample = prospects[prospects["in_sample"]]
    scored_groups = []
    for _, group in sample.groupby("grp"):
        features = position_features(group, era_window)
        matrix = features.to_numpy()
        scores = pd.DataFrame({
            "isolation_forest": isolation_forest_score(matrix, seed),
            "local_outlier_factor": local_outlier_factor_score(matrix),
            "mahalanobis": mahalanobis_score(matrix),
        }, index=group.index)
        ranks = scores.rank(pct=True)
        ranks["average"] = ranks[DETECTORS].mean(axis=1)
        ranks["score"] = ranks[method].rank(pct=True)
        ranks["unusual"] = ranks["score"] > 1 - flag_share
        scored_groups.append(group.join(features).join(ranks))

    scored = pd.concat(scored_groups).sort_index()
    scored["kind"] = np.select(
        [~scored["unusual"], scored["speed_z"] >= SPEED_CUT, scored["speed_z"] <= -SPEED_CUT],
        [KINDS[3], KINDS[0], KINDS[2]], default=KINDS[1])
    return scored


def settings_sensitivity(prospects: pd.DataFrame, era_window: int | None = ERA_WINDOW,
                         flag_share: float = FLAG_SHARE) -> pd.DataFrame:
    """How many of each detector's flags change when its own arbitrary setting changes.

    Isolation Forest is built from random trees, so it is re-run with a different random seed.
    Local Outlier Factor needs a neighbourhood size, so it is re-run with 20 neighbours instead
    of 35. Mahalanobis distance has no setting and no randomness, so a re-run changes nothing.
    """
    reruns = {
        "isolation_forest": ("a different random seed",
                             lambda m: isolation_forest_score(m, SEED), lambda m: isolation_forest_score(m, SEED + 1)),
        "local_outlier_factor": ("20 neighbours instead of 35",
                                 local_outlier_factor_score, lambda m: local_outlier_factor_score(m, 20)),
        "mahalanobis": ("nothing to change: re-run as is", mahalanobis_score, mahalanobis_score),
    }
    sample = prospects[prospects["in_sample"]]
    matrices = [position_features(group, era_window).to_numpy() for _, group in sample.groupby("grp")]
    rows = []
    for detector, (change, first_run, second_run) in reruns.items():
        flagged = changed = 0
        for matrix in matrices:
            before = pd.Series(first_run(matrix)).rank(pct=True) > 1 - flag_share
            after = pd.Series(second_run(matrix)).rank(pct=True) > 1 - flag_share
            flagged += int(before.sum())
            changed += int((before & ~after).sum())
        rows.append({"detector": detector, "what_was_changed": change, "flags": flagged,
                     "flags_that_change": changed})
    return pd.DataFrame(rows).set_index("detector")
