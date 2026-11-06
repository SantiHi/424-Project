"""02_detect.py - unsupervised unicorn detection, position by position.
Pipeline per position group:
  1. Impute skipped drills (IterativeImputer, BayesianRidge) inside the position group.
  2. Size/era-adjust every drill: residual from OLS on height, weight, weight^2, combine year.
  3. Standardise robustly (median / MAD).
  4. Three detectors: Isolation Forest, Local Outlier Factor, robust Mahalanobis (MCD).
  5. Consensus score = mean within-position percentile rank of the three scores.
"""
import pandas as pd, numpy as np, warnings
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from scipy.stats import rankdata
warnings.filterwarnings("ignore")

SEED = 42
df = pd.read_csv("out/prepared.csv")
drills = ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
sign = {"forty": -1, "bench": 1, "vertical": 1, "broad_jump": 1, "cone": -1, "shuttle": -1}  # +1 = bigger is better
MIN_COVER = 0.25   # a drill is dropped for a position if fewer than 25% of prospects did it (QB bench)

def rz(x, ref=None):
    """robust z-score; location/scale are estimated on `ref` (observed values only)."""
    ref = x if ref is None else ref
    med = np.median(ref); mad = np.median(np.abs(ref - med)) * 1.4826
    return (x - med) / mad

parts, loadings, dropped = [], [], []
for g, sub in df.groupby("grp"):
    sub = sub.copy()
    gd = [c for c in drills if sub[c].notna().mean() >= MIN_COVER]
    feat = ["z_ht", "z_wt"] + ["adj_" + c for c in gd]
    raw = sub[["ht", "wt"] + gd]
    imp = IterativeImputer(random_state=SEED, max_iter=25, sample_posterior=False)
    X = pd.DataFrame(imp.fit_transform(raw), columns=raw.columns, index=sub.index)
    # size + era adjustment, fitted on prospects who actually ran the drill
    # (2021 had no combine: its numbers are pro-day, hand-timed and systematically flattering, so it gets its own shift)
    A = np.column_stack([np.ones(len(sub)), X.ht, X.wt, X.wt ** 2, sub.season - 2013, (sub.season == 2021).astype(float)])
    for c in drills:
        if c not in gd:
            sub["adj_" + c] = np.nan; continue
        obs = sub[c].notna().values
        beta, *_ = np.linalg.lstsq(A[obs], X[c].values[obs], rcond=None)
        res = X[c].values - A @ beta
        sub["adj_" + c] = rz(res, res[obs]) * sign[c]  # oriented: positive = better than size predicts
    sub["z_ht"] = rz(X.ht.values); sub["z_wt"] = rz(X.wt.values)
    # data QA: rows whose size is impossible for the listed position are mislabeled, not unicorns
    # (almost all come from the 2021 pro-day-only class). Rules are deliberately narrow so that real
    # size outliers (370-lb tackles, a 288-lb quarterback) stay in.
    bad = pd.Series(False, index=sub.index)
    if g in ("IOL", "OT"): bad = sub.wt < 270
    if g == "LB": bad = (sub.wt > 275) | (sub.wt < 210)
    if bad.any():
        dropped.append(sub.loc[bad, ["player_name", "season", "pos", "grp", "ht", "wt", "forty"]])
        sub = sub[~bad].copy()

    # two feature "views": full workout (all drills, imputed where skipped) and size-speed (always observed)
    views = {"full": feat, "core": ["z_ht", "z_wt", "adj_forty"]}
    for v, cols in views.items():
        F = sub[cols].values
        s_if = -IsolationForest(n_estimators=500, random_state=SEED).fit(F).score_samples(F)
        s_lof = -LocalOutlierFactor(n_neighbors=min(35, len(sub) // 10)).fit(F).negative_outlier_factor_
        s_md = np.sqrt(MinCovDet(random_state=SEED, support_fraction=0.85).fit(F).mahalanobis(F))
        for nm, s_ in [("if", s_if), ("lof", s_lof), ("md", s_md)]:
            sub[f"s_{nm}_{v}"] = s_
            sub[f"p_{nm}_{v}"] = rankdata(s_) / len(s_)
        sub["cons_" + v] = rankdata(sub[[f"p_if_{v}", f"p_lof_{v}", f"p_md_{v}"]].mean(axis=1)) / len(sub)
    for nm in ["if", "lof", "md"]:
        sub["p_" + nm] = sub[[f"p_{nm}_full", f"p_{nm}_core"]].max(axis=1)
        sub["flag_" + nm] = sub["p_" + nm] > 0.975
    sub["consensus"] = sub[["cons_full", "cons_core"]].max(axis=1)
    sub["cons_pct"] = rankdata(sub.consensus) / len(sub)
    F = sub[feat].values

    # athleticism composite (measured drills only get full weight; imputed drills contribute their shrunken value)
    sub["ath"] = sub[["adj_" + c for c in gd]].mean(axis=1)
    sub["ath_z"] = (sub.ath - sub.ath.mean()) / sub.ath.std()

    pca = PCA(n_components=2, random_state=SEED).fit(F)
    pcs = pca.transform(F)
    sub["pc1"], sub["pc2"] = pcs[:, 0], pcs[:, 1]
    loadings.append(pd.DataFrame(pca.components_.T, index=feat, columns=["pc1", "pc2"]).assign(grp=g, evr1=pca.explained_variance_ratio_[0], evr2=pca.explained_variance_ratio_[1]))
    parts.append(sub)

out = pd.concat(parts)
for q, nm in [(0.98, "u02"), (0.95, "u05"), (0.90, "u10")]:
    out[nm] = out.cons_pct > q
out["unicorn"] = out.u05

# direction of the anomaly
def typ(r):
    if not r.unicorn: return "Typical"
    if r.ath_z >= 0.75: return "Freak"
    if r.ath_z <= -0.75: return "Red flag"
    return "Odd build"
out["utype"] = out.apply(typ, axis=1)
out.to_csv("out/scored.csv", index=False)
pd.concat(loadings).to_csv("out/pca_loadings.csv")

print(out.utype.value_counts())
print(out.groupby("grp").utype.value_counts().unstack().fillna(0).astype(int))
print("detector agreement (top-5% flags):")
print(out[["flag_if", "flag_lof", "flag_md"]].corr().round(2))
print(out[["s_if_full", "s_lof_full", "s_md_full"]].corr(method="spearman").round(2))
print(out[["s_if_core", "s_lof_core", "s_md_core"]].corr(method="spearman").round(2))
dq = pd.concat(dropped); dq.to_csv("out/dropped_data_errors.csv", index=False); print(dq.to_string())
show = ["player_name", "season", "grp", "ht", "wt", "forty", "vertical", "broad_jump", "ath_z", "cons_pct", "draft_ovr", "w_av"]
print(out[out.utype == "Freak"].sort_values("consensus", ascending=False)[show].head(25).to_string())
print(out.groupby("grp").ath_z.describe().round(2))
for n in ["Calvin Johnson", "D.K. Metcalf", "Julio Jones", "Micah Parsons", "Anthony Richardson", "Cam Newton", "Derrick Henry", "Travon Walker", "Trent Williams", "J.J. Watt", "Vernon Davis", "Myles Garrett", "Aaron Donald", "Saquon Barkley", "Jadeveon Clowney", "Tristan Wirfs", "Kyle Pitts"]:
    print(out[out.player_name == n][show + ["utype"]].to_string(header=False))
