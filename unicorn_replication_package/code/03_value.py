"""03_value.py - draft-slot value model and bust / good value / steal grades.
E[AV | pick, class, position] = f(pick) * g(class) * m(position)
  f : monotone LOWESS of career weighted AV on overall pick, fitted on mature classes (2000-2015)
  g : class factor = realised class AV / AV implied by f (absorbs unfinished careers for recent classes)
  m : position multiplier (AV is more generous to some positions than others)
"""
import pandas as pd, numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

LAST_GRADED = 2022          # classes with >= 4 completed seasons
MATURE = 2015

d = pd.read_csv("out/draft_all.csv")
d = d[(d.season >= 2000) & (d.season <= 2025)].copy()
d = d[~d.position.isin(["K", "P", "LS"])]
d["w_av"] = d.w_av.fillna(0).clip(lower=0)
pm = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE", "T": "OT", "OT": "OT", "G": "IOL", "C": "IOL", "OL": "IOL", "OG": "IOL",
      "DT": "DT", "NT": "DT", "DL": "DT", "DE": "EDGE", "OLB": "LB", "LB": "LB", "ILB": "LB", "CB": "CB", "DB": "CB",
      "S": "S", "SAF": "S", "FS": "S", "SS": "S"}
players = pd.read_csv("data/players.csv", low_memory=False)[["pfr_id", "position"]].dropna().drop_duplicates("pfr_id")
d = d.merge(players.rename(columns={"pfr_id": "pfr_player_id", "position": "nfl_pos"}), on="pfr_player_id", how="left")
generic = d.position.isin(["OL", "DL", "DB"]) & d.nfl_pos.map(pm).notna() & ~d.nfl_pos.isin(["OL", "DL", "DB", "LB", "OLB"])
d["vgrp"] = np.where(generic, d.nfl_pos.map(pm), d.position.map(pm))
s = pd.read_csv("out/scored.csv")
key = s[s.drafted][["draft_year", "draft_ovr", "grp"]].rename(columns={"draft_year": "season", "draft_ovr": "pick"})
d = d.merge(key, on=["season", "pick"], how="left")
d["vgrp"] = d.grp.fillna(d.vgrp); d = d[d.vgrp.notna()].drop(columns="grp")

# ---- f(pick)
m = d[d.season <= MATURE]
grid = np.arange(1, 271)
lw = lowess(m.w_av, np.log(m.pick), frac=0.25, it=0, xvals=np.log(grid))
f = np.minimum.accumulate(lw)                      # enforce monotone decline
d["f"] = np.interp(d.pick, grid, f)
# ---- g(class)
g = d.groupby("season").apply(lambda x: x.w_av.sum() / x.f.sum())
d["g"] = d.season.map(g)
# ---- m(position)
mm = d[d.season <= LAST_GRADED].groupby("vgrp").apply(lambda x: x.w_av.sum() / (x.f * x.g).sum())
d["m"] = d.vgrp.map(mm)
d["exp_av"] = d.f * d.g * d.m
# ---- spread around expectation, as a function of pick (mature classes, in units of f)
mat = d[d.season <= MATURE]
rel = (mat.w_av - mat.exp_av) / (mat.g * mat.m)
sd = np.sqrt(np.clip(lowess(rel ** 2, np.log(mat.pick), frac=0.3, it=0, xvals=np.log(grid)), 1e-6, None))
d["sd_av"] = np.interp(d.pick, grid, sd) * d.g * d.m
d["voe"] = d.w_av - d.exp_av                       # value over expected
d["z"] = d.voe / d.sd_av
d["ratio"] = d.w_av / d.exp_av

def grade(r):
    if r.season > LAST_GRADED: return "Too early"
    if r.z >= 0.75 and r.ratio >= 1.5: return "Steal"
    if r.z <= -0.75 and r.ratio < 0.5: return "Bust"
    return "Good value"
d["grade"] = d.apply(grade, axis=1)
d.to_csv("out/draft_valued.csv", index=False)
pd.DataFrame({"pick": grid, "f": f, "sd": sd}).to_csv("out/value_curve.csv", index=False)
g.rename("g").to_csv("out/class_factor.csv"); mm.rename("m").to_csv("out/pos_mult.csv")

if __name__ == "__main__":
    print(pd.Series(f, index=grid)[[1, 5, 10, 20, 32, 64, 100, 150, 200, 250]].round(1).to_string())
    print(pd.Series(sd, index=grid)[[1, 5, 10, 20, 32, 64, 100, 150, 200, 250]].round(1).to_string())
    print(g.round(2).to_string()); print(mm.round(2).to_string())
    gg = d[d.season <= LAST_GRADED]
    print(pd.crosstab(gg["round"], gg.grade, normalize="index").round(3))
    print(gg.grade.value_counts(normalize=True).round(3))
    print(gg.groupby("round").apply(lambda x: (-x.exp_av / x.sd_av).mean()).round(2))
