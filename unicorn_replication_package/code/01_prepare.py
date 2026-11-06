"""01_prepare.py - load nflverse combine + draft data, harmonise positions, build the modelling frame."""
import pandas as pd, numpy as np

comb = pd.read_csv("data/combine.csv")
draft = pd.read_csv("data/draft_picks.csv")

# ---- height "6-4" -> inches
def ht_in(s):
    try:
        f, i = str(s).split("-"); return int(f) * 12 + int(i)
    except Exception:
        return np.nan
comb["ht"] = comb["ht"].map(ht_in)

# ---- attach draft outcome by (draft year, overall pick)
d = draft[draft.season >= 2000][["season", "round", "pick", "team", "pfr_player_id", "pfr_player_name",
                                 "position", "w_av", "dr_av", "games", "seasons_started", "probowls", "allpro"]]
d = d.rename(columns={"season": "draft_year", "pick": "draft_ovr", "position": "draft_pos"})
# the 2026 class has no draft columns in the combine file yet: fill them by exact name match to the draft file
d26 = draft[draft.season == 2026].drop_duplicates("pfr_player_name").set_index("pfr_player_name")
m26 = (comb.season == 2026) & comb.player_name.isin(d26.index)
comb.loc[m26, "draft_year"] = 2026
comb.loc[m26, "draft_round"] = comb.loc[m26, "player_name"].map(d26["round"])
comb.loc[m26, "draft_ovr"] = comb.loc[m26, "player_name"].map(d26["pick"])
print("2026 prospects matched to a draft pick:", int(m26.sum()), "of", int((comb.season == 2026).sum()))
df = comb.merge(d, on=["draft_year", "draft_ovr"], how="left")
# NFL roster position (nflverse players file) resolves generic combine labels such as OL / DL / DB
players = pd.read_csv("data/players.csv", low_memory=False)[["pfr_id", "position"]].dropna().drop_duplicates("pfr_id")
df = df.merge(players.rename(columns={"position": "nfl_pos"}), on="pfr_id", how="left")
# the source file occasionally hands one draft pick to two same-named prospects (e.g. two Derrick Johnsons
# in 2005). Keep the row whose combine position matches the drafted position; drop the namesake.
fam = {"QB": "O", "RB": "O", "FB": "O", "WR": "O", "TE": "O", "T": "OL", "OT": "OL", "G": "OL", "OG": "OL", "C": "OL", "OL": "OL",
       "DT": "F7", "NT": "F7", "DE": "F7", "DL": "F7", "EDGE": "F7", "LB": "F7", "OLB": "F7", "ILB": "F7",
       "CB": "DB", "DB": "DB", "S": "DB", "SAF": "DB", "FS": "DB", "SS": "DB"}
dup = df.draft_ovr.notna() & df.duplicated(["draft_year", "draft_ovr"], keep=False)
df["_match"] = (df.pos.map(fam) == df.draft_pos.map(fam)).astype(int)
keep = df[dup].sort_values("_match", ascending=False).drop_duplicates(["draft_year", "draft_ovr"]).index
print("namesake rows dropped:", int(dup.sum()) - len(keep))
df = df[~dup | df.index.isin(keep)].drop(columns="_match").copy()
df["drafted"] = df["draft_ovr"].notna()

# ---- position groups
def group(r):
    p, wt, ht = r["pos"], r["wt"], r["ht"]
    dp = r["nfl_pos"] if isinstance(r["nfl_pos"], str) else r["draft_pos"]
    if p in ("K", "P", "LS", "FB", "CB/WR"): return None
    if p == "QB": return "QB"
    if p == "RB": return "RB"
    if p == "WR": return "WR"
    if p == "TE": return "TE"
    if p == "OT": return "OT"
    if p in ("OG", "G", "C"): return "IOL"
    if p == "OL":
        if dp in ("T", "OT"): return "OT"
        if dp in ("G", "C"): return "IOL"
        return "OT" if ht >= 77 else "IOL"
    if p == "DT": return "DT"
    if p in ("DE", "EDGE"): return "EDGE"
    if p == "DL":
        if dp in ("DT", "NT"): return "DT"
        if dp in ("DE", "OLB", "LB"): return "EDGE"
        return "DT" if wt >= 285 else "EDGE"
    if p == "OLB": return "EDGE" if wt >= 245 else "LB"
    if p in ("ILB", "LB"): return "LB"
    if p == "CB": return "CB"
    if p in ("S", "SAF"): return "S"
    if p == "DB":
        if dp == "CB": return "CB"
        if dp in ("S", "SAF", "FS", "SS"): return "S"
        return "S" if wt >= 200 else "CB"
    return None
df["grp"] = df.apply(group, axis=1)
df = df[df.grp.notna()].copy()

drills = ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
df["n_drills"] = df[drills].notna().sum(axis=1)
n0 = len(df)
# inclusion: measured height, weight and a timed 40 (other drills are imputed when skipped)
df = df[df.forty.notna() & df.ht.notna() & df.wt.notna()].copy()
print("position players:", n0, "-> modelling sample:", len(df))
print(df.grp.value_counts().to_string())
print("drafted share:", df.drafted.mean().round(3))

# franchise codes (fold relocations)
fr = {"STL": "LAR", "SDG": "LAC", "OAK": "LVR"}
df["franchise"] = df["team"].replace(fr)
df.to_csv("out/prepared.csv", index=False)
draft.to_csv("out/draft_all.csv", index=False)
