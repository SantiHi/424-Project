"""04_analysis.py - join unicorn flags to draft value; all statistics and LaTeX tables used in the paper."""
import pandas as pd, numpy as np, json
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

rng = np.random.default_rng(42)
s = pd.read_csv("out/scored.csv")
v = pd.read_csv("out/draft_valued.csv")
vv = v[["season", "pick", "exp_av", "sd_av", "voe", "z", "ratio", "grade", "vgrp"]].rename(columns={"season": "draft_year", "pick": "draft_ovr"})
df = s.merge(vv, on=["draft_year", "draft_ovr"], how="left")
df["grade"] = df.grade.fillna("Undrafted")
df["w_av"] = df.w_av.fillna(0)
df.to_csv("out/master.csv", index=False)
G = df[df.grade.isin(["Bust", "Good value", "Steal"])].copy()        # graded = drafted 2000-2022
R = {}
order = ["Freak", "Odd build", "Red flag", "Typical"]

def boot_ci(x, n=4000):
    x = np.asarray(x); m = rng.choice(x, (n, len(x))).mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

# ---------- census
R["n_model"] = len(df); R["n_unicorn"] = int(df.unicorn.sum())
R["type_counts"] = df.utype.value_counts().to_dict()
R["drafted_share"] = df[df.season <= 2025].groupby("utype").drafted.mean().round(3).to_dict()
R["median_pick"] = df[df.drafted].groupby("utype").draft_ovr.median().to_dict()
R["r1_share"] = df[df.drafted].groupby("utype").apply(lambda x: (x.draft_round == 1).mean()).round(3).to_dict()
R["n_graded"] = len(G); R["graded_type_counts"] = G.utype.value_counts().to_dict()
R["grade_dist_all_drafted"] = v[v.season <= 2022].grade.value_counts(normalize=True).round(3).to_dict()

# logistic: P(drafted) and pick vs athleticism / anomaly
import statsmodels.formula.api as smf
dd = df[df.season <= 2025].copy(); dd["dr"] = dd.drafted.astype(int)
lg = smf.logit("dr ~ ath_z + cons_pct + C(grp)", dd).fit(disp=0)
R["logit_drafted"] = {k: [float(lg.params[k]), float(lg.bse[k]), float(lg.pvalues[k])] for k in ["ath_z", "cons_pct"]}
dk = dd[dd.drafted].copy(); dk["lp"] = np.log(dk.draft_ovr)
ol = smf.ols("lp ~ ath_z + cons_pct + C(grp)", dk).fit(cov_type="HC1")
R["ols_logpick"] = {k: [float(ol.params[k]), float(ol.bse[k]), float(ol.pvalues[k])] for k in ["ath_z", "cons_pct"]}

# ---------- value by type
rows = []
for t in order:
    x = G[G.utype == t]
    gd = x.grade.value_counts(normalize=True)
    ci = boot_ci(x.z)
    rows.append(dict(type=t, n=len(x), med_pick=x.draft_ovr.median(), mean_av=x.w_av.mean(), mean_exp=x.exp_av.mean(),
                     mean_voe=x.voe.mean(), mean_z=x.z.mean(), lo=ci[0], hi=ci[1],
                     bust=gd.get("Bust", 0), good=gd.get("Good value", 0), steal=gd.get("Steal", 0)))
x = G[G.unicorn]; gd = x.grade.value_counts(normalize=True); ci = boot_ci(x.z)
rows.append(dict(type="All unicorns", n=len(x), med_pick=x.draft_ovr.median(), mean_av=x.w_av.mean(), mean_exp=x.exp_av.mean(),
                 mean_voe=x.voe.mean(), mean_z=x.z.mean(), lo=ci[0], hi=ci[1], bust=gd.get("Bust", 0), good=gd.get("Good value", 0), steal=gd.get("Steal", 0)))
VT = pd.DataFrame(rows); VT.to_csv("out/value_by_type.csv", index=False); R["value_by_type"] = rows
typ = G[G.utype == "Typical"]
tests = {}
for t in ["Freak", "Odd build", "Red flag"]:
    x = G[G.utype == t]
    tt = stats.ttest_ind(x.z, typ.z, equal_var=False); mw = stats.mannwhitneyu(x.z, typ.z)
    ct = pd.crosstab(G[G.utype.isin([t, "Typical"])].utype, G[G.utype.isin([t, "Typical"])].grade)
    chi = stats.chi2_contingency(ct)
    tests[t] = dict(t=float(tt.statistic), p_t=float(tt.pvalue), p_mw=float(mw.pvalue), chi2=float(chi[0]), p_chi=float(chi[1]))
R["tests"] = tests
# regression with controls
G["lp"] = np.log(G.draft_ovr)
m1 = smf.ols("z ~ C(utype, Treatment('Typical')) + lp + C(grp) + C(draft_year)", G).fit(cov_type="HC1")
R["ols_z"] = {k.split("[T.")[1][:-1]: [float(m1.params[k]), float(m1.bse[k]), float(m1.pvalues[k])] for k in m1.params.index if "utype" in k}
m2 = smf.ols("z ~ ath_z + cons_pct + lp + C(grp) + C(draft_year)", G).fit(cov_type="HC1")
R["ols_z_cont"] = {k: [float(m2.params[k]), float(m2.bse[k]), float(m2.pvalues[k])] for k in ["ath_z", "cons_pct"]}
m3 = smf.ols("z ~ ath_z * cons_pct + lp + C(grp) + C(draft_year)", G).fit(cov_type="HC1")
R["ols_z_inter"] = {k: [float(m3.params[k]), float(m3.bse[k]), float(m3.pvalues[k])] for k in ["ath_z", "cons_pct", "ath_z:cons_pct"]}

# by round bucket
G["rb"] = pd.cut(G.draft_round, [0, 1, 3, 7], labels=["Round 1", "Rounds 2-3", "Rounds 4-7"])
rb = G.groupby(["rb", "utype"], observed=True).agg(n=("z", "size"), mean_z=("z", "mean"), voe=("voe", "mean"),
        bust=("grade", lambda g: (g == "Bust").mean()), steal=("grade", lambda g: (g == "Steal").mean())).reset_index()
rb.to_csv("out/value_by_round_type.csv", index=False); R["by_round"] = rb.round(3).to_dict("records")

# athleticism deciles and anomaly deciles
G["ath_dec"] = pd.qcut(G.ath_z, 10, labels=False) + 1
G["anom_dec"] = pd.qcut(G.cons_pct, 10, labels=False) + 1
G.groupby("ath_dec").agg(mean_z=("z", "mean"), se=("z", "sem"), steal=("grade", lambda g: (g == "Steal").mean()), bust=("grade", lambda g: (g == "Bust").mean())).to_csv("out/ath_deciles.csv")
G.groupby("anom_dec").agg(mean_z=("z", "mean"), se=("z", "sem"), steal=("grade", lambda g: (g == "Steal").mean()), bust=("grade", lambda g: (g == "Bust").mean())).to_csv("out/anom_deciles.csv")
R["corr_ath_z"] = float(stats.spearmanr(G.ath_z, G.z)[0]); R["corr_anom_z"] = float(stats.spearmanr(G.cons_pct, G.z)[0])

# by position (freaks vs typical)
bp = G.groupby(["grp", "utype"]).agg(n=("z", "size"), mean_z=("z", "mean"), steal=("grade", lambda g: (g == "Steal").mean()), bust=("grade", lambda g: (g == "Bust").mean())).reset_index()
bp.to_csv("out/value_by_pos_type.csv", index=False)
up = G[G.unicorn].groupby("grp").agg(n=("z", "size"), mean_z=("z", "mean"), steal=("grade", lambda g: (g == "Steal").mean()), bust=("grade", lambda g: (g == "Bust").mean()))
up["typ_z"] = typ.groupby("grp").z.mean(); up.to_csv("out/unicorn_by_pos.csv"); R["unicorn_by_pos"] = up.round(3).reset_index().to_dict("records")

# ---------- sensitivity to the contamination threshold and to the detector
sens = []
for lab, mask in [("Consensus top 2%", G.cons_pct > .98), ("Consensus top 5%", G.cons_pct > .95), ("Consensus top 10%", G.cons_pct > .90),
                  ("Isolation Forest only (5%)", G.p_if > .975), ("LOF only (5%)", G.p_lof > .975), ("Mahalanobis only (5%)", G.p_md > .975),
                  ("Full-workout view only (5%)", G.cons_full > .95), ("Size-speed view only (5%)", G.cons_core > .95)]:
    for t, m in [("Freak", mask & (G.ath_z >= .75)), ("Odd build", mask & (G.ath_z.abs() < .75)), ("Red flag", mask & (G.ath_z <= -.75))]:
        x = G[m]; rest = G[~mask]
        sens.append(dict(defn=lab, type=t, n=len(x), mean_z=x.z.mean(), steal=(x.grade == "Steal").mean(), bust=(x.grade == "Bust").mean(),
                         p=float(stats.ttest_ind(x.z, rest.z, equal_var=False).pvalue)))
pd.DataFrame(sens).to_csv("out/sensitivity.csv", index=False); R["sensitivity"] = sens
# complete-workout robustness: only players with >= 5 measured drills
cw = G[G.n_drills >= 5]
R["complete_case"] = cw.groupby("utype").agg(n=("z", "size"), mean_z=("z", "mean")).round(3).reset_index().to_dict("records")
# excluding still-young classes
R["mature_only"] = G[G.draft_year <= 2018].groupby("utype").agg(n=("z", "size"), mean_z=("z", "mean"), steal=("grade", lambda g: (g == "Steal").mean()), bust=("grade", lambda g: (g == "Bust").mean())).round(3).reset_index().to_dict("records")

# ---------- archetypes among unicorns (k-means on the oriented feature profile)
U = df[df.unicorn].copy()
U["explosive"] = U[["adj_vertical", "adj_broad_jump"]].mean(axis=1)
U["agility"] = U[["adj_cone", "adj_shuttle"]].mean(axis=1)
U["strength"] = U.adj_bench.fillna(0)
cf = ["z_ht", "z_wt", "adj_forty", "explosive", "agility", "strength"]
Xc = U[cf].clip(-5, 5).values
sil = {}
for k in range(3, 9):
    km = KMeans(k, n_init=25, random_state=42).fit(Xc); sil[k] = float(silhouette_score(Xc, km.labels_))
R["silhouette"] = sil
K = 5
km = KMeans(K, n_init=50, random_state=42).fit(Xc); U["cluster"] = km.labels_
cen = pd.DataFrame(km.cluster_centers_, columns=cf)
cen["n"] = U.cluster.value_counts().sort_index().values
UG = U[U.grade.isin(["Bust", "Good value", "Steal"])]
cen["n_graded"] = UG.groupby("cluster").size(); cen["mean_z"] = UG.groupby("cluster").z.mean()
cen["steal"] = UG.groupby("cluster").grade.apply(lambda g: (g == "Steal").mean()); cen["bust"] = UG.groupby("cluster").grade.apply(lambda g: (g == "Bust").mean())
cen["ath_z"] = U.groupby("cluster").ath_z.mean()
cen["examples"] = [", ".join(U[(U.cluster == c)].sort_values("w_av", ascending=False).player_name.head(5)) for c in range(K)]
cen["top_pos"] = [", ".join(U[U.cluster == c].grp.value_counts().head(3).index) for c in range(K)]
cen.to_csv("out/archetypes.csv"); U[["player_name", "season", "grp", "cluster"]].to_csv("out/unicorn_clusters.csv", index=False)
print(cen.round(2).to_string()); print(sil)

# ---------- teams
names = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
         "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GNB": "Green Bay Packers",
         "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KAN": "Kansas City Chiefs", "LAC": "Los Angeles Chargers", "LAR": "Los Angeles Rams",
         "LVR": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NWE": "New England Patriots", "NOR": "New Orleans Saints", "NYG": "New York Giants",
         "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SEA": "Seattle Seahawks", "SFO": "San Francisco 49ers", "TAM": "Tampa Bay Buccaneers",
         "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}
GU = G[G.unicorn]
def best(x): r = x.loc[x.z.idxmax()]; return f"{r.player_name} ({int(r.draft_year)}, \\#{int(r.draft_ovr)})"
def worst(x): r = x.loc[x.z.idxmin()]; return f"{r.player_name} ({int(r.draft_year)}, \\#{int(r.draft_ovr)})"
T = GU.groupby("franchise").agg(n=("z", "size"), freaks=("utype", lambda u: (u == "Freak").sum()), steals=("grade", lambda g: (g == "Steal").sum()),
                                busts=("grade", lambda g: (g == "Bust").sum()), voe=("voe", "sum"), mean_z=("z", "mean")).reset_index()
T["best"] = T.franchise.map(GU.groupby("franchise").apply(best)); T["worst"] = T.franchise.map(GU.groupby("franchise").apply(worst))
allp = G.groupby("franchise").agg(n_all=("z", "size"), ath_all=("ath_z", "mean"), z_all=("z", "mean"), anom_all=("cons_pct", "mean")).reset_index()
T = T.merge(allp, on="franchise"); T["share"] = T.n / T.n_all; T["team"] = T.franchise.map(names)
T = T.sort_values("voe", ascending=False); T.to_csv("out/teams.csv", index=False)
R["team_corr_ath_z"] = [float(c) for c in stats.pearsonr(T.ath_all, T.z_all)]
R["team_corr_share_z"] = [float(c) for c in stats.pearsonr(T.share, T.z_all)]
# is team-level unicorn skill distinguishable from luck? permutation of team labels among unicorn picks
obs = GU.groupby("franchise").z.mean().std()
perm = [GU.assign(f=rng.permutation(GU.franchise.values)).groupby("f").z.mean().std() for _ in range(3000)]
R["team_perm"] = dict(obs_sd=float(obs), perm_mean=float(np.mean(perm)), p=float((np.array(perm) >= obs).mean()))
print(T[["franchise", "n", "freaks", "steals", "busts", "voe", "mean_z", "share", "ath_all", "z_all"]].round(2).to_string())

# ---------- LaTeX tables
def esc(x): return str(x).replace("&", "\\&").replace("'", "'")
def tex(dfx, path, cols, fmt):
    with open(path, "w") as fh:
        for _, r in dfx.iterrows():
            fh.write(" & ".join(fmt[c](r[c]) if c in fmt else esc(r[c]) for c in cols) + " \\\\\n")
i0 = lambda x: f"{int(round(x))}"; f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:+.2f}"; pc = lambda x: f"{100*x:.0f}\\%"
htf = lambda x: f"{int(x)//12}-{int(x)%12}"; f40 = lambda x: f"{x:.2f}"
gcol = {"Steal": "\\steal", "Bust": "\\bust", "Good value": "\\good", "Too early": "\\early", "Undrafted": "\\textit{Undrafted}"}
# top drafted+graded freaks by consensus score
top = G[G.utype == "Freak"].sort_values("consensus", ascending=False).head(25)
tex(top, "paper/tables/top_freaks.tex", ["player_name", "grp", "draft_year", "ht", "wt", "forty", "ath_z", "draft_ovr", "franchise", "w_av", "exp_av", "z", "grade"],
    {"draft_year": i0, "ht": htf, "wt": i0, "forty": f40, "ath_z": f2, "draft_ovr": i0, "w_av": i0, "exp_av": f1, "z": f2, "grade": lambda g: gcol[g]})
for nm, sub in [("unicorn_steals", GU.sort_values("z", ascending=False).head(15)), ("unicorn_busts", GU.sort_values("z").head(15))]:
    tex(sub, f"paper/tables/{nm}.tex", ["player_name", "grp", "utype", "draft_year", "draft_ovr", "franchise", "w_av", "exp_av", "z"],
        {"draft_year": i0, "draft_ovr": i0, "w_av": i0, "exp_av": f1, "z": f2})
tex(T.sort_values("voe", ascending=False), "paper/tables/teams.tex", ["team", "n", "freaks", "steals", "busts", "voe", "mean_z", "best", "worst"],
    {"n": i0, "freaks": i0, "steals": i0, "busts": i0, "voe": lambda x: f"{x:+.0f}", "mean_z": f2})
tex(VT, "paper/tables/value_by_type.tex", ["type", "n", "med_pick", "mean_av", "mean_exp", "mean_voe", "mean_z", "lo", "hi", "bust", "good", "steal"],
    {"n": i0, "med_pick": i0, "mean_av": f1, "mean_exp": f1, "mean_voe": lambda x: f"{x:+.1f}", "mean_z": f2, "lo": f2, "hi": f2, "bust": pc, "good": pc, "steal": pc})
# recent, not yet gradeable unicorns
rec = df[df.unicorn & (df.season >= 2023) & (df.utype == "Freak")].sort_values(["season", "consensus"], ascending=[True, False])
tex(rec, "paper/tables/recent.tex", ["player_name", "grp", "season", "ht", "wt", "forty", "ath_z", "draft_ovr", "franchise"],
    {"season": i0, "ht": htf, "wt": i0, "forty": f40, "ath_z": f2, "draft_ovr": lambda x: "--" if pd.isna(x) else i0(x), "franchise": lambda x: "--" if pd.isna(x) else x})
df[df.unicorn].sort_values(["grp", "consensus"], ascending=[True, False])[["player_name", "season", "pos", "grp", "school", "ht", "wt", "forty", "bench", "vertical", "broad_jump", "cone", "shuttle",
    "ath_z", "cons_pct", "utype", "draft_year", "draft_round", "draft_ovr", "franchise", "w_av", "exp_av", "voe", "z", "grade"]].round(3).to_csv("out/all_unicorns.csv", index=False)
json.dump(R, open("out/results.json", "w"), indent=1, default=float)
print(VT.round(3).to_string()); print(json.dumps({k: R[k] for k in ["tests", "ols_z", "ols_z_cont", "ols_z_inter", "drafted_share", "median_pick", "r1_share", "logit_drafted", "ols_logpick", "team_perm", "team_corr_ath_z", "team_corr_share_z", "corr_ath_z", "corr_anom_z", "complete_case", "mature_only", "graded_type_counts"]}, indent=0, default=float))
print(rb.round(2).to_string()); print(pd.DataFrame(sens).round(3).to_string()); print(up.round(2))

# ---------- extras quoted in the text
X = {}
r1f, r1t = G[(G.draft_round == 1) & (G.utype == "Freak")], G[(G.draft_round == 1) & (G.utype == "Typical")]
X["r1_freak_t"] = [float(a) for a in stats.ttest_ind(r1f.z, r1t.z, equal_var=False)]
X["r1_freak_ci"] = boot_ci(r1f.z)
X["r1_fisher_bust"] = float(stats.fisher_exact([[(r1f.grade == "Bust").sum(), (r1f.grade != "Bust").sum()], [(r1t.grade == "Bust").sum(), (r1t.grade != "Bust").sum()]])[1])
X["r1_fisher_steal"] = float(stats.fisher_exact([[(r1f.grade == "Steal").sum(), (r1f.grade != "Steal").sum()], [(r1t.grade == "Steal").sum(), (r1t.grade != "Steal").sum()]])[1])
X["median_z"] = G.groupby("utype").z.median().round(3).to_dict()
X["redflag_ex_brady"] = float(G[(G.utype == "Red flag") & (G.player_name != "Tom Brady")].z.mean())
X["per_class"] = df.groupby("season").unicorn.sum().to_dict()
X["undrafted_freaks"] = df[(df.utype == "Freak") & ~df.drafted & (df.season <= 2025)].player_name.tolist()
X["n_classes"] = int(df.season.nunique()); X["n_drills_mean"] = float(df.n_drills.mean())
X["imputed_share"] = float(1 - df[["forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]].notna().values.mean())
X["flag_overlap"] = {"all3": int((df.flag_if & df.flag_lof & df.flag_md).sum()), "any": int((df.flag_if | df.flag_lof | df.flag_md).sum()),
                     "if": int(df.flag_if.sum()), "lof": int(df.flag_lof.sum()), "md": int(df.flag_md.sum())}
X["view_overlap"] = {"both": int(((df.cons_full > .95) & (df.cons_core > .95)).sum()), "full_only": int(((df.cons_full > .95) & (df.cons_core <= .95)).sum()), "core_only": int(((df.cons_full <= .95) & (df.cons_core > .95)).sum())}
X["spearman_full"] = df[["s_if_full", "s_lof_full", "s_md_full"]].corr(method="spearman").round(2).values.tolist()
X["grade_by_round"] = pd.crosstab(v[v.season <= 2022]["round"], v[v.season <= 2022].grade, normalize="index").round(3).to_dict("index")
X["n_valued"] = int((v.season <= 2022).sum())
json.dump(X, open("out/extras.json", "w"), indent=1, default=float)
print(json.dumps(X, default=float)[:3000])

# ---------- archetype names assigned by rule, so they survive any relabelling of clusters
A = cen.copy(); nm = {}
nm[int(A.adj_forty.idxmin())] = "Slow testers"
nm[int(A.z_wt.idxmax())] = "Mass outliers"
nm[int(A.z_wt.idxmin())] = "Miniatures"
rest = [i for i in A.index if i not in nm]
tall = max(rest, key=lambda i: A.z_ht[i]); nm[int(tall)] = "Size-speed titans"
nm[int([i for i in rest if i != tall][0])] = "Workout warriors"
json.dump(nm, open("out/archetype_names.json", "w"))
A["name"] = A.index.map(nm)
with open("paper/tables/archetypes.tex", "w") as fh:
    for i, r in A.sort_values("mean_z", ascending=False).iterrows():
        fh.write(f"{r['name']} & {int(r.n)} & {int(r.n_graded)} & {r.ath_z:+.2f} & {r.mean_z:+.2f} & {100*r.steal:.0f}\\% & {100*r.bust:.0f}\\% & {r.top_pos} & {r.examples} \\\\\n")
