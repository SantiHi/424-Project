"""06_tables.py - remaining LaTeX tables and the numbers macro file."""
import pandas as pd, numpy as np, json
df = pd.read_csv("out/master.csv"); R = json.load(open("out/results.json")); X = json.load(open("out/extras.json"))
G = df[df.grade.isin(["Bust", "Good value", "Steal"])]
drills = ["forty", "bench", "vertical", "broad_jump", "cone", "shuttle"]
order = ["QB", "RB", "WR", "TE", "OT", "IOL", "DT", "EDGE", "LB", "CB", "S"]
with open("paper/tables/sample.tex", "w") as fh:
    for g in order + ["All"]:
        s = df if g == "All" else df[df.grp == g]; gg = G if g == "All" else G[G.grp == g]
        cov = s[["bench", "vertical", "broad_jump", "cone", "shuttle"]].notna().mean() * 100
        row = [g, len(s), int(s.drafted.sum()), len(gg), int(s.unicorn.sum()), int((s.utype == "Freak").sum()), int((s.utype == "Odd build").sum()), int((s.utype == "Red flag").sum())] + [f"{c:.0f}" for c in cov]
        if g == "All": fh.write("\\midrule\n")
        fh.write(" & ".join(str(x) for x in row) + " \\\\\n")
S = pd.read_csv("out/sensitivity.csv")
with open("paper/tables/sensitivity.tex", "w") as fh:
    for d in S.defn.unique():
        r = S[S.defn == d].set_index("type")
        fh.write(d.replace("%", "\\%") + " & " + " & ".join(f"{int(r.n[t])} & {r.mean_z[t]:+.2f} & {100*r.steal[t]:.0f}\\% & {100*r.bust[t]:.0f}\\%" for t in ["Freak", "Odd build", "Red flag"]) + " \\\\\n")
B = pd.read_csv("out/value_by_round_type.csv")
with open("paper/tables/rounds.tex", "w") as fh:
    for rb in ["Round 1", "Rounds 2-3", "Rounds 4-7"]:
        for i, t in enumerate(["Freak", "Odd build", "Red flag", "Typical"]):
            r = B[(B.rb == rb) & (B.utype == t)]
            if len(r) == 0: continue
            r = r.iloc[0]
            fh.write(f"{rb.replace('-', '--') if i == 0 else ''} & {t} & {int(r.n)} & {r.voe:+.1f} & {r.mean_z:+.2f} & {100*r.bust:.0f}\\% & {100*r.steal:.0f}\\% \\\\\n")
        if rb != "Rounds 4-7": fh.write("\\addlinespace\n")
P = pd.read_csv("out/unicorn_by_pos.csv").sort_values("mean_z", ascending=False)
with open("paper/tables/positions.tex", "w") as fh:
    for _, r in P.iterrows(): fh.write(f"{r.grp} & {int(r.n)} & {r.mean_z:+.2f} & {r.typ_z:+.2f} & {100*r.steal:.0f}\\% & {100*r.bust:.0f}\\% \\\\\n")
def p(x): return "<0.001" if x < .001 else f"{x:.3f}"
with open("paper/tables/regression.tex", "w") as fh:
    for k in ["Freak", "Odd build", "Red flag"]:
        b, se, pv = R["ols_z"][k]; fh.write(f"{k} unicorn (vs.\\ typical) & {b:+.3f} & {se:.3f} & {p(pv)} & & & \\\\\n")
    for k, lab in [("ath_z", "Athleticism composite (per SD)"), ("cons_pct", "Anomaly percentile (0--1)")]:
        b, se, pv = R["ols_z_cont"][k]; fh.write(f"{lab} & & & & {b:+.3f} & {se:.3f} & {p(pv)} \\\\\n")
print(open("paper/tables/sample.tex").read())
c = ["player_name", "season", "grp", "ht", "wt", "forty", "vertical", "broad_jump", "bench", "cone", "shuttle", "ath_z", "cons_pct", "cons_full", "cons_core", "utype", "draft_ovr", "franchise", "w_av", "exp_av", "z", "grade"]
names = ["Calvin Johnson", "Tom Brady", "D.K. Metcalf", "JaMarcus Russell", "Tariq Woolen", "Calais Campbell", "Jordan Davis", "Patrick Peterson", "Derrick Henry", "Mekhi Becton", "Vernon Davis", "Micah Parsons", "J.J. Watt", "Mike Williams", "Darren Sproles", "Vincent Jackson", "Taylor Mays", "Obi Melifonwu", "Aaron Donald", "Myles Garrett", "Saquon Barkley", "Julio Jones", "Anthony Richardson", "Michael Vick", "Fletcher Cox", "Kevin Williams","Mario Williams","Dwight Freeney","Jason Peters","Jared Lorenzen"]
pd.set_option("display.width", 300)
print(df[df.player_name.isin(names)][c].round(2).to_string())
r1 = G[(G.draft_round == 1) & (G.utype == "Freak")].sort_values("z", ascending=False)
print(r1[["player_name", "draft_year", "grp", "draft_ovr", "franchise", "w_av", "exp_av", "z", "grade"]].round(2).to_string())
print(len(df[df.unicorn & df.drafted & (df.season<=2025)]), len(df[df.unicorn & (df.season<=2025)]), df[df.season<=2025].shape)
print(G[G.unicorn].grade.value_counts().to_dict(), G[G.utype=='Freak'].grade.value_counts().to_dict())

# compact steal / bust lists for the side-by-side tables
ab = {"Freak": "F", "Odd build": "O", "Red flag": "R"}
GU = G[G.unicorn]
for nm, sub in [("unicorn_steals_short", GU.sort_values("z", ascending=False).head(15)), ("unicorn_busts_short", GU.sort_values("z").head(15))]:
    with open(f"paper/tables/{nm}.tex", "w") as fh:
        for _, r in sub.iterrows():
            fh.write(f"{r.player_name} & {r.grp} & {ab[r.utype]} & '{int(r.draft_year) % 100:02d} & {int(r.draft_ovr)} & {r.franchise} & {int(r.w_av)} & {r.exp_av:.0f} & {r.z:+.1f} \\\\\n")
