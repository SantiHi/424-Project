"""05_figures.py - every figure in the paper (vector PDF, scatter layers rasterised)."""
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy import stats

rng = np.random.default_rng(7)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelsize": 8.5, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "pdf.fonttype": 42})
C = {"Freak": "#E69F00", "Odd build": "#7B5EA7", "Red flag": "#D1495B", "Typical": "#B8BDC4"}
GC = {"Steal": "#2A9D8F", "Good value": "#8FA3B8", "Bust": "#D1495B"}
INK = "#22303C"
order = ["Freak", "Odd build", "Red flag", "Typical"]
df = pd.read_csv("out/master.csv"); G = df[df.grade.isin(["Bust", "Good value", "Steal"])].copy()
v = pd.read_csv("out/draft_valued.csv"); curve = pd.read_csv("out/value_curve.csv")
def save(name): plt.savefig(f"paper/figures/{name}.pdf", dpi=300); plt.close()
def ci(x, n=3000):
    x = np.asarray(x); m = rng.choice(x, (n, len(x))).mean(axis=1); return np.percentile(m, 2.5), np.percentile(m, 97.5)
def label(ax, sub, xc, yc, fs=6.5, **kw):
    T = [ax.text(r[xc], r[yc], r.player_name, fontsize=fs, color=INK) for _, r in sub.iterrows()]
    adjust_text(T, ax=ax, arrowprops=dict(arrowstyle="-", color="#777", lw=0.4), **kw)

# ---- 1 size-speed maps
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.5))
for ax, g in zip(axs, ["WR", "EDGE", "CB"]):
    s = df[df.grp == g]
    ax.scatter(s[~s.unicorn].wt, s[~s.unicorn].forty, s=5, c=C["Typical"], alpha=.55, lw=0, rasterized=True)
    for t in order[:3]:
        u = s[s.utype == t]; ax.scatter(u.wt, u.forty, s=20, c=C[t], edgecolor="white", lw=.4, label=t, zorder=3)
    lab = s[s.unicorn].sort_values("w_av", ascending=False).head(7)
    lab = pd.concat([lab, s[s.unicorn & (s.cons_pct > .995)]]).drop_duplicates("player_name").head(9)
    ax.invert_yaxis(); label(ax, lab, "wt", "forty"); ax.invert_yaxis()
    ax.invert_yaxis(); ax.set_title(g); ax.set_xlabel("Weight (lb)"); ax.set_ylabel("40-yard dash (s)"); ax.yaxis.set_major_formatter("{x:.2f}")
axs[0].legend(loc="lower left", fontsize=7, markerscale=1.1)
plt.tight_layout(); save("fig_sizespeed")

# ---- 2 PCA maps
fig, axs = plt.subplots(2, 3, figsize=(10.2, 5.7))
for ax, g in zip(axs.ravel(), ["WR", "RB", "TE", "QB", "DT", "S"]):
    s = df[df.grp == g]
    ax.scatter(s[~s.unicorn].pc1, s[~s.unicorn].pc2, s=5, c=C["Typical"], alpha=.55, lw=0, rasterized=True)
    for t in order[:3]:
        u = s[s.utype == t]; ax.scatter(u.pc1, u.pc2, s=20, c=C[t], edgecolor="white", lw=.4, label=t, zorder=3)
    lab = s[s.unicorn].sort_values("w_av", ascending=False).head(5)
    lab = pd.concat([lab, s[s.unicorn].sort_values("consensus", ascending=False).head(3)]).drop_duplicates("player_name")
    label(ax, lab, "pc1", "pc2"); ax.set_title(g); ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
axs[0, 0].legend(fontsize=7, loc="best")
plt.tight_layout(); save("fig_pca")

# ---- 3 detector agreement
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.3))
pairs = [("p_if_full", "p_lof_full", "Isolation Forest", "Local Outlier Factor"), ("p_if_full", "p_md_full", "Isolation Forest", "Robust Mahalanobis"), ("p_lof_full", "p_md_full", "Local Outlier Factor", "Robust Mahalanobis")]
for ax, (a, b, la, lb) in zip(axs, pairs):
    both = (df[a] > .95) & (df[b] > .95); one = ((df[a] > .95) | (df[b] > .95)) & ~both
    ax.scatter(df[a][~both & ~one], df[b][~both & ~one], s=2, c=C["Typical"], alpha=.4, lw=0, rasterized=True)
    ax.scatter(df[a][one], df[b][one], s=5, c="#7B5EA7", lw=0, rasterized=True, label=f"flagged by one (n={one.sum()})")
    ax.scatter(df[a][both], df[b][both], s=7, c="#E69F00", lw=0, rasterized=True, label=f"flagged by both (n={both.sum()})")
    ax.axvline(.95, color=INK, lw=.6, ls="--"); ax.axhline(.95, color=INK, lw=.6, ls="--")
    rho = stats.spearmanr(df[a], df[b])[0]
    ax.set_title(f"Spearman ρ = {rho:.2f}"); ax.set_xlabel(la + " percentile"); ax.set_ylabel(lb + " percentile"); ax.legend(fontsize=6.5, loc="lower right", markerscale=1.5)
plt.tight_layout(); save("fig_detectors")

# ---- 4 value curve
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.5), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]; m = v[v.season <= 2015]
ax.scatter(m.pick, m.w_av, s=4, c=C["Typical"], alpha=.5, lw=0, rasterized=True, label="Drafted player, 2000–2015 classes")
ax.fill_between(curve.pick, np.clip(curve.f - curve.sd, 0, None), curve.f + curve.sd, color="#2A6F97", alpha=.15, lw=0, label="± 1 SD")
ax.plot(curve.pick, curve.f, color="#2A6F97", lw=2, label="Expected career AV, f(pick)")
for rr in range(1, 7): ax.axvline(32.5 * rr + (rr > 2) * 3 * (rr - 2), color="#cfcfcf", lw=.5, ls=":")
ax.set_xlim(0, 262); ax.set_ylim(0, 190); ax.set_xlabel("Overall pick"); ax.set_ylabel("Career weighted AV"); ax.legend(fontsize=7); ax.set_title("Slot value curve")
ax = axs[1]; g = pd.read_csv("out/class_factor.csv")
ax.bar(g.season, g.g, color=np.where(g.season <= 2022, "#2A6F97", "#c9d3dc"), width=.75)
ax.axhline(1, color=INK, lw=.6); ax.set_xlabel("Draft class"); ax.set_ylabel("Class factor g"); ax.set_title("Career-length adjustment")
ax.text(2023.4, .55, "too early\nto grade", fontsize=6.5, color="#667", ha="left")
plt.tight_layout(); save("fig_valuecurve")

# ---- 5 draft premium
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.3))
d25 = df[df.season <= 2025]
ax = axs[0]; sh = d25.groupby("utype").drafted.mean().reindex(order)
b = ax.bar(order, sh.values * 100, color=[C[t] for t in order]); ax.bar_label(b, fmt="%.0f%%", fontsize=7.5); ax.set_ylabel("Share drafted (%)"); ax.set_title("Who gets drafted"); ax.set_ylim(0, 100)
ax = axs[1]
for t in order:
    x = np.sort(d25[d25.drafted & (d25.utype == t)].draft_ovr.values); ax.step(x, np.arange(1, len(x) + 1) / len(x), color=C[t] if t != "Typical" else "#6b7280", lw=1.6, label=t, where="post")
ax.set_xlabel("Overall pick"); ax.set_ylabel("Cumulative share of drafted"); ax.set_title("How early they go"); ax.legend(fontsize=7, loc="lower right")
ax = axs[2]; dd = d25.copy(); dd["dec"] = pd.qcut(dd.ath_z, 10, labels=False) + 1
q = dd.groupby("dec").agg(dr=("drafted", "mean")); q["r1"] = dd[dd.drafted].groupby("dec").apply(lambda x: (x.draft_round <= 2).mean())
ax.plot(q.index, q.dr * 100, marker="o", ms=4, color="#2A6F97", label="Drafted"); ax.plot(q.index, q.r1 * 100, marker="s", ms=4, color="#E69F00", label="Rounds 1–2, if drafted")
ax.set_xlabel("Athleticism decile (within position)"); ax.set_ylabel("%"); ax.set_title("The market pays for athleticism"); ax.legend(fontsize=7); ax.set_xticks(range(1, 11))
plt.tight_layout(); save("fig_premium")

# ---- 6 grades by type
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]; left = np.zeros(4)
sh = pd.crosstab(G.utype, G.grade, normalize="index").reindex(order)[["Bust", "Good value", "Steal"]]
for gr in ["Bust", "Good value", "Steal"]:
    bars = ax.barh(order, sh[gr] * 100, left=left, color=GC[gr], label=gr, height=.62)
    for yi, (w, l) in enumerate(zip(sh[gr] * 100, left)): ax.text(l + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
    left += sh[gr].values * 100
ax.set_yticks(range(4)); ax.set_yticklabels([f"{t}\n(n={int((G.utype == t).sum())})" for t in order]); ax.invert_yaxis(); ax.set_xlim(0, 100)
ax.set_xlabel("Share of drafted players, 2000–2022 classes (%)"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.17), fontsize=7.5); ax.grid(False)
ax = axs[1]
for i, t in enumerate(order):
    x = G[G.utype == t].z; lo, hi = ci(x); ax.errorbar(x.mean(), i, xerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", color=C[t] if t != "Typical" else "#6b7280", capsize=3, ms=6, lw=1.5)
ax.axvline(0, color=INK, lw=.6); ax.set_yticks(range(4)); ax.set_yticklabels(order); ax.invert_yaxis(); ax.set_xlabel("Mean value over slot (z), 95% bootstrap CI"); ax.set_title("Value relative to draft slot")
plt.tight_layout(); save("fig_grades")

# ---- 7 unicorn pick-vs-AV map
fig, ax = plt.subplots(figsize=(10.2, 4.7))
U = G[G.unicorn]
ax.plot(curve.pick, curve.f, color=INK, lw=1.3, ls="--", label="Slot expectation (position-neutral)", zorder=2)
mk = {"Freak": "o", "Odd build": "D", "Red flag": "v"}
for t in order[:3]:
    for gr in ["Bust", "Good value", "Steal"]:
        s = U[(U.utype == t) & (U.grade == gr)]
        ax.scatter(s.draft_ovr, s.w_av, s=34, marker=mk[t], c=GC[gr], edgecolor="white", lw=.5, zorder=3)
ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 32, 64, 100, 150, 250]); ax.set_xticklabels([1, 2, 5, 10, 20, 32, 64, 100, 150, 250]); ax.set_xlim(.85, 290); ax.set_ylim(-4, 128)
names = pd.concat([U.sort_values("z", ascending=False).head(13), U.sort_values("z").head(7), U.sort_values("w_av", ascending=False).head(10)]).drop_duplicates("player_name")
names = names[names.player_name != "Tom Brady"]
label(ax, names, "draft_ovr", "w_av", fs=6.8)
ax.annotate("Tom Brady (2000, #199): 184 AV, off the chart", xy=(199, 127), xytext=(60, 121), fontsize=7, color=INK, arrowprops=dict(arrowstyle="->", color="#2A9D8F", lw=1))
ax.set_xlabel("Overall pick (log scale)"); ax.set_ylabel("Career weighted AV")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=mk[t], ls="", color="#555", label=t, ms=6) for t in mk] + [Line2D([], [], marker="s", ls="", color=GC[g], label=g, ms=7) for g in GC] + [Line2D([], [], color=INK, ls="--", label="Slot expectation")]
ax.legend(handles=h, ncol=1, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
plt.tight_layout(); save("fig_unicorn_map")

# ---- 8 by round
G["rb"] = pd.cut(G.draft_round, [0, 1, 3, 7], labels=["Round 1", "Rounds 2–3", "Rounds 4–7"])
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.3))
ax = axs[0]; w = .2
for i, t in enumerate(order):
    for j, r in enumerate(["Round 1", "Rounds 2–3", "Rounds 4–7"]):
        x = G[(G.utype == t) & (G.rb == r)].z
        if len(x) < 8: continue
        lo, hi = ci(x); ax.errorbar(j + (i - 1.5) * w, x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", color=C[t] if t != "Typical" else "#6b7280", capsize=2.5, ms=5, lw=1.3, label=t if j == 1 else None)
        ax.text(j + (i - 1.5) * w, max(lo, -.72) - .06, f"{len(x)}", ha="center", va="top", fontsize=6, color="#667")
ax.axhline(0, color=INK, lw=.6); ax.set_xticks(range(3)); ax.set_xticklabels(["Round 1", "Rounds 2–3", "Rounds 4–7"]); ax.set_ylabel("Mean value over slot (z)"); ax.set_title("Value over slot"); ax.set_ylim(-.95, 1.35); ax.legend(fontsize=6.5, ncol=1, loc="upper right")
for ax, gr, ttl in [(axs[1], "Bust", "Bust rate"), (axs[2], "Steal", "Steal rate")]:
    for i, t in enumerate(order):
        vals = [((G[(G.utype == t) & (G.rb == r)].grade == gr).mean() * 100 if ((G.utype == t) & (G.rb == r)).sum() >= 8 else np.nan) for r in ["Round 1", "Rounds 2–3", "Rounds 4–7"]]
        ax.bar(np.arange(3) + (i - 1.5) * w, vals, w * .92, color=C[t], label=t)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Round 1", "Rounds 2–3", "Rounds 4–7"]); ax.set_ylabel("%"); ax.set_title(ttl)
plt.tight_layout(); save("fig_rounds")

# ---- 9 deciles
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.2))
for ax, f, xl, col in [(axs[0], "out/ath_deciles.csv", "Athleticism decile (1 = worst tester, 10 = best)", "#E69F00"), (axs[1], "out/anom_deciles.csv", "Anomaly-score decile (10 = most unusual profile)", "#7B5EA7")]:
    q = pd.read_csv(f); ax.errorbar(q.iloc[:, 0], q.mean_z, yerr=1.96 * q.se, fmt="o-", color=col, capsize=3, ms=5, lw=1.4)
    ax.axhline(0, color=INK, lw=.6); ax.set_xticks(range(1, 11)); ax.set_xlabel(xl); ax.set_ylabel("Mean value over slot (z)")
axs[0].set_title("Direction matters a little"); axs[1].set_title("Rarity alone does not")
plt.tight_layout(); save("fig_deciles")

# ---- 10 positions
fig, ax = plt.subplots(figsize=(6.6, 3.6))
U = G[G.unicorn]; pos = U.groupby("grp").z.mean().sort_values().index
for i, p in enumerate(pos):
    x = U[U.grp == p].z; lo, hi = ci(x); col = "#2A9D8F" if x.mean() > 0 else "#D1495B"
    ax.errorbar(x.mean(), i, xerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", color=col, capsize=3, ms=6, lw=1.4)
    ax.scatter(G[(G.grp == p) & ~G.unicorn].z.mean(), i, marker="|", s=90, color=INK, zorder=4)
ax.set_yticks(range(len(pos))); ax.set_yticklabels([f"{p} (n={int((U.grp == p).sum())})" for p in pos]); ax.axvline(0, color=INK, lw=.6)
ax.set_xlim(-1.05, 1.75); ax.set_ylim(-.6, len(pos) - .1); ax.annotate("QB interval runs to +3.6\n(one pick: Tom Brady)", xy=(1.74, list(pos).index("QB")), xytext=(0.62, list(pos).index("QB") + 0.55), fontsize=7, color="#445", arrowprops=dict(arrowstyle="->", color="#777", lw=.7))
ax.set_xlabel("Mean value over slot (z) of drafted unicorns, 95% CI;  | = typical players at the position")
plt.tight_layout(); save("fig_positions")

# ---- 11 teams
T = pd.read_csv("out/teams.csv").sort_values("voe")
fig, axs = plt.subplots(1, 2, figsize=(10.2, 5.5), gridspec_kw={"width_ratios": [1.55, 1]})
ax = axs[0]; ax.barh(T.team, T.voe, color=np.where(T.voe > 0, "#2A9D8F", "#D1495B"), height=.7)
for y, (val, n, st, bu) in enumerate(zip(T.voe, T.n, T.steals, T.busts)):
    ax.text(val + (3 if val >= 0 else -3), y, f"{n} picks · {st}S / {bu}B", va="center", ha="left" if val >= 0 else "right", fontsize=6.3, color="#445")
ax.axvline(0, color=INK, lw=.6); ax.set_xlabel("Total career AV over slot expectation from drafted unicorns (2000–2022)"); ax.set_xlim(-215, 190); ax.tick_params(axis="y", labelsize=6.8); ax.grid(axis="y", visible=False)
ax = axs[1]; ax.scatter(T.ath_all, T.z_all, s=22, c="#2A6F97", zorder=3)
Tx = [ax.text(r.ath_all, r.z_all, r.franchise, fontsize=6.3, color=INK) for _, r in T.iterrows()]; adjust_text(Tx, ax=ax)
rr, pp = stats.pearsonr(T.ath_all, T.z_all); ax.axhline(0, color=INK, lw=.6)
ax.set_xlabel("Mean athleticism z of all the team's combine-tested picks"); ax.set_ylabel("Mean value over slot (z), all picks"); ax.set_title(f"Drafting athletes ≠ drafting well (r = {rr:.2f}, p = {pp:.2f})")
plt.tight_layout(); save("fig_teams")

# ---- 12 archetypes
A = pd.read_csv("out/archetypes.csv", index_col=0)
names = json.load(open("out/archetype_names.json"))
cf = ["z_ht", "z_wt", "adj_forty", "explosive", "agility", "strength"]
fig, ax = plt.subplots(figsize=(7.4, 3.0))
im = ax.imshow(A[cf].values, cmap="RdBu", vmin=-3, vmax=3, aspect="auto")
ax.set_xticks(range(6)); ax.set_xticklabels(["Height", "Weight", "Speed\n(40, size-adj.)", "Explosion\n(jumps)", "Agility\n(cone, shuttle)", "Strength\n(bench)"], fontsize=7.5)
ax.set_yticks(range(len(A))); ax.set_yticklabels([f"{names[str(i)]} (n={int(A.n[i])})" for i in A.index], fontsize=8)
for i in range(len(A)):
    for j, c in enumerate(cf): ax.text(j, i, f"{A[c].iloc[i]:+.1f}", ha="center", va="center", fontsize=7.5, color="white" if abs(A[c].iloc[i]) > 1.6 else INK)
ax.grid(False); cb = plt.colorbar(im, ax=ax, fraction=.03, pad=.02); cb.set_label("Cluster centre (robust z)", fontsize=7.5)
plt.tight_layout(); save("fig_archetypes")

# ---- 13 per class
fig, ax = plt.subplots(figsize=(7.4, 2.9))
pc = pd.crosstab(df[df.unicorn].season, df[df.unicorn].utype).reindex(columns=order[:3]); bot = np.zeros(len(pc))
for t in order[:3]: ax.bar(pc.index, pc[t], bottom=bot, color=C[t], label=t, width=.78); bot += pc[t].values
ax.set_xlabel("Combine class"); ax.set_ylabel("Unicorns flagged"); ax.legend(fontsize=7.5, ncol=3, loc="upper left")
ax.annotate("2021: no combine;\npro-day pool is larger\nand less selected", xy=(2021, bot[list(pc.index).index(2021)]), xytext=(2012.2, 36), fontsize=7, arrowprops=dict(arrowstyle="-", color="#777", lw=.6))
plt.tight_layout(); save("fig_perclass")

# ---- 14 sensitivity
S = pd.read_csv("out/sensitivity.csv"); S = S[S.type == "Freak"]
fig, ax = plt.subplots(figsize=(7.4, 3.0))
defs = {"Consensus top 2%": G.cons_pct > .98, "Consensus top 5%": G.cons_pct > .95, "Consensus top 10%": G.cons_pct > .90, "Isolation Forest only (5%)": G.p_if > .975,
        "LOF only (5%)": G.p_lof > .975, "Mahalanobis only (5%)": G.p_md > .975, "Full-workout view only (5%)": G.cons_full > .95, "Size-speed view only (5%)": G.cons_core > .95}
for i, (k, m) in enumerate(defs.items()):
    x = G[m & (G.ath_z >= .75)].z; lo, hi = ci(x)
    ax.errorbar(x.mean(), i, xerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", color="#E69F00" if k != "Consensus top 5%" else INK, capsize=3, ms=6, lw=1.4)
    ax.text(.72, i, f"n={len(x)}", fontsize=7, va="center", color="#556")
ax.axvline(0, color=INK, lw=.6); ax.set_yticks(range(len(defs))); ax.set_yticklabels(list(defs)); ax.invert_yaxis(); ax.set_xlim(-.45, .85)
ax.set_xlabel("Mean value over slot (z) of drafted Freak unicorns, 95% bootstrap CI")
plt.tight_layout(); save("fig_sensitivity")
print("figures done")
