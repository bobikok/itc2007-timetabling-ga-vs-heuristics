"""Анализ результатов: таблицы LaTeX, статистические тесты, рисунки.

Запуск: python analyze.py ../results ../paper
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, wilcoxon  # noqa: E402

METHODS = ["Greedy-Rand", "Greedy-Sat", "GA", "MA", "HC", "SA"]
COLORS = {"Greedy-Rand": "#b9b8b0", "Greedy-Sat": "#8a8a85", "GA": "#eb6834", "MA": "#eda100",
          "HC": "#2a78d6", "SA": "#1baf7a"}
INST = [f"comp0{i}" for i in range(1, 8)]


def main(res, paper):
    os.makedirs(os.path.join(paper, "figures"), exist_ok=True)
    df = pd.read_csv(os.path.join(res, "runs.csv"))
    st = pd.read_csv(os.path.join(res, "instance_stats.csv"))
    out = {}

    # ---- табл. 1: характеристики задач
    lines = []
    for _, r in st.iterrows():
        lines.append(f"{r['instance']} & {r['courses']} & {r['lectures']} & {r['rooms']} & {r['periods']} & "
                     f"{r['curricula']} & {r['teachers']} & {r['occupancy']*100:.0f} & {r['conflict_density']*100:.1f} \\\\")
    open(os.path.join(paper, "tab_instances.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # ---- табл. 2: мягкий штраф (среднее ± ст. откл.), доля допустимых
    agg = df.groupby(["instance", "method"]).agg(soft=("soft", "mean"), sd=("soft", "std"), best=("soft", "min"),
                                                  feas=("feasible", "mean"), hard=("hard", "mean")).reset_index()
    lines = []
    for inst in INST:
        a = agg[agg.instance == inst].set_index("method")
        feas_methods = [m for m in METHODS if m in a.index and a.loc[m, "feas"] == 1]
        bestm = min(feas_methods, key=lambda m: a.loc[m, "soft"]) if feas_methods else None
        cells = []
        for m in METHODS:
            if m not in a.index:
                cells.append("--")
                continue
            r = a.loc[m]
            if r.feas < 1:
                s = f"\\textit{{{r.soft:.0f}}}$^{{*}}$"
            else:
                s = f"{r.soft:.0f}"
                if m == bestm:
                    s = f"\\textbf{{{s}}}"
            if not np.isnan(r.sd) and r.sd > 0:
                s += f"\\,{{\\scriptsize$\\pm${r.sd:.0f}}}"
            cells.append(s)
        lines.append(inst + " & " + " & ".join(cells) + " \\\\")
    open(os.path.join(paper, "tab_main.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    out["agg"] = agg.round(2).to_dict(orient="records")

    # ---- относительное отклонение от лучшего найденного (по задаче), средний ранг
    best_inst = df[df.feasible].groupby("instance").soft.min()
    df["gap"] = df.apply(lambda r: (r.soft - best_inst[r.instance]) / max(best_inst[r.instance], 1) * 100
                         if r.feasible else np.nan, axis=1)
    mean_inst = df.groupby(["instance", "method"]).soft.mean().unstack()[METHODS]
    ranks = mean_inst.rank(axis=1)
    out["mean_rank"] = ranks.mean().round(2).to_dict()
    out["mean_gap_pct"] = df.groupby("method").gap.mean().round(1).to_dict()
    out["feasible_rate"] = df.groupby("method").feasible.mean().round(3).to_dict()
    out["friedman"] = float(friedmanchisquare(*[mean_inst[m] for m in METHODS]).pvalue)

    # ---- парные сравнения по 21 паре (задача, запуск): Уилкоксон
    piv = df[df.method.isin(["HC", "SA", "GA", "MA"])].pivot_table(index=["instance", "seed"], columns="method", values="soft")
    pairs = {}
    for a_, b_ in [("SA", "GA"), ("HC", "GA"), ("MA", "GA"), ("HC", "SA"), ("HC", "MA"), ("SA", "MA")]:
        d = piv[a_] - piv[b_]
        pairs[f"{a_} vs {b_}"] = dict(mean_diff=round(float(d.mean()), 1), wins=int((d < 0).sum()), n=int(d.notna().sum()),
                                     p=float(wilcoxon(piv[a_], piv[b_]).pvalue))
    out["pairwise"] = pairs
    # улучшение относительно старта
    g = df[df.method == "Greedy-Sat"].set_index("instance").soft
    out["improvement_vs_greedy_pct"] = {m: round(float(np.mean([(1 - mean_inst.loc[i, m] / g[i]) * 100 for i in INST])), 1)
                                        for m in ["HC", "SA", "GA", "MA"]}
    gr = df[df.method == "Greedy-Rand"].groupby("instance").agg(soft=("soft", "mean"), feas=("feasible", "mean"), hard=("hard", "mean"))
    out["greedy_rand"] = gr.round(2).to_dict()
    out["greedy_sat"] = df[df.method == "Greedy-Sat"].set_index("instance")[["hard", "soft", "time"]].to_dict()

    # ---- состав штрафа у лучших решений методов
    comp = df[df.method.isin(["Greedy-Sat", "GA", "MA", "HC", "SA"])].groupby("method")[
        ["S1_capacity", "S2_min_days", "S3_isolated", "S4_room_stability"]].mean()
    out["soft_breakdown"] = comp.round(1).to_dict()
    # GA/MA: поколения и оценки
    out["ga_stats"] = df[df.method.isin(["GA", "MA"])].groupby("method")[["generations", "evaluations"]].mean().round(0).to_dict()
    out["ls_moves"] = df[df.method.isin(["HC", "SA"])].groupby("method")["moves"].mean().round(0).to_dict()

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#888"})
    # ---- рис. 1: штраф по задачам (точки — запуски)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ms = ["Greedy-Sat", "GA", "MA", "HC", "SA"]
    w = 0.16
    for k, m in enumerate(ms):
        for i, inst in enumerate(INST):
            vals = df[(df.instance == inst) & (df.method == m)].soft.values
            x = i + (k - 2) * w
            ax.bar(x, vals.mean(), width=w * 0.9, color=COLORS[m], label=m if i == 0 else None)
            if len(vals) > 1:
                ax.scatter([x] * len(vals), vals, s=6, color="#222", zorder=3)
    ax.set_xticks(range(len(INST)))
    ax.set_xticklabels(INST)
    ax.set_ylabel("Мягкий штраф S (меньше — лучше)")
    ax.set_yscale("log")
    ax.grid(axis="y", color="#ddd", lw=0.6, which="both")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(os.path.join(paper, "figures", "soft_by_instance.pdf"))
    fig.savefig(os.path.join(res, "soft_by_instance.png"), dpi=150)
    plt.close(fig)

    # ---- рис. 2: сходимость на двух задачах
    curves = json.load(open(os.path.join(res, "curves.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    for ax, inst in zip(axes, ["comp01", "comp07"]):
        for m in ["GA", "MA", "HC", "SA"]:
            cs = [c["curve"] for c in curves if c["instance"] == inst and c["method"] == m and c["curve"]]
            grid = np.linspace(0, 60, 121)
            ys = []
            for c in cs:
                t = np.array([p[0] for p in c])
                y = np.array([p[1] for p in c], dtype=float)
                ys.append(np.interp(grid, t, y, left=np.nan))
            if ys:
                ax.plot(grid, np.nanmedian(np.array(ys), axis=0), color=COLORS[m], lw=2, label=m)
        ax.set_title(inst, fontsize=9, loc="left")
        ax.set_xlabel("Процессорное время, с")
        ax.set_yscale("log")
        ax.grid(color="#ddd", lw=0.6, which="both")
    axes[0].set_ylabel("Лучшая стоимость F")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(paper, "figures", "convergence.pdf"))
    fig.savefig(os.path.join(res, "convergence.png"), dpi=150)
    plt.close(fig)

    json.dump(out, open(os.path.join(res, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    for k, v in out.items():
        if k != "agg":
            print(k, v)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
