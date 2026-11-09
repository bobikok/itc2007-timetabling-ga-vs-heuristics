"""Эксперимент: сравнение эвристик и метаэвристик на экземплярах ITC-2007 (UD2).

Запуск: python run_experiment.py <папка с compXX.lp> <папка результатов> [budget_s] [seeds] [workers]
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from algorithms import cpu, genetic, greedy, hill_climbing, simulated_annealing, _mutate
from ctt import Instance, State, full_cost, soft_breakdown

INSTANCES = [f"comp0{i}" for i in range(1, 8)]


def record(I, name, method, seed, sol, secs, extra=None):
    h, s = full_cost(I, *sol)
    row = dict(instance=name, method=method, seed=seed, hard=h, soft=s, feasible=h == 0, time=round(secs, 2))
    row.update(soft_breakdown(I, *sol))
    if extra:
        row.update(extra)
    return row


def run_one(args):
    path, method, seed, budget = args
    I = Instance(path)
    name = os.path.basename(path).replace(".lp", "")
    rng = np.random.default_rng(seed)
    start = greedy(I, "saturation")  # общий старт для всех метаэвристик (не входит в бюджет)
    log = []
    t0 = cpu()
    if method == "HC":
        S = State(I, start.t, start.r)
        sol, n = hill_climbing(S, rng, budget, log=log)
        extra = dict(moves=n)
        S.sync()
        assert (S.hard, S.soft) == full_cost(I, S.t, S.r), "инкрементальная оценка разошлась с полной"
    elif method == "SA":
        S = State(I, start.t, start.r)
        sol, n = simulated_annealing(S, rng, budget, log=log)
        extra = dict(moves=n)
        S.sync()
        assert (S.hard, S.soft) == full_cost(I, S.t, S.r), "инкрементальная оценка разошлась с полной"
    else:
        pop = [(start.t.copy(), start.r.copy())]
        while len(pop) < 30:
            pop.append(_mutate(I, start.t, start.r, rng, 0.05))
        if method == "GA":
            sol, gens, evals = genetic(I, pop, rng, budget, pop_size=30, local_moves=0, log=log)
        else:
            sol, gens, evals = genetic(I, pop, rng, budget, pop_size=20, local_moves=2000, log=log)
        extra = dict(generations=gens, evaluations=evals)
    row = record(I, name, method, seed, sol, cpu() - t0, extra)
    return row, dict(instance=name, method=method, seed=seed, curve=log)


def main(inst_dir, out_dir, budget=60.0, seeds=3, workers=3):
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(inst_dir, f"{n}.lp") for n in INSTANCES]
    rows, curves, stats = [], [], []
    for p in paths:
        I = Instance(p)
        name = os.path.basename(p).replace(".lp", "")
        st = I.stats()
        st["instance"] = name
        stats.append(st)
        t0 = cpu()
        g = greedy(I, "saturation")
        rows.append(record(I, name, "Greedy-Sat", 0, (g.t, g.r), cpu() - t0))
        for s in range(seeds):
            t0 = cpu()
            g = greedy(I, "random", rng=np.random.default_rng(100 + s))
            rows.append(record(I, name, "Greedy-Rand", s, (g.t, g.r), cpu() - t0))
        print(name, st, "greedy:", rows[-seeds - 1]["hard"], rows[-seeds - 1]["soft"], flush=True)
    jobs = [(p, m, s, budget) for p in paths for m in ["HC", "SA", "GA", "MA"] for s in range(seeds)]
    t_start = time.time()
    with Pool(workers) as pool:
        for i, (row, curve) in enumerate(pool.imap_unordered(run_one, jobs)):
            rows.append(row)
            curves.append(curve)
            print(f"[{i + 1}/{len(jobs)}] {row['instance']} {row['method']} s{row['seed']}: "
                  f"hard={row['hard']} soft={row['soft']} ({time.time() - t_start:.0f}s)", flush=True)
            import pandas as pd
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    pd.DataFrame(stats).to_csv(os.path.join(out_dir, "instance_stats.csv"), index=False)
    json.dump(curves, open(os.path.join(out_dir, "curves.json"), "w"))
    json.dump(dict(budget_s=budget, seeds=seeds, pop_size=30, tournament=3, elite=2, mut_rate=0.01,
                   ma_local_moves=2000, ma_pop_size=20, sa_Tend=0.2, init_mut_rate=0.05),
              open(os.path.join(out_dir, "config.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], float(a[3]) if len(a) > 3 else 20.0, int(a[4]) if len(a) > 4 else 3,
         int(a[5]) if len(a) > 5 else 3)
