"""Проверка корректности: случайные ходы с откатом и без, сравнение
инкрементальной стоимости с полной; короткие прогоны всех методов."""
import sys
import time

import numpy as np

from algorithms import genetic, greedy, hill_climbing, random_neighbor, simulated_annealing, _mutate
from ctt import Instance, State, full_cost

I = Instance(sys.argv[1])
print(I.stats())
t0 = time.time()
g = greedy(I, "saturation")
print("greedy-sat", g.hard, g.soft, f"{time.time() - t0:.2f}s")
rng = np.random.default_rng(0)
S = State(I, rng.integers(I.T, size=I.L), rng.integers(I.R, size=I.L))
for i in range(20000):
    d, undo = random_neighbor(S, rng)
    if rng.random() < 0.5:
        undo()
    if i % 4000 == 0:
        S.sync(); assert (S.hard, S.soft) == full_cost(I, S.t, S.r), (i, S.hard, S.soft, full_cost(I, S.t, S.r))
S.sync(); assert (S.hard, S.soft) == full_cost(I, S.t, S.r)
print("incremental == full: OK")
t0 = time.time()
n = 0
while time.time() - t0 < 1:
    d, undo = random_neighbor(S, rng); undo(); n += 1
print("moves/s ~", n)
for name, fn in [("HC", lambda S: hill_climbing(S, rng, 3)), ("SA", lambda S: simulated_annealing(S, rng, 3))]:
    S = State(I, g.t, g.r)
    sol, n = fn(S)
    print(name, full_cost(I, *sol), "moves", n)
pop = [(g.t.copy(), g.r.copy())] + [_mutate(I, g.t, g.r, rng, 0.05) for _ in range(29)]
for name, lm in [("GA", 0), ("MA", 500)]:
    sol, gens, ev = genetic(I, pop, rng, 3, local_moves=lm)
    print(name, full_cost(I, *sol), "gens", gens, "evals", ev)
