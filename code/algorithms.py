"""Алгоритмы составления расписания: жадные эвристики, локальный спуск,
имитация отжига, генетический и меметический алгоритмы.

У всех стохастических методов одинаковый бюджет времени (секунды CPU-времени
процесса) и одинаковая окрестность: перенос лекции в случайный слот (t, r)
или обмен слотами двух лекций разных курсов.
"""
import math
import time

import numpy as np

from ctt import HARD_W, State, full_cost


def cpu():
    return time.process_time()


# ===================================================================== жадные эвристики
def greedy(I, order="saturation", rng=None):
    """Конструктивная эвристика. Лекции ставятся по одной в слот (t, r)
    с минимальным приростом стоимости.
      order='saturation' — сначала самые «трудные» курсы: мало доступных
                           периодов, много конфликтов, много лекций (аналог DSatur);
      order='random'     — случайный порядок курсов."""
    rng = rng or np.random.default_rng(0)
    if order == "saturation":
        avail = I.T - I.unavail.sum(1)
        degree = I.conflict.sum(1) - 1
        key = degree * I.n_lect / np.maximum(avail, 1)
        courses = list(np.argsort(-key, kind="stable"))
    else:
        courses = list(rng.permutation(I.C))
    lect_of = {c: np.where(I.lect_course == c)[0] for c in range(I.C)}
    # начальное «пустое» состояние: все лекции стоят в фиктивных местах,
    # поэтому собираем решение пошагово с собственными счётчиками
    t = np.full(I.L, -1)
    r = np.full(I.L, -1)
    occ = np.zeros((I.T, I.R), dtype=bool)
    busy_c = np.zeros((I.C, I.T), dtype=bool)  # период занят конфликтующим курсом
    room_pref = {}
    for c in courses:
        used_days = set()
        for l in lect_of[c]:
            best, best_key = None, None
            for tt in range(I.T):
                h = int(busy_c[c, tt]) + int(I.unavail[c, tt])
                free = np.where(~occ[tt])[0]
                if len(free) == 0:
                    continue
                # предпочтение: аудитория курса (стабильность), иначе минимально достаточная
                if c in room_pref and room_pref[c] in free:
                    rr = room_pref[c]
                else:
                    fit = free[I.cap[free] >= I.students[c]]
                    rr = fit[np.argmin(I.cap[fit])] if len(fit) else free[np.argmax(I.cap[free])]
                s = max(0, I.students[c] - I.cap[rr]) + (0 if (tt // I.P) not in used_days else 3)
                k = (h, s, rng.random())
                if best_key is None or k < best_key:
                    best, best_key = (tt, rr), k
            if best is None:  # все аудитории заняты — ставим с нарушением
                best = (int(rng.integers(I.T)), int(rng.integers(I.R)))
            tt, rr = best
            t[l], r[l] = tt, rr
            occ[tt, rr] = True
            busy_c[I.conflict[c], tt] = True
            used_days.add(tt // I.P)
            room_pref.setdefault(c, rr)
    return State(I, t, r)


# ===================================================================== окрестность
def random_neighbor(S, rng):
    """Случайный ход: 50% — перенос лекции, 50% — обмен двух лекций.
    Возвращает (delta, undo)."""
    I = S.I
    a = int(rng.integers(I.L))
    if rng.random() < 0.5:
        t0, r0 = S.tl[a], S.rl[a]
        d = S.move(a, int(rng.integers(I.T)), int(rng.integers(I.R)))
        return d, lambda: S.move(a, t0, r0)
    b = int(rng.integers(I.L))
    if I.lect_course[a] == I.lect_course[b]:
        return 0, lambda: None
    d = S.swap(a, b)
    return d, lambda: S.swap(a, b)


# ===================================================================== локальный спуск
def hill_climbing(S, rng, budget, log=None, max_moves=None):
    """Принимаем ходы, не ухудшающие стоимость (с «боковыми» ходами)."""
    t0 = cpu()
    best = S.cost
    best_sol = (np.array(S.tl), np.array(S.rl))
    n = 0
    while cpu() - t0 < budget and (max_moves is None or n < max_moves):
        d, undo = random_neighbor(S, rng)
        n += 1
        if d > 0:
            undo()
        elif S.cost < best:
            best, best_sol = S.cost, (np.array(S.tl), np.array(S.rl))
        if log is not None and n % 2000 == 0:
            log.append((cpu() - t0, best))
    return best_sol, n


# ===================================================================== имитация отжига
def simulated_annealing(S, rng, budget, T0=None, Tend=0.2, log=None):
    """Классическая имитация отжига с геометрическим охлаждением по времени:
    T(τ) = T0·(Tend/T0)^τ, τ — доля израсходованного бюджета."""
    start = cpu()
    if T0 is None:  # калибровка: средний модуль ухудшающих мягких ходов
        ds = []
        for _ in range(300):
            d, undo = random_neighbor(S, rng)
            if 0 < d < HARD_W:
                ds.append(d)
            undo()
        T0 = max(1.0, float(np.mean(ds)) if ds else 5.0)
    best = S.cost
    best_sol = (np.array(S.tl), np.array(S.rl))
    n, T = 0, T0
    while True:
        if n % 200 == 0:
            frac = (cpu() - start) / budget
            if frac >= 1:
                break
            T = T0 * (Tend / T0) ** frac
        d, undo = random_neighbor(S, rng)
        n += 1
        if d > 0 and rng.random() >= math.exp(-d / T):
            undo()
        elif S.cost < best:
            best, best_sol = S.cost, (np.array(S.tl), np.array(S.rl))
        if log is not None and n % 2000 == 0:
            log.append((cpu() - start, best))
    return best_sol, n


# ===================================================================== генетический алгоритм
def _crossover(I, pa, pb, rng):
    """Покурсовой равномерный кроссинговер: все лекции курса берутся
    от одного из родителей (сохраняются «строительные блоки» курса)."""
    take_a = rng.random(I.C) < 0.5
    m = take_a[I.lect_course]
    t = np.where(m, pa[0], pb[0])
    r = np.where(m, pa[1], pb[1])
    return t, r


def _mutate(I, t, r, rng, rate):
    t, r = t.copy(), r.copy()
    k = max(1, rng.binomial(I.L, rate))
    for l in rng.choice(I.L, size=k, replace=False):
        if rng.random() < 0.5:
            t[l], r[l] = rng.integers(I.T), rng.integers(I.R)
        else:
            o = rng.integers(I.L)
            t[l], t[o] = t[o], t[l]
            r[l], r[o] = r[o], r[l]
    return t, r


def _fitness(I, t, r):
    h, s = full_cost(I, t, r)
    return HARD_W * h + s


def genetic(I, init_pop, rng, budget, pop_size=30, tour=3, elite=2, mut_rate=0.01,
            local_moves=0, log=None):
    """Генетический алгоритм (local_moves=0) или меметический (local_moves>0:
    каждый потомок улучшается локальным спуском из local_moves ходов)."""
    start = cpu()
    pop = [(p[0].copy(), p[1].copy()) for p in init_pop[:pop_size]]
    fit = [_fitness(I, *p) for p in pop]
    gens = evals = 0
    while cpu() - start < budget:
        order = np.argsort(fit)
        new = [pop[i] for i in order[:elite]]
        new_fit = [fit[i] for i in order[:elite]]
        while len(new) < pop_size and cpu() - start < budget:
            def pick():
                idx = rng.choice(len(pop), size=tour, replace=False)
                return pop[min(idx, key=lambda i: fit[i])]
            child = _crossover(I, pick(), pick(), rng)
            child = _mutate(I, *child, rng, mut_rate)
            if local_moves:
                S = State(I, *child)
                sol, n = hill_climbing(S, rng, budget=1e9, max_moves=local_moves)
                evals += n
                child = sol
            new.append(child)
            new_fit.append(_fitness(I, *child))
            evals += 1
        pop, fit = new, new_fit
        gens += 1
        if log is not None:
            log.append((cpu() - start, min(fit)))
    i = int(np.argmin(fit))
    return pop[i], gens, evals
