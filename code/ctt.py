"""Curriculum-Based Course Timetabling (ITC-2007, формулировка UD2).

Модель задачи, чтение экземпляров (ASP-факты из коллекции Potassco), полная и
инкрементальная оценка решения.

Жёсткие ограничения (штраф HARD_W за каждое нарушение):
  H1 Lectures      — лекции одного курса в разных периодах;
  H2 Conflicts     — лекции одной учебной программы (curriculum) и одного
                     преподавателя в разных периодах;
  H3 RoomOccupancy — не более одной лекции в аудитории за период;
  H4 Availability  — лекции не ставятся в недоступные периоды.
Мягкие ограничения UD2:
  S1 RoomCapacity (×1), S2 MinWorkingDays (×5),
  S3 IsolatedLectures / CurriculumCompactness (×2), S4 RoomStability (×1).
"""
import re
from collections import defaultdict

import numpy as np

HARD_W = 1000
W_CAP, W_MWD, W_ISO, W_STAB = 1, 5, 2, 1


class Instance:
    def __init__(self, path):
        txt = open(path, encoding="utf-8").read()
        self.name = re.search(r'name\("([^"]+)"\)', txt).group(1)
        self.D = int(re.search(r"^days\((\d+)\)", txt, re.M).group(1))
        self.P = int(re.search(r"^periods_per_day\((\d+)\)", txt, re.M).group(1))
        self.T = self.D * self.P
        courses = re.findall(r"^course\((\"[^\"]+\"|\w+),(\"[^\"]+\"|\w+),(\d+),(\d+),(\d+),\d+\)\.", txt, re.M)
        rooms = re.findall(r"^room\((\"[^\"]+\"|\w+),(\d+),\d+\)\.", txt, re.M)
        self.course_ids = [c[0] for c in courses]
        cidx = {c: i for i, c in enumerate(self.course_ids)}
        teachers = sorted({c[1] for c in courses})
        tidx = {t: i for i, t in enumerate(teachers)}
        self.C, self.R = len(courses), len(rooms)
        self.teacher = np.array([tidx[c[1]] for c in courses])
        self.n_lect = np.array([int(c[2]) for c in courses])
        self.mwd = np.array([int(c[3]) for c in courses])
        self.students = np.array([int(c[4]) for c in courses])
        self.room_ids = [r[0] for r in rooms]
        self.cap = np.array([int(r[1]) for r in rooms])
        cur = defaultdict(list)
        for q, c in re.findall(r"^curricula\((\"[^\"]+\"|\w+),(\"[^\"]+\"|\w+)\)\.", txt, re.M):
            cur[q].append(cidx[c])
        self.curricula = [sorted(v) for v in cur.values()]
        self.Q = len(self.curricula)
        self.course_curr = [[] for _ in range(self.C)]
        for q, cs in enumerate(self.curricula):
            for c in cs:
                self.course_curr[c].append(q)
        self.unavail = np.zeros((self.C, self.T), dtype=bool)
        for c, d, p in re.findall(r"^unavailability_constraint\((\"[^\"]+\"|\w+),(\d+),(\d+)\)\.", txt, re.M):
            self.unavail[cidx[c], int(d) * self.P + int(p)] = True
        # лекции: номер -> курс
        self.lect_course = np.repeat(np.arange(self.C), self.n_lect)
        self.L = len(self.lect_course)
        # граф конфликтов курсов (общий преподаватель или общая программа)
        self.conflict = np.zeros((self.C, self.C), dtype=bool)
        for cs in self.curricula:
            for a in cs:
                for b in cs:
                    self.conflict[a, b] = True
        for a in range(self.C):
            self.conflict[a, self.teacher == self.teacher[a]] = True
        np.fill_diagonal(self.conflict, True)

    def stats(self):
        return dict(instance=self.name, courses=self.C, lectures=self.L, rooms=self.R, periods=self.T,
                    curricula=self.Q, teachers=int(self.teacher.max() + 1),
                    unavail=int(self.unavail.sum()),
                    occupancy=round(self.L / (self.R * self.T), 3),
                    conflict_density=round((self.conflict.sum() - self.C) / (self.C * (self.C - 1)), 3))


class State:
    """Расписание с инкрементальным пересчётом стоимости."""

    def __init__(self, inst, t, r):
        self.I = inst
        # быстрые списки Python для горячего цикла
        if not hasattr(inst, "_py"):
            inst._py = dict(lc=inst.lect_course.tolist(), cc=[list(q) for q in inst.course_curr],
                            tch=inst.teacher.tolist(), mwd=inst.mwd.tolist(), st=inst.students.tolist(),
                            cap=inst.cap.tolist(), un=inst.unavail.tolist())
        self.t = np.array(t, dtype=np.int64)
        self.r = np.array(r, dtype=np.int64)
        self.rebuild()

    # ------------------------------------------------------------ полная оценка
    def rebuild(self):
        I = self.I
        self.tl, self.rl = self.t.tolist(), self.r.tolist()
        self.occ = [[0] * I.R for _ in range(I.T)]
        self.crs = [[0] * I.T for _ in range(I.C)]
        self.cur = [[0] * I.T for _ in range(I.Q)]
        self.tch = [[0] * I.T for _ in range(int(I.teacher.max()) + 1)]
        self.dayc = [[0] * I.D for _ in range(I.C)]
        self.roomc = [[0] * I.R for _ in range(I.C)]
        self.ndays = [0] * I.C
        self.nrooms = [0] * I.C
        for l in range(I.L):
            self._add(l, self.tl[l], self.rl[l])
        self.hard, self.soft = full_cost(I, self.t, self.r)

    @property
    def cost(self):
        return HARD_W * self.hard + self.soft

    def sync(self):
        """Перенести текущее решение в numpy-массивы t, r."""
        self.t[:] = self.tl
        self.r[:] = self.rl

    def _add(self, l, t, r):
        py = self.I._py
        c = py["lc"][l]
        self.occ[t][r] += 1
        self.crs[c][t] += 1
        for q in py["cc"][c]:
            self.cur[q][t] += 1
        self.tch[py["tch"][c]][t] += 1
        d = t // self.I.P
        self.dayc[c][d] += 1
        if self.dayc[c][d] == 1:
            self.ndays[c] += 1
        self.roomc[c][r] += 1
        if self.roomc[c][r] == 1:
            self.nrooms[c] += 1

    def _remove(self, l, t, r):
        py = self.I._py
        c = py["lc"][l]
        self.occ[t][r] -= 1
        self.crs[c][t] -= 1
        for q in py["cc"][c]:
            self.cur[q][t] -= 1
        self.tch[py["tch"][c]][t] -= 1
        d = t // self.I.P
        self.dayc[c][d] -= 1
        if self.dayc[c][d] == 0:
            self.ndays[c] -= 1
        self.roomc[c][r] -= 1
        if self.roomc[c][r] == 0:
            self.nrooms[c] -= 1

    # ------------------------------------------------------------ локальные части стоимости
    def _hard_period(self, c, t, qs, tc):
        """Нарушения H1/H2 в периоде t, связанные с курсом c."""
        h = self.crs[c][t] - 1 if self.crs[c][t] > 1 else 0
        x = self.tch[tc][t]
        if x > 1:
            h += x - 1
        cur = self.cur
        for q in qs:
            x = cur[q][t]
            if x > 1:
                h += x - 1
        return h

    def _iso(self, q, d):
        P = self.I.P
        row = self.cur[q]
        base = d * P
        s = 0
        for p in range(P):
            x = row[base + p]
            if x and not (p > 0 and row[base + p - 1]) and not (p < P - 1 and row[base + p + 1]):
                s += x
        return s

    def _local(self, c, t1, r1, t2, r2, l_t, l_r, qs, tc, days):
        py = self.I._py
        o1, o2 = self.occ[t1][r1], self.occ[t2][r2]
        h = (o1 - 1 if o1 > 1 else 0) + ((o2 - 1 if o2 > 1 else 0) if (t1, r1) != (t2, r2) else 0)
        h += self._hard_period(c, t1, qs, tc)
        if t2 != t1:
            h += self._hard_period(c, t2, qs, tc)
        h += 1 if py["un"][c][l_t] else 0
        md = py["mwd"][c] - self.ndays[c]
        s = (W_MWD * md if md > 0 else 0) + W_STAB * (self.nrooms[c] - 1 if self.nrooms[c] > 1 else 0)
        ex = py["st"][c] - py["cap"][l_r]
        if ex > 0:
            s += W_CAP * ex
        iso = 0
        for q in qs:
            for d in days:
                iso += self._iso(q, d)
        return h, s + W_ISO * iso

    def move(self, l, t2, r2):
        """Перенести лекцию l в (t2, r2); возвращает изменение стоимости."""
        t1, r1 = self.tl[l], self.rl[l]
        if t1 == t2 and r1 == r2:
            return 0
        I = self.I
        py = I._py
        c = py["lc"][l]
        qs = py["cc"][c]
        tc = py["tch"][c]
        d1, d2 = t1 // I.P, t2 // I.P
        days = (d1,) if d1 == d2 else (d1, d2)
        hb, sb = self._local(c, t1, r1, t2, r2, t1, r1, qs, tc, days)
        self._remove(l, t1, r1)
        self._add(l, t2, r2)
        self.tl[l], self.rl[l] = t2, r2
        ha, sa = self._local(c, t1, r1, t2, r2, t2, r2, qs, tc, days)
        dh, ds = ha - hb, sa - sb
        self.hard += dh
        self.soft += ds
        return HARD_W * dh + ds

    def swap(self, a, b):
        ta, ra, tb, rb = self.tl[a], self.rl[a], self.tl[b], self.rl[b]
        return self.move(a, tb, rb) + self.move(b, ta, ra)


def full_cost(I, t, r):
    """Независимая полная оценка (используется для проверки инкрементальной)."""
    t, r = np.asarray(t), np.asarray(r)
    c = I.lect_course
    hard = 0
    occ = np.zeros((I.T, I.R), dtype=np.int64)
    np.add.at(occ, (t, r), 1)
    hard += np.maximum(occ - 1, 0).sum()
    crs = np.zeros((I.C, I.T), dtype=np.int64)
    np.add.at(crs, (c, t), 1)
    hard += np.maximum(crs - 1, 0).sum()
    tch = np.zeros((int(I.teacher.max()) + 1, I.T), dtype=np.int64)
    np.add.at(tch, (I.teacher[c], t), 1)
    hard += np.maximum(tch - 1, 0).sum()
    cur = np.zeros((I.Q, I.T), dtype=np.int64)
    for q, cs in enumerate(I.curricula):
        cur[q] = crs[cs].sum(0)
    hard += np.maximum(cur - 1, 0).sum()
    hard += int(I.unavail[c, t].sum())
    soft = W_CAP * int(np.maximum(I.students[c] - I.cap[r], 0).sum())
    dayc = np.zeros((I.C, I.D), dtype=np.int64)
    np.add.at(dayc, (c, t // I.P), 1)
    soft += W_MWD * int(np.maximum(I.mwd - (dayc > 0).sum(1), 0).sum())
    roomc = np.zeros((I.C, I.R), dtype=np.int64)
    np.add.at(roomc, (c, r), 1)
    soft += W_STAB * int(np.maximum((roomc > 0).sum(1) - 1, 0).sum())
    cd = cur.reshape(I.Q, I.D, I.P)
    left = np.zeros_like(cd)
    left[:, :, 1:] = cd[:, :, :-1]
    right = np.zeros_like(cd)
    right[:, :, :-1] = cd[:, :, 1:]
    iso = cd * ((left == 0) & (right == 0))
    soft += W_ISO * int(iso.sum())
    return int(hard), int(soft)


def soft_breakdown(I, t, r):
    t, r = np.asarray(t), np.asarray(r)
    c = I.lect_course
    cap = W_CAP * int(np.maximum(I.students[c] - I.cap[r], 0).sum())
    dayc = np.zeros((I.C, I.D), dtype=np.int64)
    np.add.at(dayc, (c, t // I.P), 1)
    mwd = W_MWD * int(np.maximum(I.mwd - (dayc > 0).sum(1), 0).sum())
    roomc = np.zeros((I.C, I.R), dtype=np.int64)
    np.add.at(roomc, (c, r), 1)
    stab = W_STAB * int(np.maximum((roomc > 0).sum(1) - 1, 0).sum())
    total_h, total_s = full_cost(I, t, r)
    return dict(S1_capacity=cap, S2_min_days=mwd, S3_isolated=total_s - cap - mwd - stab, S4_room_stability=stab)
