# -*- coding: utf-8 -*-
"""直观检查：基线模型 M0（无调节）之下，实际"重复上一轮选择"的比例 − 模型预测的重复概率，按调节变量分组。"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
X = L.base_X(); lam = L.base_lambda(); sp = L.Spec()
z = L.run(X, sp, np.r_[lam], L.A_REAL, L.G_REAL, L.S_REAL, L.ATT, out="z")
rel = L.run(X, sp, np.r_[lam], L.A_REAL, L.G_REAL, L.S_REAL, L.ATT, out="rel")
p = expit(z); A = L.A_REAL
prev = np.c_[np.full((A.shape[0], 1), np.nan), A[:, :-1]]
rep = (A == prev).astype(float); prep = np.where(prev == 1, p, 1 - p)
ok = ~np.isnan(prev); ok[:, :2] = False
pm, lag = L.public_mods(L.ATT)
out = {}
def table(name, M, bins, labels):
    rows = []
    for (lo, hi), lab in zip(bins, labels):
        m = ok & (M >= lo) & (M < hi)
        rows.append(dict(组=lab, 观测数=int(m.sum()), 实际重复=round(float(rep[m].mean()), 4), 预测重复=round(float(prep[m].mean()), 4),
                         差值=round(float((rep[m] - prep[m]).mean()), 4), 差值SE=round(float((rep[m] - prep[m]).std() / np.sqrt(m.sum())), 4)))
    out[name] = rows
stab = np.broadcast_to(pm["stab"], A.shape); dev = np.broadcast_to(pm["dev"], A.shape)
table("状态持续轮数", stab, [(0, 1), (1, 2), (2, 3), (3, 5)], ["刚翻转(0)", "1", "2", "≥3"])
table("上轮偏离幅度", dev, [(0, .25), (.25, .55), (.55, .85), (.85, 9)], ["≤2 人", "3–5 人", "6–8 人", "≥9 人"])
q = np.nanpercentile(rel[ok], [0, 25, 50, 75, 100]); q[-1] += 1
table("相对可靠性(四分位)", rel, list(zip(q[:-1], q[1:])), ["Q1 惯性更可靠", "Q2", "Q3", "Q4 信念更可靠"])
L.save_json(out, L.OUT / "残差重复率.json")
for k, v in out.items():
    print(k)
    for r in v: print("  ", r)
