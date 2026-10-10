# -*- coding: utf-8 -*-
"""第 7 步（旁支）：不依赖结构模型的 γ 粗估——"信念份额外推"（已被 s9 的工具变量取代）
  思路：去的人与不去的人的预测差 gap_i 混合了"信不挤才去"（选择效应）与"选择改写预测"（γ）。
       选择效应的大小应随这个人"按信念行动"的程度增大；把 gap_i 对第 4 层模型估出的信念份额回归，
       外推到信念份额 = 0（完全不按信念行动的人），剩下的就只有 γ。
  信念份额 = |β|·SD(V) / (|β|·SD(V) + |κ|·SD(c))（HRGPR 的逐人参数与状态）；按 gap 的抽样方差加权回归。
  局限：信念份额是模型量，且与合理化的程度相关（见 s10、s11），外推只能给出粗略量级。
输出：结果/s7c_信念份额外推.json
"""
import json
import numpy as np
from scipy import stats
import pred_lib as PL
P, A, OK, n = PL.P, PL.A, PL.OK, PL.n
ms = PL.model_states("HRGPR")
share = np.abs(ms["beta"]) * ms["V"].std(1) / (np.abs(ms["beta"]) * ms["V"].std(1) + np.abs(ms["X"][:, 2]) * ms["c"].std(1))
gap = np.full(n, np.nan); se = np.full(n, np.nan)
for i in range(n):
    m = OK[i]; a = A[i, m]; y = P[i, m]
    if 5 <= a.sum() <= len(a) - 5:
        g1, g0 = y[a == 1], y[a == 0]; gap[i] = g1.mean() - g0.mean(); se[i] = np.sqrt(g1.var() / len(g1) + g0.var() / len(g0))
ok = ~np.isnan(gap); w = 1 / se[ok] ** 2
X = np.column_stack([np.ones(ok.sum()), share[ok]])
b = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ gap[ok]); cov = np.linalg.inv((X * w[:, None]).T @ X)
res = gap[ok] - X @ b; cov = cov * np.sum(w * res ** 2) / (ok.sum() - 2)
groups = {}
for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, 1.01)):
    m = ok & (share >= lo) & (share < hi)
    groups[f"{lo}–{hi}"] = dict(人数=int(m.sum()), 预测差中位数=float(np.median(gap[m])), 加权均值=float(np.average(gap[m], weights=1 / se[m] ** 2)))
out = dict(外推截距=float(b[0]), 截距SE=float(np.sqrt(cov[0, 0])), 斜率=float(b[1]), 人数=int(ok.sum()), 按信念份额分组=groups,
           Spearman=[float(x) for x in stats.spearmanr(gap[ok], share[ok])])
PL.save(out, "s7c_信念份额外推.json")
print(json.dumps(out, ensure_ascii=False, indent=1))
