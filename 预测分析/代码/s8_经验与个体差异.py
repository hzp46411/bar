# -*- coding: utf-8 -*-
"""第 8 步：经验与个体差异
  1 经验：选择与自己预测的一致率、心理测量斜率，按 100 轮区块（第 4 层：经验使信念权重升到约 1.9 倍）
  2 个体差异：每人的一致率与第 4 层模型中该人的"信念份额"（|β|·SD(V) / (|β|·SD(V) + |κ|·SD(c))）的相关
输出：结果/s8_经验与个体差异.json
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import pred_lib as PL
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
ms = PL.model_states("HRGPR")
br = (P <= 60).astype(float); cons = np.where(OK, A == br, np.nan)
out = {"按区块": []}
for k in range(4):
    s = slice(k * 100, (k + 1) * 100)
    x = ((60.5 - P[:, s]) / 10)[OK[:, s]]; a = A[:, s][OK[:, s]]; g = np.nonzero(OK[:, s])[0]
    def nll(q):
        eta = q[:n][g] + q[n] * x; return float(np.sum(np.logaddexp(0, eta) - a * eta))
    q = minimize(nll, np.zeros(n + 1), method="L-BFGS-B").x
    out["按区块"].append(dict(区块=f"{k*100+1}–{(k+1)*100}", 一致率=float(np.nanmean(cons[:, s])), 共用心理测量斜率=float(q[n])))
share = np.abs(ms["beta"]) * ms["V"].std(1) / (np.abs(ms["beta"]) * ms["V"].std(1) + np.abs(ms["X"][:, 2]) * ms["c"].std(1))
ci = np.nanmean(cons, 1)
out["个体差异"] = {"Spearman(一致率, 模型信念份额)": [float(x) for x in stats.spearmanr(ci, share)],
               "信念主导（份额 > .5）的人的一致率中位数": float(np.median(ci[share > .5])),
               "惯性主导的人的一致率中位数": float(np.median(ci[share <= .5])),
               "Mann-Whitney p": float(stats.mannwhitneyu(ci[share > .5], ci[share <= .5]).pvalue)}
PL.save(out, "s8_经验与个体差异.json")
import json; print(json.dumps(out, ensure_ascii=False, indent=1))
