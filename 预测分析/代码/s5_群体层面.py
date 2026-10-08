# -*- coding: utf-8 -*-
"""第 5 步：群体层面——共同的信念冲击与自我否定的预言
  平均预测与本轮人数的负相关有两个来源：
    a 组成效应：去的人报得低、不去的人报得高；本轮去的人越多，平均预测机械地越低
    b 共同信念冲击：某一轮大家一起觉得会挤 → 都不去 → 果然不挤（自我否定的预言，El Farol 的核心逻辑）
  做法：先去掉组成效应——每人的预测减去本人均值，再减去"去 / 不去"的平均差（按人估计），得到信念残差 e_it；
       每轮平均 ē_t；分别只用"去的人"和"不去的人"的残差均值，也做同样检验。
  σ 的来源：第 4 层模型（HRGPRS）中每轮共同冲击的后验均值 ε̂_t，与 ē_t 的相关；把 ē_t 加进选择模型后 σ 是否下降。
输出：结果/s5_群体层面.json
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import pred_lib as PL
L = PL.L
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
NP = PL.NPREV
# 每人：P = μ_i + g_i·去 + e
E = np.full((n, T), np.nan); gap = np.zeros(n)
for i in range(n):
    m = OK[i]; a = A[i, m]; y = P[i, m]
    if 0 < a.mean() < 1:
        gap[i] = y[a == 1].mean() - y[a == 0].mean()
    mu = y.mean() - gap[i] * a.mean()
    E[i, m] = y - mu - gap[i] * a
e_bar = np.nanmean(E, 0)
e_go = np.array([np.nanmean(E[A[:, t] == 1, t]) for t in range(T)])
e_st = np.array([np.nanmean(E[A[:, t] == 0, t]) if (A[:, t] == 0).any() else np.nan for t in range(T)])
sl = slice(1, None)
def reg(x, lab):
    X = np.column_stack([np.ones(T - 1), x[sl], (NP - 60)[sl] / 10]); y = (N - 60)[sl]
    b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
    cov = np.linalg.inv(X.T @ X) * (e @ e / (len(y) - 3))
    return dict(变量=lab, 系数=float(b[1]), t=float(b[1] / np.sqrt(cov[1, 1])), 相关=float(stats.pearsonr(x[sl], N[sl])[0]))
out = {"逐人去 / 不去的预测差（中位数）": float(np.median(gap)),
       "本轮人数 − 60 = a + b·(变量) + c·(上一轮−60)/10": [reg(np.nanmean(P, 0) - 60, "原始平均预测（含组成效应）"),
                                                    reg(e_bar, "去掉组成效应的平均信念残差 ē_t"),
                                                    reg(e_go, "只用去的人的信念残差"), reg(e_st, "只用不去的人的信念残差")],
       "ē_t 的轮间 SD（人）": float(np.nanstd(e_bar)),
       "ē_t 的一阶自相关": float(np.corrcoef(e_bar[1:], e_bar[:-1])[0, 1]),
       "ē_t 与上一轮人数的相关": float(stats.pearsonr(e_bar[sl], NP[sl])[0])}
# ---------- σ 的来源 ----------
f, sp, X, phi = PL.fit("HRGPRS")
z = L.run(X, sp, phi, A, L.G_REAL, L.S_REAL, N, out="z"); sig = f["sigma"]
nodes, w = L.GH_NODES, L.GH_W
ll = (A[None] * (z[None] + sig * nodes[:, None, None]) - np.logaddexp(0, z[None] + sig * nodes[:, None, None])).sum(1)   # (Q, T)
post = np.exp(ll - ll.max(0)) * w[:, None]; post /= post.sum(0)
eps_hat = (nodes[:, None] * post).sum(0)                     # 每轮共同冲击的后验均值
r = stats.pearsonr(e_bar[sl], eps_hat[sl])
out["σ 的来源"] = {"corr(ē_t, ε̂_t)": [float(r[0]), float(r[1])],
                 "corr(只用不去的人的残差, ε̂_t)": float(stats.pearsonr(e_st[sl], eps_hat[sl])[0]),
                 "corr(只用去的人的残差, ε̂_t)": float(stats.pearsonr(e_go[sl], eps_hat[sl])[0])}
# 把 ē_t 加进选择模型（两步：HRGPRS 其余参数固定），σ 是否下降
eb = np.nan_to_num(e_bar)
nll = lambda q: float(np.sum(np.logaddexp(0, z + q[0] * eb[None]) - A * (z + q[0] * eb[None])))
q = minimize(nll, [0.0], method="L-BFGS-B").x
s1, l1 = L.marginal_sigma(z + q[0] * eb[None], A); s0, l0 = L.marginal_sigma(z, A)
out["σ 的来源"].update({"选择模型中 ē_t 的系数（每人）": float(q[0]), "σ：加入前": float(s0), "σ：加入后": float(s1),
                       "σ² 被解释的比例": float(1 - s1 ** 2 / s0 ** 2), "LR（含 σ）": float(2 * (l1 - l0))})
# 预测分歧与人数离 60 的距离（Arthur：预期的异质性帮助协调）
disp = np.nanstd(E, 0)
out["预测分歧"] = {"corr(信念残差的人间 SD, |本轮人数 − 60|)": [float(x) for x in stats.pearsonr(disp[sl], np.abs(N - 60)[sl])]}
PL.save(dict(out, ebar=e_bar, eps_hat=eps_hat), "s5_群体层面.json")
import json; print(json.dumps(out, ensure_ascii=False, indent=1))
