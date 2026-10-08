# -*- coding: utf-8 -*-
"""可靠性仲裁的操作化比较（两步：HRGP 个体参数固定，去掉其中可靠性的全部作用后，换不同的可靠性定义重新估计 θB、θH；含 σ 的 LR，df 2）
  A 状态预测（现定义）：信念对结果的预测准确度 − "重复上一轮"会不会赢
  B 建议成绩：按信念建议行动会不会赢 − 按习惯方向行动会不会赢（Lee 2014 的"哪个系统的建议更灵"）
  C 只看信念建议成绩   D 只看习惯建议成绩
  速率 a ∈ {0.05, 0.1, 0.2, 0.4}（指数加权，只用 t−1 及以前的信息）"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

A, G, S = L.A_REAL, L.G_REAL, L.S_REAL
fit = json.loads((L.OUT / "拟合" / "HRGP.json").read_text(encoding="utf-8"))
sp = L.Spec(**fit["spec"]); X = np.array(fit["X"]); sh = dict(fit["shared"])
sh["θR_rel"] = sh["θG_rel"] = 0.0
phi = np.array([sh[k] for k in sp.names()])
run = lambda Xv: L.run(Xv, sp, phi, A, G, S, L.ATT, out="z")
z = run(X); Xb = X.copy(); Xb[:, 1] = 0; Xk = X.copy(); Xk[:, 2] = 0
bp = z - run(Xb); hp = z - run(Xk); rest = z - bp - hp
rho, beta, aH = expit(X[:, 0]), X[:, 1], float(expit(fit["shared"]["logit_aH"]))
n, T = A.shape
BL = np.full(n, 1 / 3); BH = np.full(n, 1 / 3); H = np.full(n, .5)
accB = np.zeros((n, T)); wB = np.zeros((n, T)); wH = np.zeros((n, T)); wR = np.full((n, T), np.nan)
for t in range(T):
    recB = (beta * (BL - 0.7 * BH) > 0).astype(float); recH = (2 * H - 1 > 0).astype(float)
    accB[:, t] = 1 - (np.abs(G[:, t] - BL) + np.abs(S[:, t] - BH)) / 2
    wB[:, t] = recB * G[:, t] + (1 - recB) * S[:, t]; wH[:, t] = recH * G[:, t] + (1 - recH) * S[:, t]
    if t > 0:
        wR[:, t] = A[:, t - 1] * G[:, t] + (1 - A[:, t - 1]) * S[:, t]
    BL += rho * (G[:, t] - BL); BH += rho * (S[:, t] - BH); H += aH * (A[:, t] - H)
def ewma_prev(x, a):
    m = np.zeros_like(x); cur = np.full(n, .5)
    for t in range(T):
        m[:, t] = cur
        if not np.isnan(x[:, t]).all():
            cur = (1 - a) * cur + a * np.nan_to_num(x[:, t], nan=.5)
    return m
nll = lambda d, M: float((np.logaddexp(0, rest + bp * np.exp(d[0] * M) + hp * np.exp(d[1] * M)) - A * (rest + bp * np.exp(d[0] * M) + hp * np.exp(d[1] * M))).sum())
_, ll0 = L.marginal_sigma(rest + bp + hp, A)
rows = []
for a in (0.05, 0.1, 0.2, 0.4):
    defs = {"A 状态预测 − 重复成绩": ewma_prev(accB, a) - ewma_prev(wR, a), "B 信念建议成绩 − 习惯建议成绩": ewma_prev(wB, a) - ewma_prev(wH, a),
            "C 只看信念建议成绩": ewma_prev(wB, a), "D 只看习惯建议成绩": ewma_prev(wH, a)}
    for name, Mraw in defs.items():
        M = (Mraw - Mraw.mean()) / Mraw.std()
        r = minimize(nll, [0, 0], args=(M,), method="L-BFGS-B")
        zz = rest + bp * np.exp(r.x[0] * M) + hp * np.exp(r.x[1] * M)
        _, ll = L.marginal_sigma(zz, A)
        LR = 2 * (ll - ll0)
        rows.append(dict(定义=name, 速率=a, θ信念=round(float(r.x[0]), 4), θ惯性=round(float(r.x[1]), 4), 比例变化=round(float(r.x[0] - r.x[1]), 4),
                         LR含σ_df2=round(float(LR), 2), p=float(chi2.sf(max(LR, 0), 2))))
        print(rows[-1], flush=True)
L.save_json(rows, L.OUT / "可靠性_操作化比较.json")
