# -*- coding: utf-8 -*-
"""
预测数据分析的共用代码
  数据：数据/allpredict.csv（100 人 × 435 轮；第 1 行为公布人数；预测的是本轮总人数，含自己；无奖励；界面只显示上一轮人数）
  流程：每轮 选择 → 预测本轮人数 → 公布本轮人数。所以预测是"选择之后、结果之前"的陈述信念。
  与第 4 层对齐：只用最后 400 轮；选择与模型量（信念 V、习惯痕迹 c、稳定、偏离、成绩）取自 宏观调节双系统 的主模型拟合。
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
W = Path(__file__).resolve().parents[1]
OUT = W / "结果"
L4 = W.parent / "宏观调节双系统" / "代码"
sys.path.insert(0, str(L4))
import arb_lib as L                                   # noqa: E402

T, CAP = L.T, L.CAP
A, G, S, N = L.A_REAL, L.G_REAL, L.S_REAL, L.ATT     # (100, 400) 选择（1 = 去）；N 公布人数
_raw = pd.read_csv(W / "数据" / "allpredict.csv", header=None)
assert np.array_equal(_raw.iloc[0, 1:].astype(float).values[-T:], N), "预测文件的人数行与选择数据不一致"
P = _raw.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").values.astype(float)[:, -T:]   # (100, 400) 预测
OK = ~np.isnan(P)
n = A.shape[0]
NPREV = np.r_[np.nan, N[:-1]]                         # 决策时看到的上一轮人数
PM, LAG = L.public_mods(N)                            # 稳定、偏离、轮次（原始值）


def fit(name="HRGPR"):
    f = json.loads((L.OUT / "拟合" / f"{name}.json").read_text(encoding="utf-8"))
    sp = L.Spec(**f["spec"]); X = np.array(f["X"]); phi = np.array([f["shared"][k] for k in sp.names()])
    return f, sp, X, phi


def model_states(name="HRGPR"):
    """主模型在真实历史上的逐人逐轮量：z（logit）、信念 V、BBL 的 B_L / B_H、习惯痕迹 c、成绩 sB / sH、信念部分与惯性部分。"""
    from scipy.special import expit
    f, sp, X, phi = fit(name)
    z = L.run(X, sp, phi, A, G, S, N, out="z")
    rho, beta = expit(X[:, 0]), X[:, 1]
    lam, thR, thG, aH = sp.unpack(phi)
    BL = np.full(n, 1 / 3); BH = np.full(n, 1 / 3); H = np.full(n, .5)
    V = np.zeros((n, T)); BLs = np.zeros((n, T)); BHs = np.zeros((n, T)); C = np.zeros((n, T))
    for t in range(T):
        V[:, t] = BL - 0.7 * BH; BLs[:, t] = BL; BHs[:, t] = BH; C[:, t] = 2 * H - 1
        BL += rho * (G[:, t] - BL); BH += rho * (S[:, t] - BH); H += aH * (A[:, t] - H)
    Xb = X.copy(); Xb[:, 1] = 0; Xk = X.copy(); Xk[:, 2] = 0
    bel = z - L.run(Xb, sp, phi, A, G, S, N, out="z"); hab = z - L.run(Xk, sp, phi, A, G, S, N, out="z")
    sB = L.run(X, sp, phi, A, G, S, N, out="relB") if "relB" in (sp.bonly + sp.honly + sp.ratio) else None
    sH = L.run(X, sp, phi, A, G, S, N, out="relH") if "relH" in (sp.bonly + sp.honly + sp.ratio) else None
    return dict(z=z, p=expit(z), V=V, BL=BLs, BH=BHs, c=C, bel=bel, hab=hab, sB=sB, sH=sH, X=X, beta=beta, rho=rho, sigma=f["sigma"])


def save(obj, name):
    L.save_json(obj, OUT / name)


# ------------------------------------------------------------------ 回归工具（个人固定效应 + 按人聚类的稳健标准误）
def _demean(M, g):
    M = np.asarray(M, float).copy()
    for k in np.unique(g):
        m = g == k
        M[m] -= M[m].mean(0)
    return M


def _cluster_V(X, u, g, bread):
    meat = np.zeros((X.shape[1], X.shape[1]))
    for k in np.unique(g):
        m = g == k; s = X[m].T @ u[m]; meat += np.outer(s, s)
    G = len(np.unique(g)); nobs, kx = X.shape
    c = G / (G - 1) * (nobs - 1) / (nobs - kx)
    return c * bread @ meat @ bread


def ols_fe(y, X, g, names):
    """y、X 已展平；g 为个人编号。返回系数、聚类标准误、t。"""
    y = _demean(y[:, None], g)[:, 0]; X = _demean(X, g)
    bread = np.linalg.inv(X.T @ X); b = bread @ X.T @ y; u = y - X @ b
    V = _cluster_V(X, u, g, bread); se = np.sqrt(np.diag(V))
    return {nm: dict(b=float(b[j]), se=float(se[j]), t=float(b[j] / se[j])) for j, nm in enumerate(names)}


def iv_fe(y, xend, Xex, Z, g, names_ex):
    """两阶段最小二乘（一个内生变量），个人固定效应，按人聚类。返回内生变量系数与第一阶段的聚类稳健 F。"""
    y = _demean(y[:, None], g)[:, 0]; xend = _demean(xend[:, None], g)[:, 0]; Xex = _demean(Xex, g); Z = _demean(Z, g)
    W1 = np.column_stack([Z, Xex]); br1 = np.linalg.inv(W1.T @ W1); pi = br1 @ W1.T @ xend; v = xend - W1 @ pi
    V1 = _cluster_V(W1, v, g, br1); kz = Z.shape[1]
    Fz = float(pi[:kz] @ np.linalg.solve(V1[:kz, :kz], pi[:kz]) / kz)
    xh = W1 @ pi
    X2 = np.column_stack([xh, Xex]); br2 = np.linalg.inv(X2.T @ X2); b = br2 @ X2.T @ y
    u = y - np.column_stack([xend, Xex]) @ b
    V = _cluster_V(X2, u, g, br2); se = np.sqrt(np.diag(V))
    res = {"内生变量": dict(b=float(b[0]), se=float(se[0]), t=float(b[0] / se[0]))}
    res.update({nm: dict(b=float(b[j + 1]), se=float(se[j + 1])) for j, nm in enumerate(names_ex)})
    res["第一阶段F"] = Fz
    return res
