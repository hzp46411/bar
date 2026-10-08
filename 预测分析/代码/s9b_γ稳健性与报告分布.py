# -*- coding: utf-8 -*-
"""第 9 步（续）：γ 的稳健性检验，以及"被习惯推去 / 推留"时报告的完整分布
  A 稳健性（结果变量 P − 60）
    A1 按 κ̂ 的符号分组：习惯者（κ̂ > 0）与交替者（κ̂ < 0）各自用 c 作工具。
       两组的第一阶段方向相反；若 c 通过"信念的持续性"直接影响预测，两组的 γ 会符号相反；若真是"选择 → 预测"，两组都为负。
    A2 加轮次固定效应（吸收同轮共同的信念冲击）
    A3 控制上一轮的亲身经历（自己上轮去 × 上轮挤）
    A4 滞后工具：κ̂ · c_{t−1}（只用两轮及更早的选择史），同时控制自己上轮的选择
    A5 不控制 c 本身
  B 报告的完整分布（顺从者 = 选择被习惯推动的人）
    对 k = 45…75：结果 1[P ≤ k]·a 以 a 为内生变量 → 去时的累积分布 F1(k)；1[P ≤ k]·(1 − a) 以 (1 − a) 为内生变量 → 留时的 F0(k)
    恒定平移（每人少报 γ）：F1 − F0 在一大段 k 上都为正；类别式合理化（把报告挪到与选择一致的一侧）：F1 − F0 集中在 60 附近
输出：结果/s9b_γ稳健性与报告分布.json
"""
import numpy as np
import pred_lib as PL
from scipy.special import expit
from scipy.optimize import minimize
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
L = PL.L; STD = L.load_std()
ms = PL.model_states("HRGPR"); C = ms["c"]
f4 = PL.fit("HRGPR"); aH = float(expit(f4[0]["shared"]["logit_aH"]))
NP = PL.NPREV; SGN = np.sign(NP - 60.5); DMV = (NP - 60) / 10
zs = lambda m, v: (v - STD[m][0]) / STD[m][1]
stab = zs("stab", PL.PM["stab"]); dev = zs("dev", PL.PM["dev"]); tim = zs("time", PL.PM["time"])
BT = np.zeros((n, T)); h = np.full(n, 60.0)
for t in range(T):
    BT[:, t] = h; pt = np.where(OK[:, t], P[:, t], h); h = h + aH * (pt - h)
BT = (BT - 60) / 10
# 只用两轮及更早选择的习惯痕迹 c⁻：c_{t} 去掉 a_{t−1} 的贡献 ⇒ 用 c_{t−1}
Cm = np.full((n, T), np.nan); Cm[:, 1:] = C[:, :-1]


def kappa_hat(train):
    k = np.zeros(n)
    for i in range(n):
        m = train & OK[i] & ~np.isnan(SGN)
        X = np.column_stack([np.ones(m.sum()), C[i, m], SGN[m], DMV[m]]); a = A[i, m]
        f = lambda q: float(np.sum(np.logaddexp(0, X @ q) - a * (X @ q))) + 0.01 * q @ q
        k[i] = minimize(f, np.zeros(4), method="L-BFGS-B", bounds=[(-10, 10)] * 4).x[1]
    return k


tix = np.arange(T); half = tix < T // 2
kA = kappa_hat(half); kB = kappa_hat(~half)
K = np.zeros((n, T)); K[:, ~half] = kA[:, None]; K[:, half] = kB[:, None]
kfull = kappa_hat(np.ones(T, bool))
t0 = 3
idx = [(i, t) for i in range(n) for t in range(t0, T) if OK[i, t] and OK[i, t - 1] and OK[i, t - 2]]
ii = np.array([x[0] for x in idx]); tt = np.array([x[1] for x in idx])
yN = P[ii, tt] - 60.0; a = A[ii, tt]; c = C[ii, tt]
Xc = np.column_stack([c, SGN[tt], DMV[tt], (P[ii, tt - 1] - 60) / 10, (P[ii, tt - 2] - 60) / 10, BT[ii, tt], stab[tt], dev[tt], tim[tt]])
nm = ["习惯痕迹c", "上轮挤(±1)", "上轮偏离/10", "自己上轮预测/10", "自己上上轮预测/10", "信念痕迹/10", "稳定", "偏离", "轮次"]
Z = (K[ii, tt] * c)[:, None]
out = {}
iv = lambda y, x, Xe, z, sel=slice(None), names=None: PL.iv_fe(y[sel], x[sel], Xe[sel], z[sel], ii[sel], names or [f"x{j}" for j in range(Xe.shape[1])])
short = lambda q: dict(γ=round(q["内生变量"]["b"], 3), se=round(q["内生变量"]["se"], 3), F=round(q["第一阶段F"], 1))
# A1 按 κ̂ 符号分组（用全样本 κ̂ 分组，组内以 c 为工具，c 不再作控制）
Xnc = Xc[:, 1:]
for lab, sel in (("习惯者 κ̂ > 0", kfull[ii] > 0), ("交替者 κ̂ < 0", kfull[ii] < 0)):
    q = iv(yN, a, Xnc, c[:, None], sel); fs = PL.ols_fe(a[sel], np.column_stack([c[sel], Xnc[sel]]), ii[sel], ["c"] + nm[1:])["c"]
    out[f"A1 {lab}"] = dict(short(q), 人数=int(len(np.unique(ii[sel]))), 第一阶段系数=round(fs["b"], 3)); print(lab, out[f"A1 {lab}"], flush=True)
# A2 轮次固定效应
D = np.zeros((len(tt), T)); D[np.arange(len(tt)), tt] = 1; D = D[:, np.unique(tt)[1:]]
Xfe = np.column_stack([Xc[:, :6], D])
out["A2 轮次固定效应"] = short(iv(yN, a, Xfe, Z)); print("A2", out["A2 轮次固定效应"], flush=True)
# A3 上轮亲身经历
exp_ = (A[ii, tt - 1] * SGN[tt])[:, None]
out["A3 控制 上轮去×上轮挤"] = short(iv(yN, a, np.column_stack([Xc, exp_, A[ii, tt - 1]]), Z)); print("A3", out["A3 控制 上轮去×上轮挤"], flush=True)
# A4 滞后工具
Zl = (K[ii, tt] * Cm[ii, tt])[:, None]
out["A4 滞后工具 κ̂·c_{t−1}（控制上轮选择）"] = short(iv(yN, a, np.column_stack([Xc[:, 1:], A[ii, tt - 1], Cm[ii, tt]]), Zl)); print("A4", out["A4 滞后工具 κ̂·c_{t−1}（控制上轮选择）"], flush=True)
# A5 不控制 c
out["A5 不控制 c"] = short(iv(yN, a, Xc[:, 1:], Z)); print("A5", out["A5 不控制 c"], flush=True)
# B 报告分布
ks = list(range(45, 76))
F1, F0, F1se, F0se = [], [], [], []
for k in ks:
    yk = (P[ii, tt] <= k).astype(float)
    q1 = iv(yk * a, a, Xc, Z); q0 = iv(yk * (1 - a), 1 - a, Xc, Z)
    F1.append(q1["内生变量"]["b"]); F0.append(q0["内生变量"]["b"]); F1se.append(q1["内生变量"]["se"]); F0se.append(q0["内生变量"]["se"])
F1, F0 = np.array(F1), np.array(F0)
# 固定效应吸收了水平：用"全体观测"的同一累积概率作锚，报告 F1 − F0 与原始（OLS）差
raw1 = np.array([np.mean(P[ii, tt][a == 1] <= k) for k in ks]); raw0 = np.array([np.mean(P[ii, tt][a == 0] <= k) for k in ks])
dif = []
for k in ks:
    yk = (P[ii, tt] <= k).astype(float); q = iv(yk, a, Xc, Z); dif.append((q["内生变量"]["b"], q["内生变量"]["se"]))
out["B 报告分布"] = dict(k=ks, 顺从者_去减留_累积差=[d[0] for d in dif], se=[d[1] for d in dif],
                     原始_去减留_累积差=(raw1 - raw0).tolist(), 原始_去=raw1.tolist(), 原始_留=raw0.tolist())
print("B", [(k, round(d[0], 2)) for k, d in zip(ks, dif)][::3], flush=True)
# 若是恒定平移 γ：顺从者的累积差 ≈ F(k − γ) − F(k)，用全体分布近似
Fall = np.array([np.mean(P[ii, tt] <= k) for k in range(30, 91)])
def shift_pred(g):
    xs = np.arange(30, 91); return [float(np.interp(k - g, xs, Fall) - np.interp(k, xs, Fall)) for k in ks]
d = np.array([x[0] for x in dif]); se = np.array([x[1] for x in dif])
best = min(np.linspace(-8, 0, 81), key=lambda g: np.sum(((d - np.array(shift_pred(g))) / se) ** 2))
out["B 恒定平移的最佳拟合"] = dict(γ=float(best), 预测=shift_pred(best), 拟合卡方=float(np.sum(((d - np.array(shift_pred(best))) / se) ** 2)))
# 类别式翻转：信念与选择不一致时，以概率 ρ 把报告换成"与选择一致一侧"的一个信念值（按信念分布在那一侧的形状）
#   推导：k ≤ 60 时 F1 − F0 = ρ·F(k)/F(60)；k ≥ 61 时 = ρ·(1 − F(k))/(1 − F(60))；k = 60 处恰为 ρ
F60 = np.mean(P[ii, tt] <= 60)
def flip_pred(r):
    xs = np.arange(30, 91)
    return [float(r * (np.interp(k, xs, Fall) / F60 if k <= 60 else (1 - np.interp(k, xs, Fall)) / (1 - F60))) for k in ks]
bestr = min(np.linspace(0, 1, 201), key=lambda r: np.sum(((d - np.array(flip_pred(r))) / se) ** 2))
out["B 类别翻转的最佳拟合"] = dict(ρ=float(bestr), 预测=flip_pred(bestr), 拟合卡方=float(np.sum(((d - np.array(flip_pred(bestr))) / se) ** 2)))
# 两者并存（近似相加）
grid = [(g, r) for g in np.linspace(-6, 2, 41) for r in np.linspace(0, 1, 51)]
gb, rb = min(grid, key=lambda q: np.sum(((d - np.array(shift_pred(q[0])) - np.array(flip_pred(q[1]))) / se) ** 2))
out["B 平移 + 翻转"] = dict(γ=float(gb), ρ=float(rb), 拟合卡方=float(np.sum(((d - np.array(shift_pred(gb)) - np.array(flip_pred(rb))) / se) ** 2)))
print("平移+翻转", out["B 平移 + 翻转"])
print("平移", best, out["B 恒定平移的最佳拟合"]["拟合卡方"], "翻转", bestr, out["B 类别翻转的最佳拟合"]["拟合卡方"])
PL.save(out, "s9b_γ稳健性与报告分布.json")
