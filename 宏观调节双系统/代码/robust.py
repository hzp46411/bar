# -*- coding: utf-8 -*-
"""
第 4 层 · 稳健性检验（两步：主模型 HRGPR 的个体参数固定，只重估相关共用参数；含 σ 的边际似然）
  1 偏离 → 惯性减弱：把线性分级反应 λ·LAG 换成分段样条后是否仍成立
  2 稳定推力的选择性：除了"推向习惯方向"，稳定是否也推向"信念建议方向"
  3 结果不敏感性（习惯特征）：稳定推力在上一轮赢 / 输之后是否一样大
  4 经验效应：时钟（轮次）vs 信念 / 习惯系统的累计成绩
  5 可靠性仲裁的逐人分布：每人一个 δB（信念成绩 → 信念权重）、δH（习惯成绩 → 惯性权重）
输出：结果/稳健性.json
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

A, G, S, N = L.A_REAL, L.G_REAL, L.S_REAL, L.ATT
FIT = json.loads((L.OUT / "拟合" / "HRGPR.json").read_text(encoding="utf-8"))
SP = L.Spec(**FIT["spec"]); X = np.array(FIT["X"]); SH = dict(FIT["shared"]); NAMES = SP.names()
STD = L.load_std(); PM, LAG = L.public_mods(N)
Z = lambda m, v: (v - STD[m][0]) / STD[m][1]
n, T = A.shape


def parts(zero=()):
    """把 zero 中的共用参数置 0 后，返回 (其余部分, 信念部分, 惯性部分)；其余部分含 b、λ·LAG、推力。"""
    sh = dict(SH)
    for k in zero:
        sh[k] = 0.0
    phi = np.array([sh[k] for k in NAMES])
    run = lambda Xv: L.run(Xv, SP, phi, A, G, S, N, out="z")
    z = run(X); Xb = X.copy(); Xb[:, 1] = 0; Xk = X.copy(); Xk[:, 2] = 0
    bp = z - run(Xb); hp = z - run(Xk)
    return z - bp - hp, bp, hp


def fit(zfun, k, x0=None):
    nll = lambda p: float((np.logaddexp(0, zfun(p)) - A * zfun(p)).sum())
    r = minimize(nll, np.zeros(k) if x0 is None else x0, method="L-BFGS-B")
    _, ll = L.marginal_sigma(zfun(r.x), A)
    return r.x, ll


# 习惯痕迹方向 c、信念建议方向、上一轮输赢
aH = float(expit(SH["logit_aH"])); H = np.full(n, .5); c = np.zeros((n, T))
for t in range(T):
    c[:, t] = 2 * H - 1; H += aH * (A[:, t] - H)
rho, beta = expit(X[:, 0]), X[:, 1]
BL = np.full(n, 1 / 3); BH = np.full(n, 1 / 3); Vt = np.zeros((n, T))
for t in range(T):
    Vt[:, t] = BL - 0.7 * BH; BL += rho * (G[:, t] - BL); BH += rho * (S[:, t] - BH)
sB_dir = np.sign(beta[:, None] * Vt)                                   # 信念建议：+1 去，−1 不去
win = A * G + (1 - A) * S
won_prev = np.c_[np.zeros((n, 1)), win[:, :-1]]; lost_prev = np.c_[np.zeros((n, 1)), 1 - win[:, :-1]]
Mst = np.broadcast_to(Z("stab", PM["stab"]), (n, T)); Mdev = np.broadcast_to(Z("dev", PM["dev"]), (n, T))
Mtime = np.broadcast_to(Z("time", PM["time"]), (n, T))
out = {}

# ---------- 1 样条分级反应下，偏离对两个系统的作用 ----------
rest, bp, hp = parts(["θR_dev", "θG_dev"])
d = np.r_[0.0, N[:-1] - L.CAP]
basis = np.stack([d / 10, np.maximum(d - 5, 0) / 10, np.maximum(d - 10, 0) / 10, np.maximum(-d - 5, 0) / 10, np.maximum(-d - 10, 0) / 10])
rest_nolam = rest - SH["lam"] * LAG[None]
lin = lambda p: rest_nolam + (p[2] * LAG)[None] + bp * np.exp(p[0] * Mdev) + hp * np.exp(p[1] * Mdev)
spl = lambda p: rest_nolam + (p[2:] @ basis)[None] + bp * np.exp(p[0] * Mdev) + hp * np.exp(p[1] * Mdev)
x_lin, ll_lin = fit(lin, 3, np.r_[0, 0, SH["lam"]])
x_spl, ll_spl = fit(spl, 7, np.r_[0, 0, SH["lam"], 0, 0, 0, 0])
x_spl0, ll_spl0 = fit(lambda p: spl(np.r_[0, p]), 6, np.r_[0, SH["lam"], 0, 0, 0, 0])          # 样条下去掉惯性的偏离作用
out["1 样条分级反应"] = dict(线性λ下=dict(δ信念=round(x_lin[0], 4), δ惯性=round(x_lin[1], 4)),
                          样条λ下=dict(δ信念=round(x_spl[0], 4), δ惯性=round(x_spl[1], 4)),
                          样条相对线性_LR_df4=round(2 * (ll_spl - ll_lin), 2),
                          样条下惯性项_LR_df1=round(2 * (ll_spl - ll_spl0), 2), p=float(chi2.sf(max(2 * (ll_spl - ll_spl0), 0), 1)))
print(out["1 样条分级反应"], flush=True)

# ---------- 2 稳定推力的选择性 ----------
rest, bp, hp = parts(["ψ_stab"])
fH = lambda p: rest + bp + hp + p[0] * c * Mst
fB = lambda p: rest + bp + hp + p[0] * c * Mst + p[1] * sB_dir * Mst
xH, llH = fit(fH, 1, [SH["ψ_stab"]]); xB, llB = fit(fB, 2, [SH["ψ_stab"], 0])
xBo, llBo = fit(lambda p: rest + bp + hp + p[0] * sB_dir * Mst, 1)
out["2 稳定推力选择性"] = dict(习惯方向ψ=round(xB[0], 4), 信念方向ψ=round(xB[1], 4),
                           加信念方向_LR_df1=round(2 * (llB - llH), 2), p=float(chi2.sf(max(2 * (llB - llH), 0), 1)),
                           只用信念方向时ψ=round(xBo[0], 4), 只用信念方向比只用习惯方向的对数似然差=round(llBo - llH, 2))
print(out["2 稳定推力选择性"], flush=True)

# ---------- 3 结果不敏感性 ----------
fWL = lambda p: rest + bp + hp + (p[0] * won_prev + p[1] * lost_prev) * c * Mst
xWL, llWL = fit(fWL, 2, [SH["ψ_stab"]] * 2)
out["3 结果不敏感性"] = dict(赢后ψ=round(xWL[0], 4), 输后ψ=round(xWL[1], 4), 赢输不等_LR_df1=round(2 * (llWL - llH), 2),
                         p=float(chi2.sf(max(2 * (llWL - llH), 0), 1)))
print(out["3 结果不敏感性"], flush=True)

# ---------- 4 经验：时钟 vs 累计成绩 ----------
rest, bp, hp = parts(["θR_time", "θG_time"])
recB = (beta[:, None] * Vt > 0).astype(float); recH = (c > 0).astype(float)
wB = recB * G + (1 - recB) * S; wH = recH * G + (1 - recH) * S
def cum_prev(w, k0=5):
    cs = np.cumsum(w, 1); cnt = np.arange(1, T + 1)
    prev = np.c_[np.zeros((n, 1)), cs[:, :-1]]; prevn = np.r_[0, cnt[:-1]]
    return (k0 * 0.5 + prev) / (k0 + prevn)
cB = cum_prev(wB); cH = cum_prev(wH)
zs_ = lambda v: (v - v.mean()) / v.std()
cB, cH = zs_(cB), zs_(cH)
clock = lambda p: rest + bp * np.exp(p[0] * Mtime) + hp * np.exp(p[1] * Mtime)
cum = lambda p: rest + bp * np.exp(p[0] * cB) + hp * np.exp(p[1] * cH)
both = lambda p: rest + bp * np.exp(p[0] * Mtime + p[2] * cB) + hp * np.exp(p[1] * Mtime + p[3] * cH)
xc, llc = fit(clock, 2); xu, llu = fit(cum, 2); xb, llb = fit(both, 4)
out["4 经验：时钟 vs 累计成绩"] = dict(时钟=dict(δ信念=round(xc[0], 4), δ惯性=round(xc[1], 4), 对数似然=round(llc, 2)),
                                  累计成绩=dict(δ信念_信念成绩=round(xu[0], 4), δ惯性_习惯成绩=round(xu[1], 4), 对数似然=round(llu, 2)),
                                  两者=dict(参数=np.round(xb, 4).tolist(), 对数似然=round(llb, 2)),
                                  时钟之上加累计成绩_LR_df2=round(2 * (llb - llc), 2), 累计成绩之上加时钟_LR_df2=round(2 * (llb - llu), 2),
                                  累计信念成绩与轮次相关=round(float(np.corrcoef(cB.ravel(), Mtime.ravel())[0, 1]), 3))
print(out["4 经验：时钟 vs 累计成绩"], flush=True)

# ---------- 5 可靠性仲裁的逐人分布 ----------
rest, bp, hp = parts(["θB_relB", "θH_relH"])
phi = np.array([SH[k] for k in NAMES])
MB = Z("relB", L.run(X, SP, phi, A, G, S, N, out="relB")); MH = Z("relH", L.run(X, SP, phi, A, G, S, N, out="relH"))
D = np.zeros((n, 2)); f0 = np.zeros(n); f1 = np.zeros(n)
for i in range(n):
    zi = lambda p: rest[i] + bp[i] * np.exp(p[0] * MB[i]) + hp[i] * np.exp(p[1] * MH[i])
    nl = lambda p: float(np.logaddexp(0, zi(p)).sum() - A[i] @ zi(p))
    f0[i] = nl([0, 0]); r = minimize(nl, [0, 0], method="L-BFGS-B", bounds=[(-3, 3)] * 2); D[i], f1[i] = r.x, r.fun
out["5 可靠性逐人"] = dict(δ信念中位数=round(float(np.median(D[:, 0])), 4), δ信念为正比例=round(float(np.mean(D[:, 0] > 0)), 2),
                         δ信念_Wilcoxon_p=float(wilcoxon(D[:, 0]).pvalue),
                         δ惯性中位数=round(float(np.median(D[:, 1])), 4), δ惯性为正比例=round(float(np.mean(D[:, 1] > 0)), 2),
                         δ惯性_Wilcoxon_p=float(wilcoxon(D[:, 1]).pvalue), 单人显著_df2=int((2 * (f0 - f1) > 5.99).sum()))
print(out["5 可靠性逐人"], flush=True)
L.save_json(dict(out, 可靠性逐人δ=D), L.OUT / "稳健性.json")
