# -*- coding: utf-8 -*-
"""第 13 步（续）：把"4 轮前人数"加进主模型，看 ACF4 的缺口能否补上
  1 两步估计：HRGPR 的 logit 固定，加 δ4·(N(t−4) − 60)/10（全体共用），含同轮共同冲击 σ 的边际似然比检验
  2 闭环：HRGPR（含 σ）与 HRGPR + δ4 各模拟 1000 次 × 400 轮，比较 ACF1–ACF4、SD 与真实值
输出：结果/s13b_四轮项与闭环ACF4.json
"""
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
import pred_lib as PL
L = PL.L; STD = L.load_std()
A, N, T = PL.A, PL.N, PL.T
ms = PL.model_states("HRGPR"); z = ms["z"]
f, sp, X, phi = PL.fit("HRGPR"); sig = f["sigma"]
lag4 = np.r_[np.zeros(4), (N[:-4] - 60) / 10]
s0, ll0 = L.marginal_sigma(z, A)
r = minimize_scalar(lambda d: -L.marginal_sigma(z + d * lag4[None], A)[1], bounds=(-1, 1), method="bounded")
d4 = float(r.x); s1, ll1 = L.marginal_sigma(z + d4 * lag4[None], A)
out = {"两步": dict(δ4=d4, σ_前=s0, σ_后=s1, LR=float(2 * (ll1 - ll0)))}
print(out, flush=True)


def simulate(seed, d4):
    lam, thR, thG, aH = sp.unpack(phi); thBo, thHo = sp.unpack_only(phi); psi = sp.unpack_push(phi)
    zs = lambda m, v: (v - STD[m][0]) / STD[m][1]
    rng = np.random.default_rng(seed); n = X.shape[0]
    rho, beta, kap, b = expit(X[:, 0]), X[:, 1], X[:, 2], X[:, 3]
    BL = np.full(n, 1 / 3); BH = np.full(n, 1 / 3); H = np.full(n, .5); c = np.zeros(n)
    sB = np.full(n, .5); sH = np.full(n, .5); Ns = np.zeros(T); stab = 0.0
    for t in range(T):
        lag = (Ns[t - 1] - 60) / 10 if t > 0 else 0.0
        if t >= 2: stab = stab + 1 if (Ns[t - 1] > 60) == (Ns[t - 2] > 60) else 0.0
        mods = dict(stab=min(stab, 4), dev=abs(lag), time=t / T)
        rr = sum(thR[m] * zs(m, mods[m]) for m in thR); g = sum(thG[m] * zs(m, mods[m]) for m in thG)
        push = sum(psi[m] * zs(m, mods[m]) for m in psi)
        eB = thBo.get("relB", 0) * zs("relB", sB); eH = thHo.get("relH", 0) * zs("relH", sH)
        V = BL - 0.7 * BH
        l4 = (Ns[t - 4] - 60) / 10 if t >= 4 else 0.0
        zz = b + lam * lag + d4 * l4 + np.exp(g) * (beta * V * np.exp(rr / 2 + eB) + kap * c * np.exp(-rr / 2 + eH)) + push * c + sig * rng.standard_normal()
        recB = (beta * V > 0).astype(float); recH = (c > 0).astype(float)
        a = (rng.random(n) < expit(zz)).astype(float); Ns[t] = a.sum()
        oth = Ns[t] - a; Gt = (oth <= 59).astype(float); St = (oth >= 61).astype(float)
        sB = (1 - L.REL_RATE) * sB + L.REL_RATE * (recB * Gt + (1 - recB) * St)
        sH = (1 - L.REL_RATE) * sH + L.REL_RATE * (recH * Gt + (1 - recH) * St)
        BL += rho * (Gt - BL); BH += rho * (St - BH); H += aH * (a - H); c = 2 * H - 1
    return Ns


acf = lambda x, k: float(((x - x.mean())[k:] @ (x - x.mean())[:-k]) / ((x - x.mean()) @ (x - x.mean())))
real = dict(sd=float(N.std()), **{f"acf{k}": acf(N, k) for k in range(1, 5)})
out["真实"] = real
B = 1000
for lab, dd in (("HRGPR", 0.0), ("HRGPR + δ4", d4)):
    fp = []
    for sd in range(B):
        Ns = simulate(500000 + sd, dd); fp.append([Ns.std()] + [acf(Ns, k) for k in range(1, 5)])
    fp = np.array(fp); keys = ["sd", "acf1", "acf2", "acf3", "acf4"]
    res = {}
    for j, k in enumerate(keys):
        v = fp[:, j]; o = real[k]
        p = float(min(1, 2 * min((1 + (v >= o).sum()) / (B + 1), (1 + (v <= o).sum()) / (B + 1))))
        res[k] = dict(均值=float(v.mean()), SD=float(v.std()), p=p)
    out[lab] = res; print(lab, {k: (round(v["均值"], 3), round(v["p"], 3)) for k, v in res.items()}, flush=True)
PL.save(out, "s13b_四轮项与闭环ACF4.json")
