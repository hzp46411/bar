# -*- coding: utf-8 -*-
"""
第 4 层 · 步骤 3：把"宏观调节双系统比例"放进闭环 ABM
  每个智能体 = 一名被试（联合估计的参数）；调节变量（稳定、偏离、可靠性、轮次）由模拟出的人数与自身经历在线计算。
  比较 M0（无调节）与 RG（主模型）及 HRG（习惯痕迹 + 调节）的宏观指纹；每个人群 1000 次 × 400 轮，含同轮共同冲击 σ。
输出：结果/闭环.json
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.special import expit
from scipy.stats import f as F_dist
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

FIT = lambda n: json.loads((L.OUT / "拟合" / f"{n}.json").read_text(encoding="utf-8"))
SPECS = {"M0": L.Spec(), "RG": L.Spec(ratio=L.MODS, gain=L.MODS), "HRG": L.Spec(ratio=L.MODS, gain=L.MODS, habit=True),
         "H0": L.Spec(habit=True)}
B = 1000
STD = L.load_std()


def acf(x, k):
    x = x - x.mean(); return float(x[k:] @ x[:-k] / (x @ x))


def fingerprint(N, A):
    dev = N - N.mean(); cr = N > L.CAP
    runs, cur = [], 1
    for t in range(1, L.T):
        if cr[t] == cr[t - 1]: cur += 1
        else: runs.append(cur); cur = 1
    runs.append(cur)
    crp = N[:-1] > L.CAP
    D = float(A[:, 1:][:, crp].mean() - A[:, 1:][:, ~crp].mean()) if 0 < crp.mean() < 1 else np.nan
    return dict(mean=float(N.mean()), sd=float(N.std()), acf1=acf(N, 1), acf2=acf(N, 2), acf3=acf(N, 3), acf4=acf(N, 4),
                sq_acf1=acf(dev ** 2, 1), mean_run=float(np.mean(runs)), D=D, switch=float((A[:, 1:] != A[:, :-1]).mean()))


def abm_one(args):
    name, seed = args
    spec, fit = SPECS[name], FIT(name)
    X = np.array(fit["X"]); phi = np.array([fit["shared"][k] for k in spec.names()]); sig = fit["sigma"]
    lam, thR, thG, aH = spec.unpack(phi)
    zs = lambda m, v: (v - STD[m][0]) / STD[m][1]
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    rho, beta, kap, b = expit(X[:, 0]), X[:, 1], X[:, 2], X[:, 3]
    BL = np.full(n, 1 / 3); BH = np.full(n, 1 / 3); H = np.full(n, .5); c = np.zeros(n)
    relB = np.full(n, .5); relH = np.full(n, .5); a_prev = None
    A = np.zeros((n, L.T)); N = np.zeros(L.T); stab = 0.0; rbar = np.zeros(L.T)
    for t in range(L.T):
        lag = (N[t - 1] - L.CAP) / 10 if t > 0 else 0.0
        if t >= 2:
            stab = stab + 1 if (N[t - 1] > L.CAP) == (N[t - 2] > L.CAP) else 0.0
        mods = dict(stab=min(stab, 4), dev=abs(lag), time=t / L.T)
        mrel = zs("rel", relB - relH)
        r = sum((thR[m] * (mrel if m == "rel" else zs(m, mods[m])) for m in thR), np.zeros(n))
        g = sum((thG[m] * (mrel if m == "rel" else zs(m, mods[m])) for m in thG), np.zeros(n))
        z = b + lam * lag + np.exp(g) * (beta * (BL - 0.7 * BH) * np.exp(r / 2) + kap * c * np.exp(-r / 2)) + sig * rng.standard_normal()
        a = (rng.random(n) < expit(z)).astype(float)
        A[:, t] = a; N[t] = a.sum(); rbar[t] = float(np.mean(r))
        oth = N[t] - a
        Gt = (oth <= L.CAP - 1).astype(float); St = (oth >= L.CAP + 1).astype(float)
        relB = (1 - L.REL_RATE) * relB + L.REL_RATE * (1 - (np.abs(Gt - BL) + np.abs(St - BH)) / 2)
        if a_prev is not None:
            relH = (1 - L.REL_RATE) * relH + L.REL_RATE * (a_prev * Gt + (1 - a_prev) * St)
        a_prev = a
        BL += rho * (Gt - BL); BH += rho * (St - BH)
        if spec.habit:
            H += aH * (a - H); c = 2 * H - 1
        else:
            c = 2 * a - 1
    fp = fingerprint(N, A)
    fp["rbar_sd"] = float(rbar[2:].std())                                    # 群体平均比例随时间波动的幅度
    fp["rbar_corr_crowdflip"] = float(np.corrcoef(rbar[3:], (N[2:-1] > L.CAP) != (N[1:-2] > L.CAP))[0, 1]) if spec.ratio else 0.0
    return name, fp


def p2(sim, ob):
    sim = sim[~np.isnan(sim)]; n = len(sim)
    return float(min(1, 2 * min((1 + (sim >= ob).sum()) / (n + 1), (1 + (sim <= ob).sum()) / (n + 1))))


if __name__ == "__main__":
    names = [n for n in SPECS if (L.OUT / "拟合" / f"{n}.json").exists()]
    with Pool(4) as pool:
        res = pool.map(abm_one, [(n, 300000 + 7919 * i + 101 * k) for k, n in enumerate(names) for i in range(B)])
    obs = fingerprint(L.ATT, L.A_REAL)
    keys = list(obs)
    MAC = ["mean", "sd", "acf1", "acf2", "acf3"]                              # 原项目的宏观指纹（联合检验用）
    out = {"真实": obs, "人群": {}}
    for n in names:
        fps = [f for nm, f in res if nm == n]
        arr = {k: np.array([f[k] for f in fps]) for k in fps[0]}
        fp = np.column_stack([arr[k] for k in MAC]); mu, C = fp.mean(0), np.cov(fp.T)
        o = np.array([obs[k] for k in MAC]); d2 = float((o - mu) @ np.linalg.solve(C, o - mu)); k = len(MAC)
        Fs = d2 * B / (B + 1) * (B - k) / (k * (B - 1))
        out["人群"][n] = dict(联合p=float(F_dist.sf(Fs, k, B - k)),
                             **{key: dict(均值=float(np.nanmean(arr[key])), p=p2(arr[key], obs[key]) if key in obs else None) for key in arr})
    L.save_json(out, L.OUT / "闭环.json")
    for n, v in out["人群"].items():
        print(n, "联合p=%.3f" % v["联合p"], {k: round(v[k]["均值"], 3) for k in ("sd", "acf1", "acf4", "sq_acf1", "D", "switch", "rbar_sd")})
    (L.CKPT / "abm_arb.done").write_text("ok")
