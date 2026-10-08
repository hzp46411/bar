# -*- coding: utf-8 -*-
"""第 12 步：预测能不能被 BBL 的信念轨迹解释？——在"类别式合理化"联合框架（s10 ρ逐人）中替换信念的形成方式
  s10 的信念均值只用上一轮：B̄ = μ + ws·sign(N(t−1) − 60.5) + wm·(N(t−1) − 60)/10。这里换成：
    BBL      ：B̄ = m + k·V(ρᴾ)，V = B_L − 0.7·B_H，B_L、B_H 为 G（若去不挤）、S（若不去也挤）的指数加权（学习率 ρᴾ 逐人自由）——符号记忆
    BBL固定ρ ：同上，但 ρᴾ 固定为第 4 层主模型 HRGPR 从选择估出的 ρ_i——"预测里的信念与选择里的信念是不是同一个"
    大小      ：B̄ = m + k·EWMA_ρᴾ((N − 60)/10)——数量记忆
    BBL+上轮  ：B̄ = m + k·V(ρᴾ) + wm·(N(t−1) − 60)/10
  其余（选择：s_i·(K − ½) + 习惯 + 推力 + λ；报告：如实或以 ρ_f,i 合理化）与 s10 ρ逐人 完全相同，样本相同，NLL 可直接比较。
用法：python3 s12_BBL信念与预测.py <信念方式> [最大轮数]
输出：结果/s12_BBL信念与预测_<信念方式>.json
"""
import sys, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.signal import lfilter
from scipy.stats import norm
import pred_lib as PL
L = PL.L; STD = L.load_std()
MODE = sys.argv[1] if len(sys.argv) > 1 else "BBL"
MAXIT = int(sys.argv[2]) if len(sys.argv) > 2 else 30
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
G, S = PL.G, PL.S
ms = PL.model_states("HRGPR"); C = ms["c"]; RHO_CHOICE = ms["rho"]
NP = PL.NPREV; t0 = 1
DM = ((NP - 60) / 10)[t0:]
STAB = ((PL.PM["stab"] - STD["stab"][0]) / STD["stab"][1])[t0:]
DEV = ((PL.PM["dev"] - STD["dev"][0]) / STD["dev"][1])[t0:]
LAG = PL.LAG[t0:]
Ya = A[:, t0:]; Yp = np.clip(P[:, t0:], 30, 90); M = OK[:, t0:]; Cc = C[:, t0:]
NDEV = (N - 60) / 10
BEL = {"BBL": ["m", "k", "logitρᴾ"], "BBL固定ρ": ["m", "k"], "大小": ["m", "k", "logitρᴾ"], "BBL+上轮": ["m", "k", "logitρᴾ", "wm"]}[MODE]
nb = len(BEL)
IND = BEL + ["logv", "b", "s", "κ", "logitρ_f"]
SH = ["δ", "ψ", "λ"]
BND_B = {"m": (30, 90), "k": (-60, 60), "logitρᴾ": (-7, 7), "wm": (-15, 15)}
BND_I = [BND_B[x] for x in BEL] + [(0, 7), (-8, 8), (-20, 20), (-10, 10), (-8, 8)]
BND_S = [(-3, 3), (-3, 3), (-3, 3)]


def ewma_pre(x, r, init):
    """决策前的状态：y_0 = init，y_{t+1} = y_t + r·(x_t − y_t)。"""
    s, _ = lfilter([r], [1, -(1 - r)], x, zi=[(1 - r) * init])
    return np.r_[init, s[:-1]]


def bbar(xb, i):
    if MODE in ("BBL", "BBL固定ρ", "BBL+上轮"):
        r = RHO_CHOICE[i] if MODE == "BBL固定ρ" else expit(xb[2])
        V = ewma_pre(G[i], r, 1 / 3) - 0.7 * ewma_pre(S[i], r, 1 / 3)
        out = xb[0] + xb[1] * V[t0:]
        if MODE == "BBL+上轮": out = out + xb[3] * DM
        return out
    r = expit(xb[2]); E = ewma_pre(NDEV, r, 0.0)
    return xb[0] + xb[1] * E[t0:]


def ll_obs(x, sh, i):
    xb = x[:nb]; logv, b, s, kap, lrf = x[nb:]; dlt, psi, lam = sh
    m = M[i]; a = Ya[i, m]; p = Yp[i, m]
    Bb = bbar(xb, i)[m]; sd = np.exp(0.5 * logv)
    lf = norm.logpdf(p, Bb, sd)
    q = np.clip(norm.cdf((60.5 - Bb) / sd), 1e-12, 1 - 1e-12)
    z0 = b + kap * Cc[i, m] * np.exp(dlt * DEV[m]) + psi * STAB[m] * Cc[i, m] + lam * LAG[m]
    p1 = expit(z0 + s / 2); p0 = expit(z0 - s / 2); rho = expit(lrf)
    R = p <= 60
    l = np.where(a == 1, np.where(R, p1 + rho * p0 * (1 - q) / q, p0 * (1 - rho)),
                 np.where(R, (1 - p1) * (1 - rho), (1 - p0) + rho * (1 - p1) * q / (1 - q)))
    return lf + np.log(np.clip(l, 1e-300, None))


nll_person = lambda x, sh, i: -float(np.sum(ll_obs(x, sh, i)))


def starts(xi):
    out = [xi]
    if "logitρᴾ" in BEL:
        for r0 in (-3.0, 0.0, 3.0):
            z = xi.copy(); z[2] = r0; out.append(z)
    z = xi.copy(); z[nb + 4] = 3.0; out.append(z)                       # 合理化概率的另一个起点
    return out


def fit_ind(X, sh):
    out = X.copy(); f = np.zeros(n)
    for i in range(n):
        best = None
        for x0 in starts(X[i]):
            r = minimize(nll_person, x0, args=(sh, i), method="L-BFGS-B", bounds=BND_I)
            if best is None or r.fun < best.fun: best = r
        out[i], f[i] = best.x, best.fun
    return out, f


def fit_sh(X, sh):
    tot = lambda q: sum(nll_person(X[i], q, i) for i in range(n))
    r = minimize(tot, sh, method="L-BFGS-B", bounds=BND_S, options={"maxiter": 200}); return r.x, r.fun


def init_X():
    """起点：信念部分由预测的简单回归给出，其余取 s10 ρ逐人 的拟合值。"""
    import json
    d10 = json.loads((PL.OUT / "s10_合理化联合模型_ρ逐人.json").read_text(encoding="utf-8")); X10 = np.array(d10["X"])
    X = np.zeros((n, len(IND)))
    for i in range(n):
        xb = np.zeros(nb); xb[0] = 60.0
        if "logitρᴾ" in BEL: xb[2] = 0.0
        X[i, :nb] = xb
        Bb = bbar(xb, i)                                                # 仅为得到 V / E
        base = Bb - xb[0]
        m = M[i]; y = Yp[i, m]
        reg = np.column_stack([np.ones(m.sum()), (bbar(np.r_[0.0, 1.0, xb[2:]] if nb > 2 else np.r_[0.0, 1.0], i))[m]])
        bb, *_ = np.linalg.lstsq(reg, y, rcond=None); X[i, 0], X[i, 1] = bb
        X[i, nb:] = X10[i, 3:8]
    return X


if __name__ == "__main__":
    import json
    ck = PL.W / "检查点" / f"s12_{MODE}.npz"
    logf = open(PL.W / "日志" / f"s12_{MODE}.log", "a", encoding="utf-8"); log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
    if ck.exists():
        d = dict(np.load(ck)); X, sh, it, hist = d["X"], d["sh"], int(d["it"]), list(d["hist"]); log(f"从检查点继续：第 {it} 轮")
    else:
        d10 = json.loads((PL.OUT / "s10_合理化联合模型_ρ逐人.json").read_text(encoding="utf-8"))
        X = init_X(); sh = np.array([d10["共用"][k] for k in SH]); it = 0; hist = []
    t1 = time.time()
    while it < MAXIT:
        X, f = fit_ind(X, sh); sh, ftot = fit_sh(X, sh); it += 1; hist.append(ftot)
        np.savez(ck, X=X, sh=sh, it=it, hist=np.array(hist))
        log(f"第 {it} 轮：总 NLL = {ftot:.3f}  共用 = {dict(zip(SH, np.round(sh, 4)))}  用时 {time.time() - t1:.0f}s")
        if len(hist) >= 2 and hist[-2] - hist[-1] < 0.5: break
    f_i = np.array([nll_person(X[i], sh, i) for i in range(n)])
    res = dict(信念方式=MODE, 总NLL=float(f_i.sum()), 共用=dict(zip(SH, sh.tolist())), 迭代=hist, 参数名=IND,
               个体参数中位数=dict(zip(IND, np.median(X, 0).tolist())), X=X, nll_i=f_i, k_个体=len(IND), k_共用=len(SH), 观测数=int(M.sum()))
    if "logitρᴾ" in BEL:
        rp = expit(X[:, 2]); res["ρᴾ"] = rp; res["ρ_选择（HRGPR）"] = RHO_CHOICE
    PL.save(res, f"s12_BBL信念与预测_{MODE}.json")
    log("完成")
