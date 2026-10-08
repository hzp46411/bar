# -*- coding: utf-8 -*-
"""第 10 步：选择与预测的联合结构模型（含"类别式合理化"）
  背景：s9 / s9b 的工具变量表明，被习惯推去的人更常报"不挤"（+0.6），而报告只在 57–63 附近被挪动——
        这是把报告挪到与选择一致的一侧（类别式合理化），不是整体少报几个人（恒定平移）。
  模型（每人每轮）：
    信念 B_it ~ N(B̄_it, v_i)，B̄_it = μ_i + ws_i·sign(N_{t−1} − 60.5) + wm_i·(N_{t−1} − 60)/10；信念类别 K = 1[B ≤ 60.5]（不挤）
    选择：P(去 | K) = logit⁻¹(z0 + s_i·(K − ½))，z0 = b_i + κ_i·c·e^{δ·偏离} + ψ·稳定·c + λ·LAG（与第 4 层主模型一致）
    报告：若 K 与选择一致 → 如实报 B；若不一致 → 以概率 ρ 合理化：报一个"与选择一致一侧"的信念值（信念分布在那一侧的截断），否则如实报 B
  似然（R = 1[P ≤ 60]，f = N(P; B̄, v) 的密度，q = P(K = 1)，p1 = P(去 | 不挤)，p0 = P(去 | 挤)）：
    去、报不挤：f·[p1 + ρ·p0·(1 − q)/q]        去、报挤：f·p0·(1 − ρ)
    留、报挤：  f·[(1 − p0) + ρ·(1 − p1)·q/(1 − q)]   留、报不挤：f·(1 − p1)·(1 − ρ)
  变体：ρ共用 | ρ0（如实报告，ρ = 0）| ρ逐人 | s0（信念不影响选择，ρ 共用）| s0ρ逐人（信念不影响选择，ρ 逐人）
  识别：习惯推力改变选择而不改变信念；ρ > 0 时它会改变报告的类别（s9 的工具变量逻辑在似然中的版本）。
用法：python3 s10_合理化联合模型.py <变体> [最大轮数]
输出：结果/s10_合理化联合模型_<变体>.json
"""
import sys, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit
from scipy.stats import norm
import pred_lib as PL
L = PL.L; STD = L.load_std()
VAR = sys.argv[1] if len(sys.argv) > 1 else "ρ共用"
MAXIT = int(sys.argv[2]) if len(sys.argv) > 2 else 30
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
ms = PL.model_states("HRGPR"); C = ms["c"]
NP = PL.NPREV; t0 = 1
SG = np.sign(NP - 60.5)[t0:]; DM = ((NP - 60) / 10)[t0:]
STAB = ((PL.PM["stab"] - STD["stab"][0]) / STD["stab"][1])[t0:]
DEV = ((PL.PM["dev"] - STD["dev"][0]) / STD["dev"][1])[t0:]
LAG = PL.LAG[t0:]
Ya = A[:, t0:]; Yp = np.clip(P[:, t0:], 30, 90); M = OK[:, t0:]; Cc = C[:, t0:]
RI = "ρ逐人" in VAR; S0 = VAR.startswith("s0")                      # 逐人的 ρ；信念不影响选择
IND = ["μ", "ws", "wm", "logv", "b", "s", "κ"] + (["logitρ_i"] if RI else [])
SH = ["logitρ", "δ", "ψ", "λ"]
BND_I = [(30, 90), (-15, 15), (-15, 15), (0, 7), (-8, 8), (-20, 20), (-10, 10)] + ([(-8, 8)] if RI else [])
BND_S = [(-8, 8), (-3, 3), (-3, 3), (-3, 3)]


def ll_obs(x, sh, i, data=None):
    Ya_, Yp_, Cc_, M_ = data or (Ya, Yp, Cc, M)
    mu, ws, wm, logv, b, s, kap = x[:7]; lr, dlt, psi, lam = sh
    if RI: lr = x[7]
    if S0: s = 0.0
    m = M_[i]; a = Ya_[i, m]; p = Yp_[i, m]
    Bbar = mu + ws * SG[m] + wm * DM[m]; sd = np.exp(0.5 * logv)
    lf = norm.logpdf(p, Bbar, sd)
    q = np.clip(norm.cdf((60.5 - Bbar) / sd), 1e-12, 1 - 1e-12)
    z0 = b + kap * Cc_[i, m] * np.exp(dlt * DEV[m]) + psi * STAB[m] * Cc_[i, m] + lam * LAG[m]
    p1 = expit(z0 + s / 2); p0 = expit(z0 - s / 2)
    rho = 0.0 if VAR == "ρ0" else expit(lr)
    R = p <= 60
    l = np.where(a == 1,
                 np.where(R, p1 + rho * p0 * (1 - q) / q, p0 * (1 - rho)),
                 np.where(R, (1 - p1) * (1 - rho), (1 - p0) + rho * (1 - p1) * q / (1 - q)))
    return lf + np.log(np.clip(l, 1e-300, None))


def nll_person(x, sh, i, data=None):
    return -float(np.sum(ll_obs(x, sh, i, data)))


def fit_ind(X, sh, data=None):
    out = X.copy(); f = np.zeros(n)
    for i in range(n):
        best = None
        alt = X[i].copy(); alt[4:7] = [0.0, 1.0, 0.0]
        starts = [X[i], alt]
        if RI:                                                          # ρ_i 的似然可能多峰：再从"几乎不合理化""几乎总合理化"出发
            for r0 in (-4.0, 3.0):
                x0 = X[i].copy(); x0[7] = r0; starts.append(x0)
        for x0 in starts:
            r = minimize(nll_person, x0, args=(sh, i, data), method="L-BFGS-B", bounds=BND_I)
            if best is None or r.fun < best.fun: best = r
        out[i], f[i] = best.x, best.fun
    return out, f


def fit_sh(X, sh, data=None):
    tot = lambda q: sum(nll_person(X[i], q, i, data) for i in range(n))
    r = minimize(tot, sh, method="L-BFGS-B", bounds=BND_S, options={"maxiter": 200})
    return r.x, r.fun


def init_X(data=None):
    Ya_, Yp_, Cc_, M_ = data or (Ya, Yp, Cc, M)
    X = np.zeros((n, len(IND)))
    for i in range(n):
        m = M_[i]; Z = np.column_stack([np.ones(m.sum()), SG[m], DM[m]]); bb, *_ = np.linalg.lstsq(Z, Yp_[i, m], rcond=None)
        X[i, :3] = bb; X[i, 3] = np.log(np.var(Yp_[i, m] - Z @ bb) + 1); X[i, 4:7] = [0.0, 1.0, 0.0]
        if RI: X[i, 7] = 0.0
    return X


def estimate(X, sh, maxit, log=print, ck=None, it=0, hist=None, data=None, tol=0.5):
    hist = hist or []; t1 = time.time()
    while it < maxit:
        X, f = fit_ind(X, sh, data)
        sh, ftot = fit_sh(X, sh, data)
        it += 1; hist.append(ftot)
        if ck is not None: np.savez(ck, X=X, sh=sh, it=it, hist=np.array(hist))
        log(f"第 {it} 轮：总 NLL = {ftot:.3f}  共用 = {dict(zip(SH, np.round(sh, 4)))}  用时 {time.time() - t1:.0f}s")
        if len(hist) >= 2 and hist[-2] - hist[-1] < tol: break
    return X, sh, hist


if __name__ == "__main__":
    ck = PL.W / "检查点" / f"s10_{VAR}.npz"
    logf = open(PL.W / "日志" / f"s10_{VAR}.log", "a", encoding="utf-8"); log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
    if ck.exists():
        d = dict(np.load(ck)); X, sh, it, hist = d["X"], d["sh"], int(d["it"]), list(d["hist"]); log(f"从检查点继续：第 {it} 轮")
    else:
        X = init_X(); sh = np.array([0.0, 0.0, 0.2, -0.25]); it = 0; hist = []
        if VAR == "ρ0": sh[0] = -8.0
    X, sh, hist = estimate(X, sh, MAXIT, log, ck, it, hist)
    f_i = np.array([nll_person(X[i], sh, i) for i in range(n)])
    rho_i = expit(X[:, 7]) if RI else None
    res = dict(变体=VAR, 总NLL=float(f_i.sum()), 共用=dict(zip(SH, sh.tolist())), ρ=float(expit(sh[0])) if (VAR != "ρ0" and not RI) else None,
               迭代=hist, 个体参数中位数=dict(zip(IND, np.median(X, 0).tolist())), X=X, nll_i=f_i,
               k_个体=len(IND) - (1 if S0 else 0), k_共用=len(SH) - (1 if (VAR == "ρ0" or RI) else 0),
               ρ_i=rho_i, 观测数=int(M.sum()))
    PL.save(res, f"s10_合理化联合模型_{VAR}.json")
    (PL.W / "检查点" / f"s10_{VAR}.done").write_text("ok")
    log("完成")
