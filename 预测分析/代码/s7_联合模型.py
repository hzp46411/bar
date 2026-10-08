# -*- coding: utf-8 -*-
"""第 7 步：选择与预测的联合结构模型
  每人每轮一个潜在信念 B_it（预期的本轮总人数）：
      B_it = μ_i + ws_i·sign(N_{t−1} − 60.5) + wm_i·(N_{t−1} − 60)/10 + η_it，      η_it ~ N(0, τ_i²)
  选择（logit）：z_it = b_i + s_i·u(B_it) + κ_i·c_it·e^{δ·偏离_z} + ψ·稳定_z·c_it + λ·LAG_t
      u(B) 两种写法：分级 u = (60.5 − B)/10；类别 u = 1[B ≤ 60.5] − 0.5（挤 / 不挤）
      c_it 为第 4 层主模型的习惯痕迹（α_H 取 HRGPR 的估计）
  预测：P_it = B_it + γ·a_it + ε_it，ε_it ~ N(0, ω_i²)；τ_i² = π·v_i，ω_i² = (1 − π)·v_i（π 全体共用）
  似然：P_it 的边际 N(B̄_it + γa, v_i) × E_{η | P}[P(a | B̄ + η)]（20 点高斯–埃尔米特）
  γ 的识别：习惯、推力、偏离与选择的随机性使"去不去"有一部分与信念无关
      γ ≈ +1：只是把自己算进去；γ < 0：自我合理化（选择支持偏差）；γ > +1：投射（虚假一致）
  估计：个体参数（μ, ws, wm, log v, b, s, κ）与共用参数（γ, logit π, δ, ψ, λ）交替最大化，按轮存检查点。
  结果与局限：类别写法远好于分级写法；但两者的 π 都到边界（报告≈信念），类别写法的 γ 被"整数预测 + 60/61 台阶"卡在 ±0.5 以内，
        恒定平移 γ 的设定不能表达"把报告挪到与选择一致一侧"。γ 改由 s9 / s9b（工具变量）识别，联合模型改为 s10（类别式合理化）。
用法：python3 s7_联合模型.py 分级|类别
输出：结果/s7_联合模型_<写法>.json
"""
import sys, json, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm
import pred_lib as PL
L = PL.L; STD = L.load_std()
MODE = sys.argv[1] if len(sys.argv) > 1 else "分级"
MAXIT = int(sys.argv[2]) if len(sys.argv) > 2 else 12
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
ms = PL.model_states("HRGPR"); C = ms["c"]
NP = PL.NPREV
t0 = 1
SG = np.sign(NP - 60.5)[t0:]; DM = ((NP - 60) / 10)[t0:]
STAB = ((PL.PM["stab"] - STD["stab"][0]) / STD["stab"][1])[t0:]
DEV = ((PL.PM["dev"] - STD["dev"][0]) / STD["dev"][1])[t0:]
LAG = PL.LAG[t0:]
Ya = A[:, t0:]; Yp = np.clip(P[:, t0:], 30, 90); M = OK[:, t0:]; Cc = C[:, t0:]
xg, wg = np.polynomial.hermite_e.hermegauss(20); wg = wg / wg.sum()
IND = ["μ", "ws", "wm", "logv", "b", "s", "κ"]; SH = ["γ", "logitπ", "δ", "ψ", "λ"]


def u_of(B):
    return (60.5 - B) / 10 if MODE == "分级" else (B <= 60.5) - 0.5


def nll_person(x, sh, i):
    mu, ws, wm, logv, b, s, kap = x; g, lp, dlt, psi, lam = sh
    m = M[i]; a = Ya[i, m]; p = Yp[i, m]
    Bbar = mu + ws * SG[m] + wm * DM[m]
    v = np.exp(logv); pi = expit(lp); tau2 = pi * v; om2 = (1 - pi) * v
    r = p - Bbar - g * a
    llp = -0.5 * (np.log(2 * np.pi * v) + r ** 2 / v)
    mcond = tau2 / v * r; vcond = tau2 * om2 / v
    z0 = b + kap * Cc[i, m] * np.exp(dlt * DEV[m]) + psi * STAB[m] * Cc[i, m] + lam * LAG[m]
    if MODE == "分级":
        eta = mcond[None] + np.sqrt(vcond) * xg[:, None]                # (Q, n_obs)
        z = z0[None] + s * u_of(Bbar[None] + eta)
        pa = np.where(a[None] == 1, expit(z), expit(-z))
        lla = np.log(np.clip(wg @ pa, 1e-300, None))
    else:                                                               # 类别：对 1[B ≤ 60.5] 精确积分
        q = norm.cdf((60.5 - Bbar - mcond) / np.sqrt(vcond))            # P(信念为"不挤" | 预测, 选择)
        p1 = np.where(a == 1, expit(z0 + s / 2), expit(-(z0 + s / 2)))
        p0 = np.where(a == 1, expit(z0 - s / 2), expit(-(z0 - s / 2)))
        lla = np.log(np.clip(q * p1 + (1 - q) * p0, 1e-300, None))
    return -float(np.sum(llp + lla))


def fit_ind(X, sh):
    out = X.copy(); f = np.zeros(n)
    for i in range(n):
        best = None
        for x0 in (X[i], np.r_[X[i][:4], 0.0, 1.0, 0.0]):
            r = minimize(nll_person, x0, args=(sh, i), method="L-BFGS-B",
                         bounds=[(30, 90), (-15, 15), (-15, 15), (0, 7), (-8, 8), (-20, 20), (-10, 10)])
            if best is None or r.fun < best.fun: best = r
        out[i], f[i] = best.x, best.fun
    return out, f


def fit_sh(X, sh):
    tot = lambda q: sum(nll_person(X[i], q, i) for i in range(n))
    r = minimize(tot, sh, method="L-BFGS-B", bounds=[(-15, 15), (-6, 6), (-3, 3), (-3, 3), (-3, 3)], options={"maxiter": 200})
    return r.x, r.fun


if __name__ == "__main__":
    ck = PL.W / "检查点" / f"s7_{MODE}.npz"
    logf = open(PL.W / "日志" / f"s7_{MODE}.log", "a", encoding="utf-8"); log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
    if ck.exists():
        d = dict(np.load(ck)); X, sh, it, hist = d["X"], d["sh"], int(d["it"]), list(d["hist"])
        log(f"从检查点继续：第 {it} 轮")
    else:
        X = np.zeros((n, 7))
        for i in range(n):
            m = M[i]; Z = np.column_stack([np.ones(m.sum()), SG[m], DM[m]]); bb, *_ = np.linalg.lstsq(Z, Yp[i, m], rcond=None)
            X[i, :3] = bb; X[i, 3] = np.log(np.var(Yp[i, m] - Z @ bb) + 1); X[i, 4:] = [0.0, 1.0, 0.0]
        sh = np.array([0.0, 0.0, 0.0, 0.2, -0.25]); it = 0; hist = []
    t1 = time.time()
    while it < MAXIT:
        X, f = fit_ind(X, sh)
        sh, ftot = fit_sh(X, sh)
        it += 1; hist.append(ftot)
        np.savez(ck, X=X, sh=sh, it=it, hist=np.array(hist))
        log(f"第 {it} 轮：总 NLL = {ftot:.3f}  共用 = {dict(zip(SH, np.round(sh, 4)))}  用时 {time.time() - t1:.0f}s")
        if len(hist) >= 2 and hist[-2] - hist[-1] < 0.5:
            break
    # 结果
    f_i = np.array([nll_person(X[i], sh, i) for i in range(n)])
    res = dict(写法=MODE, 总NLL=float(f_i.sum()), 共用=dict(zip(SH, sh.tolist())), π=float(expit(sh[1])), 迭代=hist,
               个体参数中位数=dict(zip(IND, np.median(X, 0).tolist())), X=X, nll_i=f_i, k_个体=7, k_共用=5)
    PL.save(res, f"s7_联合模型_{MODE}.json")
    (PL.W / "检查点" / f"s7_{MODE}.done").write_text("ok")
    log("完成")
