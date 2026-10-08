# -*- coding: utf-8 -*-
"""第 14 步：完整的联合模型——第 4 层主模型 HRGPR 的选择方程 + 预测的测量方程（含类别式合理化）
  选择（与 HRGPR 完全相同，另加一条可选的"类别信念"通道，同样受宏观仲裁调节）：
    z(K) = b + λ·LAG + e^g·[β·V·e^{r/2+θB·relB} + s·(K − ½)·e^{r/2+θB·relB} + κ·c·e^{−r/2+θH·relH}] + (ψ_stab·稳定 + ψ_dev·偏离)·c
    V = B_L − 0.7·B_H（BBL，学习率 ρ_i），r = Σ θR·M，g = Σ θG·M（M = 稳定、偏离、轮次），c 为习惯痕迹（α_H）
  信念：B = m_i + k_i·V + η，η ~ N(0, v_i)——同一个 ρ_i 同时决定"信念怎么形成"（预测）与"信念怎么用"（选择），K = 1[B ≤ 60.5]
  预测：信念与选择一致 → 如实报 B；不一致 → 以概率 ρ_f,i 改报与选择一致一侧的值（s10 的类别式合理化）
  变体：J0 测量方程（s = 0：选择就是 HRGPR，预测只是 V 的带偏测量）
        J1 测量 + 类别信念通道（s 逐人自由）
        J2 同 J1，但预测中的信念用自己的学习率 ρᴾ_i（B = m + k·V(ρᴾ)）——与 J1 比较即"两个方程是不是同一个学习过程"
  个体参数 9 个：logit ρ, β, κ, b, m, k, log v, s, logit ρ_f；共用参数 12 个：同 HRGPR
  检验：与 HRGPR（只用选择）比较共用的仲裁参数、λ、ρ_i；选择部分的边际似然是否变差；参数恢复（s14b）
用法：python3 s14_HRGPR预测联合模型.py J0|J1 [最大轮数]
输出：结果/s14_HRGPR预测联合模型_<变体>.json
"""
import sys, time, json
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_ndtr
from scipy.stats import norm
import pred_lib as PL
L = PL.L; STD = L.load_std()
VAR = sys.argv[1] if len(sys.argv) > 1 else "J1"
MAXIT = int(sys.argv[2]) if len(sys.argv) > 2 else 20
f4, SP, X4, PHI4 = PL.fit("HRGPR")
A, G, S, N, n, T = PL.A, PL.G, PL.S, PL.N, PL.n, PL.T
PP = np.where(PL.OK, np.clip(PL.P, 30, 90), 60.0); OKP = PL.OK.copy()
IND = ["logitρ", "β", "κ", "b", "m", "k", "logv", "s", "logitρ_f"] + (["logitρᴾ"] if VAR == "J2" else [])
K = len(IND)
LO = np.r_[L.LO, 30, -60, 0, -20, -8]; HI = np.r_[L.HI, 90, 60, 7, 20, 8]
if VAR == "J2": LO = np.r_[LO, -7]; HI = np.r_[HI, 7]
FREE = np.ones(K, bool)
if VAR == "J0": FREE[7] = False                                         # s ≡ 0


def run_joint(X, phi, A_, P_, OKP_, G_, S_, N_=None, out="nll"):
    """X: (行, 9)。返回每行的 联合 NLL（out='nll'），或 (联合, 只看选择的边际) NLL（out='both'）。"""
    N_ = N if N_ is None else N_
    zs = lambda m, v: (v - STD[m][0]) / STD[m][1]
    lam, thR, thG, aH = SP.unpack(phi); thBo, thHo = SP.unpack_only(phi); psi = SP.unpack_push(phi)
    pm, lag = L.public_mods(N_)
    r_pub = sum((thR[m] * zs(m, pm[m]) for m in thR), np.zeros(T))
    g_pub = sum((thG[m] * zs(m, pm[m]) for m in thG), np.zeros(T))
    p_pub = sum((psi[m] * zs(m, pm[m]) for m in psi), np.zeros(T))
    cB = thBo.get("relB", 0.0); cH = thHo.get("relH", 0.0)
    rows = X.shape[0]
    rho, beta, kap, b = expit(X[:, 0]), X[:, 1], X[:, 2], X[:, 3]
    m_, k_, sd, s, rf = X[:, 4], X[:, 5], np.exp(0.5 * X[:, 6]), X[:, 7], expit(X[:, 8])
    rhoP = expit(X[:, 9]) if X.shape[1] > 9 else None
    BL = np.full(rows, 1 / 3); BH = np.full(rows, 1 / 3); H = np.full(rows, 0.5); c = np.zeros(rows)
    BLp = np.full(rows, 1 / 3); BHp = np.full(rows, 1 / 3)
    sB = np.full(rows, 0.5); sH = np.full(rows, 0.5)
    nll = np.zeros(rows); nlc = np.zeros(rows)
    for t in range(T):
        eB = cB * zs("relB", sB); eH = cH * zs("relH", sH)
        V = BL - 0.7 * BH
        wB = np.exp(g_pub[t] + r_pub[t] / 2 + eB)
        zb = b + lam * lag[t] + wB * beta * V + np.exp(g_pub[t] - r_pub[t] / 2 + eH) * kap * c + p_pub[t] * c
        z1 = zb + 0.5 * wB * s; z0 = zb - 0.5 * wB * s
        Bb = m_ + k_ * (V if rhoP is None else BLp - 0.7 * BHp); u = (60.5 - Bb) / sd
        lq, l1q = log_ndtr(u), log_ndtr(-u); q = np.exp(lq)
        a = A_[:, t]; p = P_[:, t]; okp = OKP_[:, t]
        p1 = expit(z1); p0 = expit(z0)
        pa = np.where(a == 1, q * p1 + (1 - q) * p0, q * (1 - p1) + (1 - q) * (1 - p0))
        nlc -= np.log(np.clip(pa, 1e-300, None))
        R = p <= 60
        # 联合：f(P)·[...]（见 s10），比值写成 exp(log) 以免 q 很小时溢出
        r10 = np.exp(np.clip(l1q - lq, -50, 50)); r01 = 1 / r10
        lik = np.where(a == 1, np.where(R, p1 + rf * p0 * r10, p0 * (1 - rf)),
                       np.where(R, (1 - p1) * (1 - rf), (1 - p0) + rf * (1 - p1) * r01))
        lf = -0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * ((p - Bb) / sd) ** 2
        nll -= np.where(okp, lf + np.log(np.clip(lik, 1e-300, None)), np.log(np.clip(pa, 1e-300, None)))
        # 看到结果之后的更新（与 arb_lib.run 相同）
        recB = (beta * V > 0).astype(float); recH = (c > 0).astype(float)
        sB = (1 - L.REL_RATE) * sB + L.REL_RATE * (recB * G_[:, t] + (1 - recB) * S_[:, t])
        sH = (1 - L.REL_RATE) * sH + L.REL_RATE * (recH * G_[:, t] + (1 - recH) * S_[:, t])
        BL += rho * (G_[:, t] - BL); BH += rho * (S_[:, t] - BH)
        if rhoP is not None:
            BLp += rhoP * (G_[:, t] - BLp); BHp += rhoP * (S_[:, t] - BHp)
        H += aH * (a - H); c = 2 * H - 1
    return (nll, nlc) if out == "both" else nll


def fit_ind(X, phi, data, extra_starts=True):
    A_, P_, OKP_, G_, S_ = data
    idx = np.where(FREE)[0]; kf = len(idx); h = 1e-5
    st = lambda Mx: np.tile(Mx, (kf + 1, 1))
    As, Ps, Os, Gs, Ss = st(A_), st(P_), st(OKP_), st(G_), st(S_)

    def obj(v):
        Xf = X.copy(); Xf[:, idx] = v.reshape(n, kf)
        Xs = st(Xf)
        for j, col in enumerate(idx):
            Xs[(j + 1) * n:(j + 2) * n, col] += h
        f = run_joint(Xs, phi, As, Ps, Os, Gs, Ss).reshape(kf + 1, n)
        return f[0].sum(), ((f[1:] - f[0]) / h).T.ravel()
    inits = [X]
    if extra_starts:
        for lr in (-4.0, 3.0):
            Z = X.copy(); Z[:, 0] = lr; inits.append(Z)
        Z = X.copy(); Z[:, 8] = -X[:, 8]; inits.append(Z)
        if VAR == "J2":
            for lr in (-3.0, 3.0):
                Z = X.copy(); Z[:, 9] = lr; inits.append(Z)
    best, bf = X.copy(), run_joint(X, phi, A_, P_, OKP_, G_, S_)
    for X0 in inits:
        res = minimize(obj, np.clip(X0[:, idx], LO[idx], HI[idx]).ravel(), jac=True, method="L-BFGS-B",
                       bounds=list(zip(np.tile(LO[idx], n), np.tile(HI[idx], n))), options={"maxiter": 3000})
        Xc = X.copy(); Xc[:, idx] = res.x.reshape(n, kf)
        f = run_joint(Xc, phi, A_, P_, OKP_, G_, S_)
        better = f < bf; best[better], bf[better] = Xc[better], f[better]
    return best, bf


def fit_sh(X, phi, data):
    A_, P_, OKP_, G_, S_ = data
    fobj = lambda q: float(run_joint(X, q, A_, P_, OKP_, G_, S_).sum())
    res = minimize(fobj, phi, method="L-BFGS-B", bounds=SP.bounds(), options={"maxiter": 300})
    return res.x, res.fun


def init_X(Xchoice, data, rho_f=None):
    """起点：选择部分取 HRGPR，信念部分由预测对 V 的逐人回归给出，ρ_f 取 s10 ρ逐人。"""
    A_, P_, OKP_, G_, S_ = data
    X = np.zeros((n, K)); X[:, :4] = Xchoice
    rho = expit(Xchoice[:, 0]); BL = np.full(n, 1 / 3); BH = np.full(n, 1 / 3); V = np.zeros((n, T))
    for t in range(T):
        V[:, t] = BL - 0.7 * BH; BL += rho * (G_[:, t] - BL); BH += rho * (S_[:, t] - BH)
    for i in range(n):
        mk = OKP_[i]; Z = np.column_stack([np.ones(mk.sum()), V[i, mk]]); bb, *_ = np.linalg.lstsq(Z, P_[i, mk], rcond=None)
        X[i, 4:6] = bb; X[i, 6] = np.log(np.var(P_[i, mk] - Z @ bb) + 1)
    X[:, 4] = np.clip(X[:, 4], 31, 89); X[:, 5] = np.clip(X[:, 5], -59, 59)
    X[:, 7] = 0.0; X[:, 8] = np.log(rho_f / (1 - rho_f)) if rho_f is not None else 0.0
    return np.clip(X, LO, HI)


def estimate(X, phi, data, maxit, log=print, ck=None, it=0, hist=None, tol=0.5):
    hist = hist or []; t1 = time.time()
    while it < maxit:
        X, f = fit_ind(X, phi, data, extra_starts=(it < 2))
        phi, tot = fit_sh(X, phi, data); it += 1; hist.append(float(tot))
        if ck is not None: np.savez(ck, X=X, phi=phi, it=it, hist=np.array(hist))
        log(f"第 {it} 轮：总 NLL = {tot:.3f}  λ = {phi[0]:.3f}  共用 = {np.round(phi, 3).tolist()}  用时 {time.time() - t1:.0f}s")
        if len(hist) >= 2 and hist[-2] - hist[-1] < tol: break
    return X, phi, hist


DATA = (A, PP, OKP, G, S)

if __name__ == "__main__":
    ck = PL.W / "检查点" / f"s14_{VAR}.npz"
    logf = open(PL.W / "日志" / f"s14_{VAR}.log", "a", encoding="utf-8"); log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
    if ck.exists():
        d = dict(np.load(ck)); X, phi, it, hist = d["X"], d["phi"], int(d["it"]), list(d["hist"]); log(f"从检查点继续：第 {it} 轮")
    elif VAR == "J2" and (PL.OUT / "s14_HRGPR预测联合模型_J1.json").exists():
        d1 = json.loads((PL.OUT / "s14_HRGPR预测联合模型_J1.json").read_text(encoding="utf-8"))   # 从 J1 出发，ρᴾ 起点 = ρ
        X1 = np.array(d1["X"]); X = np.column_stack([X1, X1[:, 0]]); phi = np.array([d1["共用"][k] for k in SP.names()]); it = 0; hist = []
    else:
        d10 = json.loads((PL.OUT / "s10_合理化联合模型_ρ逐人.json").read_text(encoding="utf-8"))
        X = init_X(X4, DATA, rho_f=np.clip(expit(np.array(d10["X"])[:, 7]), 0.01, 0.99)); phi = PHI4.copy(); it = 0; hist = []
    X, phi, hist = estimate(X, phi, DATA, MAXIT, log, ck, it, hist)
    nll, nlc = run_joint(X, phi, *DATA, out="both")
    nll_hrgpr = L.run(X4, SP, PHI4, A, G, S, N, std=STD)
    res = dict(变体=VAR, 总NLL=float(nll.sum()), 选择部分的边际NLL=float(nlc.sum()), HRGPR选择NLL=float(nll_hrgpr.sum()),
               共用=dict(zip(SP.names(), phi.tolist())), HRGPR共用=dict(zip(SP.names(), PHI4.tolist())), 迭代=hist,
               参数名=IND, X=X, nll_i=nll, nlc_i=nlc, 个体参数中位数=dict(zip(IND, np.median(X, 0).tolist())),
               ρ=expit(X[:, 0]), ρ_HRGPR=expit(X4[:, 0]), β_HRGPR=X4[:, 1], k_个体=int(FREE.sum()), k_共用=len(PHI4),
               ρᴾ=expit(X[:, 9]) if VAR == "J2" else None)
    PL.save(res, f"s14_HRGPR预测联合模型_{VAR}.json")
    log("完成")
