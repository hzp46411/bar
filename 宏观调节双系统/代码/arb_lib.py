# -*- coding: utf-8 -*-
"""
第 4 层 · 宏观调节双系统比例：共用代码
=====================================
模型（在原项目"BBL + 惯性 + 共用分级反应 λ"之上，让两个系统的权重随宏观状态变化）：

    z_it = b_i + λ·LAG_t + e^{g_it} · [ β_i·V_it·e^{+r_it/2} + κ_i·c_it·e^{−r_it/2} ]
    r_it = Σ_k θR_k · M_k,it      信念 : 惯性 权重比的对数变化（"比例"）
    g_it = Σ_k θG_k · M_k,it      两个系统共同的增益变化（"增益"）

  r 每增加 1，信念权重 × e^{1/2}、惯性权重 × e^{−1/2}，两者之比 × e；g 每增加 1，两者同时 × e。
  （等价于第 3 层筛查中的 θ信念 = θG + θR/2、θ惯性 = θG − θR/2）

调节变量 M（都只用第 t 轮决策之前可见的信息，按真实数据的均值、标准差标准化，常数固定保存）：
  stab  同一拥挤状态已持续的轮数（上限 4）              —— 公共
  dev   上一轮人数偏离容量的幅度 |N_{t−1} − 60| / 10     —— 公共
  time  轮次 t / 400                                    —— 公共
  rel   信念系统相对惯性系统的近期可靠性（个人）：
          信念可靠性 = 近期"信念对本轮结果的预测准确度" 1 − (|G − B_L| + |S − B_H|)/2 的指数加权均值
          惯性可靠性 = 近期"若重复上一轮选择会不会赢"的指数加权均值
          rel = 信念可靠性 − 惯性可靠性（权重 0.2；Lee, Shimojo & O'Doherty, 2014 的状态预测误差 / 奖励式可靠性）

可选：习惯痕迹（Miller, Shenhav & Ludvig, 2019）—— c 不再是上一轮选择，而是累积的习惯 H：
          H ← H + α_H·(a − H)，c = 2H − 1，α_H 全体共用；α_H = 1 即原模型。
估计：个体参数 (ρ, β, κ, b) 与共用参数 (λ, θR, θG[, α_H]) 交替最大化（不含同轮共同冲击 σ）；
      最后在固定的 logit 上估计 σ，似然比检验以含 σ 的边际似然为主。
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

W = Path(__file__).resolve().parents[1]                     # 工作区根目录
PROJ = W / "原项目" / "BBL惯性建模"
OUT = W / "结果"
CKPT = W / "检查点"
sys.path.insert(0, str(PROJ / "共用代码"))
import model as M                                          # noqa: E402  原项目的读数据函数

T, CAP = M.T, M.CAP
MODS = ["stab", "rel", "dev", "time"]
LABEL = {"stab": "状态持续轮数", "rel": "信念−惯性相对可靠性", "dev": "上轮偏离幅度", "time": "轮次"}
NAMES = ["rho", "beta", "kap", "b"]
LO = np.array([M.BOUNDS[p][0] for p in NAMES]); HI = np.array([M.BOUNDS[p][1] for p in NAMES])
A_REAL, G_REAL, S_REAL, ATT = M.load()
N_SUBJ = A_REAL.shape[0]
REL_RATE = 0.2
GH_NODES, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / GH_W.sum()


# ------------------------------------------------------------------ 公共调节变量
def public_mods(N):
    """由人数序列算出公共调节变量的原始值 (T,)：stab、dev、time，以及 LAG。"""
    crowd = N > CAP
    stab = np.zeros(T)
    for t in range(2, T):
        stab[t] = stab[t - 1] + 1 if crowd[t - 1] == crowd[t - 2] else 0
    stab = np.minimum(stab, 4)
    lag = np.r_[0.0, (N[:-1] - CAP) / 10]
    return dict(stab=stab, dev=np.abs(lag), time=np.arange(T) / T), lag


def base_X():
    """起点：原项目 graded.py 的联合估计（BBLI + 共用 λ）的逐人参数（ρ 换回 logit）。"""
    D = pd.read_csv(PROJ / "2_共同因素层" / "结果" / "fits_BBLI_lam.csv")
    lg = lambda p: np.log(p / (1 - p))
    return np.column_stack([lg(D["BBLI_rho"]), D["BBLI_beta"], D["BBLI_kap"], D["BBLI_b"]])


def base_lambda():
    return json.loads((PROJ / "2_共同因素层" / "结果" / "graded.json").read_text(encoding="utf-8"))["joint"]["lam"]


# ------------------------------------------------------------------ 模型设定与共用参数向量
class Spec:
    """ratio / gain：哪些调节变量进入 r、g；habit：是否用习惯痕迹。共用参数向量 = [λ, θR..., θG..., (logit α_H)]。"""

    def __init__(self, ratio=(), gain=(), habit=False, bonly=(), honly=(), push=()):
        # bonly / honly：只作用于信念权重 / 只作用于惯性权重的调节变量（用于把"比例 + 增益"拆成两个系统各自的检验）
        # push：加法"重复推力" ψ·c·M —— 不论 κ 正负，人人都被推向自己习惯的方向（乘法写法在重复型与交替型之间会相互抵消）
        self.ratio, self.gain, self.habit = list(ratio), list(gain), bool(habit)
        self.bonly, self.honly, self.push = list(bonly), list(honly), list(push)

    @property
    def k(self):
        return 1 + len(self.ratio) + len(self.gain) + len(self.bonly) + len(self.honly) + len(self.push) + int(self.habit)

    def unpack(self, phi):
        i = 1
        thR = dict(zip(self.ratio, phi[i:i + len(self.ratio)])); i += len(self.ratio)
        thG = dict(zip(self.gain, phi[i:i + len(self.gain)])); i += len(self.gain)
        i += len(self.bonly) + len(self.honly) + len(self.push)
        aH = expit(phi[i]) if self.habit else 1.0
        return phi[0], thR, thG, aH

    def unpack_only(self, phi):
        i = 1 + len(self.ratio) + len(self.gain)
        thBo = dict(zip(self.bonly, phi[i:i + len(self.bonly)])); i += len(self.bonly)
        thHo = dict(zip(self.honly, phi[i:i + len(self.honly)]))
        return thBo, thHo

    def unpack_push(self, phi):
        i = 1 + len(self.ratio) + len(self.gain) + len(self.bonly) + len(self.honly)
        return dict(zip(self.push, phi[i:i + len(self.push)]))

    def bounds(self):
        return [(-3, 3)] + [(-3, 3)] * (len(self.ratio) + len(self.gain) + len(self.bonly) + len(self.honly) + len(self.push)) + ([(-7, 7)] if self.habit else [])

    def names(self):
        return (["lam"] + [f"θR_{m}" for m in self.ratio] + [f"θG_{m}" for m in self.gain] + [f"θB_{m}" for m in self.bonly]
                + [f"θH_{m}" for m in self.honly] + [f"ψ_{m}" for m in self.push] + (["logit_aH"] if self.habit else []))

    def to_dict(self):
        d = dict(ratio=self.ratio, gain=self.gain, habit=self.habit)
        if self.bonly or self.honly:
            d.update(bonly=self.bonly, honly=self.honly)
        if self.push:
            d.update(push=self.push)
        return d


def load_std():
    p = OUT / "标准化常数.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # 第一次：用真实数据与起点参数计算并保存
    pm, _ = public_mods(ATT)
    rel = run(base_X(), Spec(), np.r_[base_lambda()], A_REAL, G_REAL, S_REAL, ATT, out="rel", std={"rel": [0, 1]})
    std = {m: [float(np.mean(v)), float(np.std(v))] for m, v in pm.items()}
    std["rel"] = [float(rel.mean()), float(rel.std())]
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(std, ensure_ascii=False, indent=1), encoding="utf-8")
    return std


# ------------------------------------------------------------------ 核心：逐轮运行（似然 / logit / 模拟）
def run(X, spec, phi, A, G, S, N, out="nll", rng=None, std=None, win=(0, T), sigma=0.0, eps=None):
    """
    X: (行, 4) [logit ρ, β, κ, b]；A/G/S: (行, T)；N: 公共人数序列 (T,)（开环：真实人数）
    rng 给定时按模型抽样并写回 A（开环模拟；G、S、N 仍取给定值）。sigma/eps：模拟时加的同轮共同冲击。
    out: "nll" 每行负对数似然；"z" 每人每轮 logit；"rel" 原始相对可靠性；"w" 返回 (r, g)
    """
    std = std or load_std()
    lam, thR, thG, aH = spec.unpack(phi)
    pm, lag = public_mods(N)
    zs = lambda m, v: (v - std[m][0]) / std[m][1]
    r_pub = sum((thR[m] * zs(m, pm[m]) for m in thR if m != "rel"), np.zeros(T))
    g_pub = sum((thG[m] * zs(m, pm[m]) for m in thG if m != "rel"), np.zeros(T))
    tR_rel, tG_rel = thR.get("rel", 0.0), thG.get("rel", 0.0)
    thBo, thHo = spec.unpack_only(phi)
    eB_pub = sum((thBo[m] * zs(m, pm[m]) for m in thBo if m != "rel"), np.zeros(T))
    eH_pub = sum((thHo[m] * zs(m, pm[m]) for m in thHo if m != "rel"), np.zeros(T))
    tB_rel, tH_rel = thBo.get("rel", 0.0), thHo.get("rel", 0.0)
    psi = spec.unpack_push(phi)
    p_pub = sum((psi[m] * zs(m, pm[m]) for m in psi if m != "rel"), np.zeros(T))
    p_rel = psi.get("rel", 0.0)
    use_rel = ("rel" in thR) or ("rel" in thG) or ("rel" in thBo) or ("rel" in thHo) or ("rel" in psi) or out == "rel"
    rows = X.shape[0]
    rho, beta, kap, b = expit(X[:, 0]), X[:, 1], X[:, 2], X[:, 3]
    BL = np.full(rows, 1 / 3); BH = np.full(rows, 1 / 3); H = np.full(rows, 0.5); c = np.zeros(rows)
    relB = np.full(rows, 0.5); relH = np.full(rows, 0.5); a_prev = None
    nll = np.zeros(rows); Z = np.zeros((rows, T)) if out in ("z", "rel", "w") else None
    Rr = np.zeros((rows, T)) if out == "w" else None; Gg = np.zeros((rows, T)) if out == "w" else None
    for t in range(T if rng is not None else win[1]):
        r, g, eB, eH, ps = r_pub[t], g_pub[t], eB_pub[t], eH_pub[t], p_pub[t]
        if use_rel:
            mrel = zs("rel", relB - relH)
            r = r + tR_rel * mrel; g = g + tG_rel * mrel; eB = eB + tB_rel * mrel; eH = eH + tH_rel * mrel; ps = ps + p_rel * mrel
        z = b + lam * lag[t] + np.exp(g) * (beta * (BL - 0.7 * BH) * np.exp(r / 2 + eB) + kap * c * np.exp(-r / 2 + eH)) + ps * c
        if rng is not None:
            zz = z + (sigma * eps[t] if eps is not None else 0.0)
            A[:, t] = rng.random(rows) < expit(zz)
        a = A[:, t]
        if out == "z":
            Z[:, t] = z
        elif out == "rel":
            Z[:, t] = relB - relH
        elif out == "w":
            Rr[:, t] = r; Gg[:, t] = g
        if t >= win[0] and out == "nll":
            nll += np.logaddexp(0, z) - a * z
        # —— 看到本轮结果之后的更新 ——
        relB = (1 - REL_RATE) * relB + REL_RATE * (1 - (np.abs(G[:, t] - BL) + np.abs(S[:, t] - BH)) / 2)
        if a_prev is not None:
            relH = (1 - REL_RATE) * relH + REL_RATE * (a_prev * G[:, t] + (1 - a_prev) * S[:, t])
        a_prev = a.copy()
        BL += rho * (G[:, t] - BL); BH += rho * (S[:, t] - BH)
        if spec.habit:
            H += aH * (a - H); c = 2 * H - 1
        else:
            c = 2 * a - 1
    if out == "nll":
        return nll
    if out == "w":
        return Rr, Gg
    return Z


# ------------------------------------------------------------------ 估计
def basin_starts(X):
    """似然面上常见的几个"盆地"：ρ 很小、β = ±15；ρ 接近 1、β = ±3（κ、b 沿用当前值）。"""
    out = []
    for lr, bt in ((-5, 15), (-5, -15), (5, 3), (5, -3)):
        Z = X.copy(); Z[:, 0], Z[:, 1] = lr, bt; out.append(Z)
    return out


def fit_individuals(spec, phi, A, G, S, N, inits, rand_starts=0, seed=0, std=None):
    """共用参数固定时，逐人最大似然（所有人一起交给优化器，似然相加、互不影响）。返回 X、每人 NLL。"""
    std = std or load_std()
    n, k = A.shape[0], 4
    stack = lambda Mx: np.tile(Mx, (k + 1, 1))
    As, Gs, Ss = stack(A), stack(G), stack(S)

    def objective(x):
        Xs = stack(x.reshape(n, k))
        for j in range(k):
            Xs[(j + 1) * n:(j + 2) * n, j] += 1e-5
        f = run(Xs, spec, phi, As, Gs, Ss, N, std=std).reshape(k + 1, n)
        return f[0].sum(), ((f[1:] - f[0]) / 1e-5).T.ravel()

    rng = np.random.default_rng(seed)
    X0s = list(inits) + [rng.uniform(LO / 2, HI / 2, (n, k)) for _ in range(rand_starts)]
    best_X, best_f = None, np.full(n, np.inf)
    for X0 in X0s:
        res = minimize(objective, np.clip(X0, LO, HI).ravel(), jac=True, method="L-BFGS-B",
                       bounds=list(zip(np.tile(LO, n), np.tile(HI, n))), options={"maxiter": 3000})
        Xc = res.x.reshape(n, k)
        f = run(Xc, spec, phi, A, G, S, N, std=std)
        if best_X is None:
            best_X = Xc.copy()
        better = f < best_f
        best_X[better], best_f[better] = Xc[better], f[better]
    return best_X, best_f


def fit_shared(spec, X, phi0, A, G, S, N, std=None):
    """个体参数固定时，估计共用参数（λ、θR、θG、α_H）。"""
    std = std or load_std()
    f = lambda p: float(run(X, spec, p, A, G, S, N, std=std).sum())
    res = minimize(f, phi0, method="L-BFGS-B", bounds=spec.bounds(), options={"maxiter": 500})
    return res.x, res.fun


def joint_fit(spec, A, G, S, N, X0, phi0, max_rounds=8, tol=0.05, rand_starts=2, seed=0, log=print, std=None, later_basins=2, ckpt=None):
    """交替最大化：个体步（多起点）↔ 共用步，直到总 NLL 的改善 < tol。
    第 1 轮起点 = 当前值 + 4 个盆地起点 + rand_starts 个随机起点；之后各轮 = 当前值 + later_basins 个盆地起点。
    ckpt：每轮结束后把 (X, φ, 历史) 存到该 .npz；再次调用时从最后完成的一轮继续（容器重启后不必从头来）。"""
    std = std or load_std()
    X, phi = X0.copy(), np.array(phi0, float)
    hist = []
    prev = np.inf
    start = 0
    if ckpt is not None and Path(ckpt).exists():
        d = np.load(ckpt)
        X, phi, hist = d["X"], d["phi"], d["hist"].tolist()
        start = len(hist); prev = hist[-1] if hist else np.inf
        log(f"  从检查点继续：已完成 {start} 轮，总 NLL = {prev:.3f}")
        if start >= 2 and hist[-2] - hist[-1] < tol:
            start = max_rounds
    for rd in range(start, max_rounds):
        inits = [X] + (basin_starts(X) if rd == 0 else basin_starts(X)[:later_basins])
        X, f = fit_individuals(spec, phi, A, G, S, N, inits, rand_starts if rd == 0 else 0, seed + rd, std)
        if spec.k > 0:
            phi, tot = fit_shared(spec, X, phi, A, G, S, N, std)
        else:
            tot = f.sum()
        hist.append(float(tot))
        log(f"  第 {rd + 1} 轮：总 NLL = {tot:.3f}  共用 = {np.round(phi, 4).tolist()}")
        if ckpt is not None:
            np.savez(ckpt, X=X, phi=phi, hist=np.array(hist))
        if prev - tot < tol:
            break
        prev = tot
    f = run(X, spec, phi, A, G, S, N, std=std)
    return X, phi, f, hist


def joint_lbfgs(spec, A, G, S, N, X0, phi0, std=None, maxiter=3000, h=1e-5):
    """全参数联合精修：个体参数 (n×4) 与共用参数一起做 L-BFGS-B。
    梯度：个体部分用"k+1 份叠放、各扰动一个参数"的一次前向计算；共用部分每个参数一次前向计算。"""
    std = std or load_std()
    n, k = A.shape[0], 4
    ks = spec.k
    stack = lambda Mx: np.tile(Mx, (k + 1, 1))
    As, Gs, Ss = stack(A), stack(G), stack(S)

    def objective(v):
        X = v[:n * k].reshape(n, k); phi = v[n * k:]
        Xs = stack(X)
        for j in range(k):
            Xs[(j + 1) * n:(j + 2) * n, j] += h
        f = run(Xs, spec, phi, As, Gs, Ss, N, std=std).reshape(k + 1, n)
        f0 = f[0].sum()
        gX = ((f[1:] - f[0]) / h).T.ravel()
        gphi = np.array([(run(X, spec, phi + h * np.eye(ks)[j], A, G, S, N, std=std).sum() - f0) / h for j in range(ks)])
        return f0, np.r_[gX, gphi]

    v0 = np.r_[np.clip(X0, LO, HI).ravel(), phi0]
    bounds = list(zip(np.tile(LO, n), np.tile(HI, n))) + spec.bounds()
    res = minimize(objective, v0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": 1e-13, "gtol": 1e-7})
    X = res.x[:n * k].reshape(n, k); phi = res.x[n * k:]
    f = run(X, spec, phi, A, G, S, N, std=std)
    return X, phi, f, dict(nit=int(res.nit), message=str(res.message))


def marginal_sigma(z, A):
    """在固定的 logit z 上估计同轮共同冲击 σ：按轮对 ε 做 40 点高斯–埃尔米特积分。返回 (σ, 边际对数似然)。"""
    def ll(s):
        zz = z[None] + s * GH_NODES[:, None, None]
        L = (A[None] * zz - np.logaddexp(0, zz)).sum(1); mx = L.max(0)
        return float((mx + np.log(GH_W @ np.exp(L - mx))).sum())
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda s: -ll(s), bounds=(0, 1.5), method="bounded")
    return float(r.x), float(-r.fun)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1,
                               default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)), encoding="utf-8")
