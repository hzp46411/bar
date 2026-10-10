# -*- coding: utf-8 -*-
"""
第 20 步（快速审计）：选择惯性（I）、无模型强化学习（F）、基于模型的世界模型（B）——2×2×2 共 8 个模型的
                   真实数据拟合、模型恢复与参数恢复
  世界模型（8 个模型都有，用来生成预测）：
    M(s) = 上一轮为 s（挤 / 不挤）时，对本轮人数的预期
    初值 M(不挤) = 60 + δ/2，M(挤) = 60 − δ/2（δ > 0：先验地预期"挤之后不挤"，即反转；δ < 0：外推）
    学习 M(s_{t−1}) ← M(s_{t−1}) + η (N_t − M(s_{t−1}))
    预测 y_t ~ N(M(s_{t−1}), σ²)                      （快速版：不含合理化与"把自己算进去"）
  选择：logit P(去) = b + w_I·c + w_F·(Q_去 − Q_留) + w_B·(1.7p − 0.7)，p = Φ((60.5 − M(s_{t−1})) / σ)
    c = 上一轮自己的选择（±1，第一轮为 0）
    Q 只用自己实际得到的收益更新所选的行动（学习率 α_F）：去且不挤 1 分，不去且挤 0.7 分，其余 0
  8 个模型：I、F、B 各有或无（"0" 只有 b）。η、δ、σ 在所有模型中都估计（由预测约束），只是在不含 B 的模型中不进入选择。
  约束：w_F ≥ 0、w_B ≥ 0（价值越高越去）；w_I 可正可负（重复或交替）。
  步骤：
    真实     逐人最大似然拟合 8 个模型（选择 + 预测的联合似然），嵌套起点保证大模型不差于它嵌套的小模型
    恢复 g   用模型 g 在真实数据上的逐人估计值生成一套假数据（开环：他人的选择与公布人数取真实值），用 8 个模型分别拟合
    汇总     模型恢复：具体模型与 I / F / B 三个因子的混淆矩阵（BIC、AIC）；参数恢复：生成模型拟合自己的数据
用法：python3 s20_三成分八模型审计.py 真实 | 恢复 <模型> | 汇总
输出：结果/s20_真实拟合.json、结果/s20_恢复_<模型>.json、结果/s20_审计汇总.json
"""
import sys, json, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, ndtr
from scipy.stats import spearmanr
import pred_lib as PL

A, G, S, N, n, T = PL.A, PL.G, PL.S, PL.N.astype(float), PL.n, PL.T
P = np.where(PL.OK, PL.P, 60.0); OKP = PL.OK.astype(float)
N_full = PL._raw.iloc[0, 1:].astype(float).values
CROWD_PREV = (np.r_[N_full[-T - 1], N[:-1]] >= 61).astype(float)        # 决策时已知的上一轮状态（1 = 挤）

NAMES = ["b", "wI", "wF", "aF", "wB", "eta", "delta", "lsig"]
LO = np.array([-8, -10, 0, -7, 0, -7, -40, np.log(1.0)])
HI = np.array([8, 10, 20, 7, 20, 7, 40, np.log(30.0)])
MODELS = ["0", "I", "F", "B", "IF", "IB", "FB", "IFB"]
FREE = {m: [0, 5, 6, 7] + ([1] if "I" in m else []) + ([2, 3] if "F" in m else []) + ([4] if "B" in m else []) for m in MODELS}
SUB = {"0": [], "I": ["0"], "F": ["0"], "B": ["0"], "IF": ["I", "F"], "IB": ["I", "B"], "FB": ["F", "B"], "IFB": ["IF", "IB", "FB"]}
DEFAULT = np.array([0.4, 0.5, 1.0, 0.0, 1.0, -2.2, 0.0, np.log(5.0)])


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """X: (行, 8) 完整参数（不用的权重为 0）。out='nll' 返回每行负对数似然；rng 给定时模拟并返回 (选择, 预测)。"""
    rows = X.shape[0]
    G_ = G if G_ is None else G_; S_ = S if S_ is None else S_
    b, wI, wF, aF, wB = X[:, 0], X[:, 1], X[:, 2], expit(X[:, 3]), X[:, 4]
    eta, delta, sig = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7])
    Mn, Mc = 60 + delta / 2, 60 - delta / 2
    QG, QS = np.full(rows, 0.5), np.full(rows, 0.35)
    c = np.zeros(rows); nll = np.zeros(rows)
    if rng is not None:
        A_ = np.zeros((rows, T)); P_ = np.zeros((rows, T))
    for t in range(T):
        sp = CROWD_PREV[t]
        Mp = Mc if sp else Mn
        p = ndtr((60.5 - Mp) / sig)
        z = b + wI * c + wF * (QG - QS) + wB * (1.7 * p - 0.7)
        if rng is not None:
            A_[:, t] = (rng.random(rows) < expit(z)).astype(float)
            P_[:, t] = np.round(Mp + sig * rng.standard_normal(rows))
        a = A_[:, t]
        if rng is None:
            nll += np.logaddexp(0, z) - a * z
            nll += OKP_[:, t] * (0.5 * np.log(2 * np.pi) + np.log(sig) + 0.5 * ((P_[:, t] - Mp) / sig) ** 2)
        if sp:
            Mc = Mc + eta * (N[t] - Mc)
        else:
            Mn = Mn + eta * (N[t] - Mn)
        r = a * G_[:, t] + (1 - a) * 0.7 * S_[:, t]
        QG = QG + aF * a * (r - QG); QS = QS + aF * (1 - a) * (r - QS)
        c = 2 * a - 1
    return (A_, P_) if rng is not None else nll


def fit_model(m, A_, P_, OKP_, inits, seed=0, n_random=1):
    """逐人最大似然；所有人一起交给优化器（梯度用堆叠的前向差分）。返回完整参数 X (n, 8) 与每人 NLL。"""
    idx = FREE[m]; k = len(idx); h = 1e-5
    st = lambda M: np.tile(M, (k + 1, 1)); As, Ps, Os, Gs, Ss = st(A_), st(P_), st(OKP_), st(G), st(S)
    base = np.zeros((n, 8))

    def obj(v):
        X = base.copy(); X[:, idx] = v.reshape(n, k)
        Xs = st(X)
        for j, col in enumerate(idx):
            Xs[(j + 1) * n:(j + 2) * n, col] += h
        f = run(Xs, As, Ps, Os, Gs, Ss).reshape(k + 1, n)
        return f[0].sum(), ((f[1:] - f[0]) / h).T.ravel()
    rng = np.random.default_rng(seed)
    starts = list(inits) + [np.tile(DEFAULT, (n, 1))]
    for _ in range(n_random):
        R = np.tile(DEFAULT, (n, 1)); R[:, idx] = rng.uniform(LO[idx] / 2, HI[idx] / 2, (n, k)); R[:, 7] = rng.uniform(np.log(3), np.log(10), n)
        starts.append(R)
    bestX, bestf = np.zeros((n, 8)), np.full(n, np.inf)
    for X0 in starts:
        X0 = X0.copy(); mask = np.ones(8, bool); mask[idx] = False; X0[:, mask] = 0.0
        res = minimize(obj, np.clip(X0[:, idx], LO[idx], HI[idx]).ravel(), jac=True, method="L-BFGS-B",
                       bounds=list(zip(np.tile(LO[idx], n), np.tile(HI[idx], n))), options={"maxiter": 2000})
        X = base.copy(); X[:, idx] = res.x.reshape(n, k)
        f = run(X, A_, P_, OKP_)
        better = f < bestf; bestX[better], bestf[better] = X[better], f[better]
    return bestX, bestf


def fit_all(A_, P_, OKP_, log=print, seed=0, n_random=1):
    R = {}; t0 = time.time()
    for m in MODELS:
        inits = [R[s][0] for s in SUB[m]]
        if m == "0":                                                # 世界模型：慢 / 中 / 快三种学习率起点
            inits = [np.tile(np.r_[DEFAULT[:5], e, 0.0, np.log(5.0)], (n, 1)) for e in (-4.0, -1.0, 1.5)]
        R[m] = fit_model(m, A_, P_, OKP_, inits, seed, n_random)
        log(f"  {m}：NLL {R[m][1].sum():.1f}（{time.time() - t0:.0f}s）")
    return R


def scores(R, OKP_):
    nobs = T + OKP_.sum(1)
    return {m: dict(NLL=R[m][1], BIC=2 * R[m][1] + len(FREE[m]) * np.log(nobs), AIC=2 * R[m][1] + 2 * len(FREE[m])) for m in R}


if __name__ == "__main__":
    mode = sys.argv[1]
    logf = open(PL.W / "日志" / f"s20_{mode}{'_' + sys.argv[2] if len(sys.argv) > 2 else ''}.log", "a", encoding="utf-8")
    log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
    if mode == "真实":
        R = fit_all(A, P, OKP, log)
        sc = scores(R, OKP)
        out = {"模型": {m: dict(参数个数=len(FREE[m]), NLL=float(sc[m]["NLL"].sum()), AIC=float(sc[m]["AIC"].sum()), BIC=float(sc[m]["BIC"].sum())) for m in MODELS},
               "逐人BIC最优": {m: int(v) for m, v in zip(MODELS, np.bincount(np.argmin(np.column_stack([sc[m]["BIC"] for m in MODELS]), 1), minlength=8))},
               "逐人AIC最优": {m: int(v) for m, v in zip(MODELS, np.bincount(np.argmin(np.column_stack([sc[m]["AIC"] for m in MODELS]), 1), minlength=8))},
               "X": {m: R[m][0] for m in MODELS}, "nll_i": {m: R[m][1] for m in MODELS}}
        PL.save(out, "s20_真实拟合.json")
        log(json.dumps({k: out[k] for k in ("模型", "逐人BIC最优", "逐人AIC最优")}, ensure_ascii=False))
    elif mode == "恢复":
        g = sys.argv[2]
        Xg = np.array(json.loads((PL.OUT / "s20_真实拟合.json").read_text(encoding="utf-8"))["X"][g])
        As, Ps = run(Xg, None, None, None, rng=np.random.default_rng(2000 + MODELS.index(g)))
        log(f"生成模型 {g}：去的比例 {As.mean():.3f}")
        R = fit_all(As, Ps, OKP, log, seed=1, n_random=0)
        sc = scores(R, OKP)
        PL.save(dict(生成模型=g, X真值=Xg, X={m: R[m][0] for m in MODELS}, BIC={m: sc[m]["BIC"] for m in MODELS}, AIC={m: sc[m]["AIC"] for m in MODELS}),
                f"s20_恢复_{g}.json")
        log("完成")
    else:
        conf = {"BIC": np.zeros((8, 8)), "AIC": np.zeros((8, 8))}
        par = {}
        for gi, g in enumerate(MODELS):
            d = json.loads((PL.OUT / f"s20_恢复_{g}.json").read_text(encoding="utf-8"))
            for crit in conf:
                ch = np.argmin(np.column_stack([d[crit][m] for m in MODELS]), 1)
                conf[crit][gi] = np.bincount(ch, minlength=8) / n
            Xt, Xe = np.array(d["X真值"]), np.array(d["X"][g])
            par[g] = {}
            for j in FREE[g]:
                tv, ev = Xt[:, j], Xe[:, j]
                if j in (3, 5):
                    tv, ev = expit(tv), expit(ev)
                par[g][NAMES[j]] = dict(Pearson=float(np.corrcoef(tv, ev)[0, 1]) if tv.std() > 0 else None,
                                        Spearman=float(spearmanr(tv, ev)[0]) if tv.std() > 0 else None)
        fac = {}
        for crit, C in conf.items():
            fac[crit] = {}
            for f in "IFB":
                has_t = np.array([f in m for m in MODELS]); has_e = np.array([f in m for m in MODELS])
                sens = float(C[has_t][:, has_e].sum(1).mean()); fp = float(C[~has_t][:, has_e].sum(1).mean())
                fac[crit][f] = dict(真有时判为有=sens, 真无时判为有=fp)
        out = dict(混淆矩阵={c: {g: dict(zip(MODELS, np.round(C[i], 3).tolist())) for i, g in enumerate(MODELS)} for c, C in conf.items()},
                   对角线={c: dict(zip(MODELS, np.round(np.diag(C), 3).tolist())) for c, C in conf.items()},
                   因子恢复=fac, 参数恢复=par)
        PL.save(out, "s20_审计汇总.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
