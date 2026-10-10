# -*- coding: utf-8 -*-
"""
第 19 步：第二个系统该写成"选择惯性"，还是"无模型强化学习（MF）"？——逐人模型比较
  背景：两步任务文献的标准写法是"基于模型（MB）+ 无模型（MF）+ 重复倾向"（Daw 等 2011）。
        本任务中，公布人数让每个人都知道没选的那个选项的收益：
          · 用"去 / 不去"两种收益同时更新的学习，就是 BBL（基于信念 / 模型的学习）；
          · MF 只用自己实际得到的收益，更新自己选了的那个选项；
          · 惯性（重复倾向）不看任何收益。
  模型（逐人最大似然，最后 400 轮；与原项目第 1 层口径相同：无共同成分、无 λ；惯性 c = 上一轮 ±1）：
    INERT   κ, b
    BBL     ρ, β, b
    MF      α, β_M, b              Q_去 ← Q_去 + α(G − Q_去)（只在去时）；Q_留 ← Q_留 + α(0.7·S − Q_留)（只在留时）；V_M = Q_去 − Q_留
    BBLI    ρ, β, κ, b             原项目的主模型
    MFI     α, β_M, κ, b
    BBLMF   ρ, β, α, β_M, b
    BBLMFI  ρ, β, α, β_M, κ, b     完整的"MB + MF + 重复倾向"
  嵌套起点保证：大模型的拟合不差于它所嵌套的小模型。
  比较：总 NLL、AIC、BIC（每人 400 个观测）、逐人 BIC 最优的模型、逐人似然比；
        样本外：前 200 轮拟合 → 后 200 轮检验，以及反过来；
        加入 MF 后惯性 κ 是否变小（惯性是否其实是 MF）；MF 的权重 β_M。
  不依赖模型：重复上一轮选择的比例是否取决于上一轮的输赢（MF 预言"赢则留、输则换"，纯惯性预言与输赢无关）。
用法：python3 s19_惯性与MF比较.py
输出：结果/s19_惯性与MF比较.json
"""
import json, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import wilcoxon
import pred_lib as PL

A, G, S = PL.A, PL.G, PL.S
n, T = A.shape
NAMES = {"INERT": ["kap", "b"], "BBL": ["rho", "beta", "b"], "MF": ["alpha", "betaM", "b"],
         "BBLI": ["rho", "beta", "kap", "b"], "MFI": ["alpha", "betaM", "kap", "b"],
         "BBLMF": ["rho", "beta", "alpha", "betaM", "b"], "BBLMFI": ["rho", "beta", "alpha", "betaM", "kap", "b"]}
BOUNDS = {"rho": (-7, 7), "alpha": (-7, 7), "beta": (-20, 20), "betaM": (-20, 20), "kap": (-10, 10), "b": (-8, 8)}
OUT = "s19_惯性与MF比较.json"


def run(model, X, A_, G_, S_, win=(0, T)):
    """逐轮运行，所有行同时计算；返回每行在窗口 win 内的负对数似然。ρ、α 在 logit 尺度。"""
    P = dict(zip(NAMES[model], X.T)); rows = X.shape[0]
    rho = expit(P["rho"]) if "rho" in P else 0.0
    al = expit(P["alpha"]) if "alpha" in P else 0.0
    beta, betaM, kap, b = P.get("beta", 0.0), P.get("betaM", 0.0), P.get("kap", 0.0), P["b"]
    BL = np.full(rows, 1 / 3); BH = np.full(rows, 1 / 3)            # BBL：若去不挤、若不去会挤的信念
    QG = np.full(rows, 1 / 3); QS = np.full(rows, 0.7 / 3)          # MF：去、留两个选项的价值（起点与 BBL 一致）
    c = np.zeros(rows); nll = np.zeros(rows)
    for t in range(win[1]):
        z = b + beta * (BL - 0.7 * BH) + betaM * (QG - QS) + kap * c
        a = A_[:, t]
        if t >= win[0]:
            nll += np.logaddexp(0, z) - a * z
        BL += rho * (G_[:, t] - BL); BH += rho * (S_[:, t] - BH)
        QG += al * a * (G_[:, t] - QG); QS += al * (1 - a) * (0.7 * S_[:, t] - QS)
        c = 2 * a - 1
    return nll


def fit(model, win=(0, T), inits=(), starts=4, seed=0):
    """逐人最大似然（所有人一起交给优化器，梯度用堆叠的前向差分）；起点 = 全 0 + 给定起点 + 随机点，逐人保留最好的。"""
    names = MODELS_K = NAMES[model]; k = len(names)
    lo = np.array([BOUNDS[p][0] for p in names]); hi = np.array([BOUNDS[p][1] for p in names])
    st = lambda M: np.tile(M, (k + 1, 1)); As, Gs, Ss = st(A), st(G), st(S)

    def obj(x):
        Xs = st(x.reshape(n, k))
        for j in range(k):
            Xs[(j + 1) * n:(j + 2) * n, j] += 1e-5
        f = run(model, Xs, As, Gs, Ss, win).reshape(k + 1, n)
        return f[0].sum(), ((f[1:] - f[0]) / 1e-5).T.ravel()
    rng = np.random.default_rng(seed)
    X0s = [np.zeros((n, k)), *inits] + [rng.uniform(lo / 2, hi / 2, (n, k)) for _ in range(starts)]
    bestX, bestf = np.zeros((n, k)), np.full(n, np.inf)
    for X0 in X0s:
        r = minimize(obj, np.clip(X0, lo, hi).ravel(), jac=True, method="L-BFGS-B",
                     bounds=list(zip(np.tile(lo, n), np.tile(hi, n))), options={"maxiter": 3000})
        X = r.x.reshape(n, k); f = run(model, X, A, G, S, win)
        better = f < bestf; bestX[better], bestf[better] = X[better], f[better]
    return bestX, bestf


def embed(X, src, dst, fill=None):
    """把 src 模型的参数放进 dst 模型（缺的参数取 0 或 fill 给的值）。β、β_M、κ 取 0 时，dst 退化为 src。"""
    out = np.zeros((X.shape[0], len(NAMES[dst])))
    for j, p in enumerate(NAMES[dst]):
        if p in NAMES[src]:
            out[:, j] = X[:, NAMES[src].index(p)]
        elif fill and p in fill:
            out[:, j] = fill[p]
    return out


def signs(model):
    """学习率快 / 慢 × 权重正 / 负 的起点（β 与 β_M 都可正可负）。"""
    out = []
    for r in (-2.0, 2.0):
        for w in (-5.0, 5.0):
            fill = {"rho": r, "alpha": r, "beta": w, "betaM": w}
            out.append(np.tile([fill.get(p, 0.0) for p in NAMES[model]], (n, 1)))
    return out


def fit_all(win=(0, T), seed=0, starts=4, warm=None, log=print):
    """按嵌套顺序拟合 7 个模型。warm：另一组拟合（如全样本）作额外起点。"""
    R = {}
    w = lambda m: [warm[m][0]] if warm else []
    t0 = time.time()
    for m in ("INERT", "BBL", "MF"):
        R[m] = fit(m, win, (signs(m) if m != "INERT" else []) + w(m), starts, seed)
        log(f"  {m}：NLL {R[m][1].sum():.1f}（{time.time() - t0:.0f}s）")
    R["BBLI"] = fit("BBLI", win, signs("BBLI") + [embed(R["BBL"][0], "BBL", "BBLI"), embed(R["INERT"][0], "INERT", "BBLI")] + w("BBLI"), starts, seed + 1)
    log(f"  BBLI：NLL {R['BBLI'][1].sum():.1f}（{time.time() - t0:.0f}s）")
    R["MFI"] = fit("MFI", win, signs("MFI") + [embed(R["MF"][0], "MF", "MFI"), embed(R["INERT"][0], "INERT", "MFI")] + w("MFI"), starts, seed + 2)
    log(f"  MFI：NLL {R['MFI'][1].sum():.1f}（{time.time() - t0:.0f}s）")
    R["BBLMF"] = fit("BBLMF", win, [embed(R["BBL"][0], "BBL", "BBLMF", {"alpha": 2.0}), embed(R["MF"][0], "MF", "BBLMF", {"rho": 2.0})] + w("BBLMF"), starts, seed + 3)
    log(f"  BBLMF：NLL {R['BBLMF'][1].sum():.1f}（{time.time() - t0:.0f}s）")
    R["BBLMFI"] = fit("BBLMFI", win, [embed(R["BBLI"][0], "BBLI", "BBLMFI", {"alpha": 2.0}), embed(R["BBLMF"][0], "BBLMF", "BBLMFI"),
                                      embed(R["MFI"][0], "MFI", "BBLMFI", {"rho": 2.0})] + w("BBLMFI"), starts, seed + 4)
    log(f"  BBLMFI：NLL {R['BBLMFI'][1].sum():.1f}（{time.time() - t0:.0f}s）")
    return R


def table(R, nobs=T):
    out = {}
    for m, (X, f) in R.items():
        k = len(NAMES[m])
        out[m] = dict(参数个数=k, NLL=float(f.sum()), AIC=float(2 * f.sum() + 2 * k * n), BIC=float(2 * f.sum() + k * n * np.log(nobs)))
    return out


def per_person(R, a, b_, nobs=T):
    """逐人 BIC 差（b_ − a，负 = b_ 更好）与似然比。"""
    ka, kb = len(NAMES[a]), len(NAMES[b_])
    dB = 2 * (R[b_][1] - R[a][1]) + (kb - ka) * np.log(nobs)
    LR = 2 * (R[a][1] - R[b_][1])
    return dict(BIC差总和=float(dB.sum()), BIC更好的人数=int((dB < 0).sum()), Wilcoxon_p=float(wilcoxon(dB).pvalue),
                LR总和=float(LR.sum()), 似然比显著的人数=int((LR > {1: 3.84, 2: 5.99, 3: 7.81}[kb - ka]).sum()) if kb > ka else None)


if __name__ == "__main__":
    logf = open(PL.W / "日志" / "s19.log", "a", encoding="utf-8")
    log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
    out = {}
    # ---------- 不依赖模型：上一轮输赢与重复
    won = np.where(A == 1, G, S)                       # 去且不挤 → 1 分；留且挤 → 0.7 分
    rep = (A[:, 1:] == A[:, :-1]).astype(float); w0 = won[:, :-1]; a0 = A[:, :-1]
    desc = {}
    for lab, m in (("去 → 赢", (a0 == 1) & (w0 == 1)), ("去 → 输", (a0 == 1) & (w0 == 0)), ("留 → 赢", (a0 == 0) & (w0 == 1)), ("留 → 输", (a0 == 0) & (w0 == 0))):
        r = np.array([rep[i][m[i]].mean() if m[i].sum() >= 5 else np.nan for i in range(n)])
        desc[lab] = dict(重复比例=float(np.nanmean(r)), 人数=int(np.sum(~np.isnan(r))))
    out["不依赖模型：上一轮输赢后的重复比例（逐人均值）"] = desc
    log(f"输赢与重复：{desc}")
    # ---------- 全样本
    log("全样本拟合")
    R = fit_all(log=log)
    out["全样本"] = table(R)
    out["逐人比较"] = {f"{b_} vs {a}": per_person(R, a, b_) for a, b_ in
                     (("BBLI", "BBLMFI"), ("MFI", "BBLMFI"), ("BBLMF", "BBLMFI"), ("BBL", "BBLI"), ("BBL", "BBLMF"), ("MF", "MFI"), ("MFI", "BBLI"))}
    bic = np.column_stack([2 * R[m][1] + len(NAMES[m]) * np.log(T) for m in NAMES])
    out["逐人 BIC 最优的模型"] = {m: int((bic.argmin(1) == j).sum()) for j, m in enumerate(NAMES)}
    # 加入 MF 后惯性是否变小；MF 的权重
    kI = R["BBLI"][0][:, NAMES["BBLI"].index("kap")]; kF = R["BBLMFI"][0][:, NAMES["BBLMFI"].index("kap")]
    bM = R["BBLMFI"][0][:, NAMES["BBLMFI"].index("betaM")]; bB = R["BBLMFI"][0][:, NAMES["BBLMFI"].index("beta")]
    bI = R["BBLI"][0][:, NAMES["BBLI"].index("beta")]
    out["加入 MF 后的参数"] = dict(κ中位数_BBLI=float(np.median(kI)), κ中位数_BBLMFI=float(np.median(kF)),
                              κ相关=float(np.corrcoef(kI, kF)[0, 1]), β相关=float(np.corrcoef(bI, bB)[0, 1]),
                              βM中位数=float(np.median(bM)), βM四分位=[float(np.percentile(bM, 25)), float(np.percentile(bM, 75))],
                              βM为正的人数=int((bM > 0).sum()))
    PL.save(dict(out, X={m: R[m][0] for m in R}, nll_i={m: R[m][1] for m in R}), OUT)
    log(json.dumps({k: out[k] for k in ("全样本", "逐人 BIC 最优的模型", "加入 MF 后的参数")}, ensure_ascii=False))
    # ---------- 样本外：前半拟合 → 后半检验；后半拟合 → 前半检验
    oos = {}
    for lab, tr, te in (("前半拟合 → 后半检验", (0, T // 2), (T // 2, T)), ("后半拟合 → 前半检验", (T // 2, T), (0, T // 2))):
        log(lab)
        Rt = fit_all(tr, seed=10, starts=2, warm=R, log=log)
        te_nll = {m: run(m, Rt[m][0], A, G, S, te) for m in NAMES}
        oos[lab] = {m: float(v.sum()) for m, v in te_nll.items()}
        d = te_nll["BBLMFI"] - te_nll["BBLI"]
        oos[lab + "：BBLMFI − BBLI"] = dict(总和=float(d.sum()), 更好的人数=int((d < 0).sum()), Wilcoxon_p=float(wilcoxon(d).pvalue))
        d2 = te_nll["BBLI"] - te_nll["BBLMF"]
        oos[lab + "：BBLI − BBLMF"] = dict(总和=float(d2.sum()), 更好的人数=int((d2 < 0).sum()), Wilcoxon_p=float(wilcoxon(d2).pvalue))
        PL.save(dict(out, 样本外=oos, X={m: R[m][0] for m in R}, nll_i={m: R[m][1] for m in R}), OUT)
    out["样本外"] = oos
    PL.save(dict(out, X={m: R[m][0] for m in R}, nll_i={m: R[m][1] for m in R}), OUT)
    log(json.dumps(oos, ensure_ascii=False))
    log("完成")
