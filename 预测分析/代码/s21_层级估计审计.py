# -*- coding: utf-8 -*-
"""
第 21 步：层级估计（经验贝叶斯 EM + 拉普拉斯近似）下，8 个模型（I / F / B 的 2×2×2）的拟合、模型恢复与参数恢复
  与 s20 完全相同的模型、真实数据与假数据（假数据用 s20 的参数与种子重新生成，逐位一致），只把"逐人最大似然"换成层级估计。
  每个模型 m：个人参数 u_i（无约束尺度）~ N(μ, diag σ²)
    w_F、w_B 取对数（w = e^u ≥ 0）；α_F、η 取 logit；σ 取对数；b、w_I、δ 不变换
    E 步：逐人求后验众数（似然 + 群体先验）；在众数处用数值二阶差分求似然的海森矩阵 H_i（投影为半正定），后验协方差 (H_i + diag 1/σ²)⁻¹
    M 步：μ = 众数的均值；σ² = 众数的方差 + 后验方差的均值（下限 1e-4）
    迭代至 max(|Δμ|/σ, |Δlog σ²|) < 0.02 或 12 轮；σ² 的上限为参数范围一半的平方
  每人的模型证据（拉普拉斯）：log p(D_i | m) ≈ −NLL_i + log N(u_i; μ, σ²) + (k/2)·log 2π − ½·log det(H_i + diag 1/σ²)
  模型选择：逐人取证据最大的模型（各模型先验相等）；另报告三种模型空间：8 个模型、三个单成分（I / F / B）、惯性 + MB 的 2×2（0 / I / B / IB）
用法：python3 s21_层级估计审计.py 真实 | 恢复 <模型> | 汇总
输出：结果/s21_<数据集>.json、结果/s21_审计汇总.json
"""
import sys, json, time, importlib.util, pathlib
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s20", str(pathlib.Path(__file__).with_name("s20_三成分八模型审计.py")))
S20 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S20)
n, T, MODELS, FREE, run = S20.n, S20.T, S20.MODELS, S20.FREE, S20.run
G, S, OKP = S20.G, S20.S, S20.OKP
NAMES = S20.NAMES
LOGW = {2, 4}                                                            # w_F、w_B：对数尺度
ULO = np.array([-8, -10, -8, -7, -8, -7, -40, 0.0])
UHI = np.array([8, 10, np.log(20), 7, np.log(20), 7, 40, np.log(30)])
SPACES = {"8 个模型": MODELS, "三个单成分": ["I", "F", "B"], "惯性+MB 的 2×2": ["0", "I", "B", "IB"]}


def to_X(U, m):
    idx = FREE[m]; X = np.zeros((U.shape[0], 8))
    for j, col in enumerate(idx):
        X[:, col] = np.exp(U[:, j]) if col in LOGW else U[:, j]
    return X


def to_U(X, m):
    idx = FREE[m]
    U = np.column_stack([np.log(np.maximum(X[:, c], np.exp(-8))) if c in LOGW else X[:, c] for c in idx])
    return np.clip(U, ULO[idx], UHI[idx])


def map_fit(m, U0, mu, s2, A_, P_):
    """逐人后验众数（所有人一起优化，梯度用堆叠的前向差分 + 先验的解析梯度）。"""
    idx = FREE[m]; k = len(idx); h = 1e-5
    st = lambda M_: np.tile(M_, (k + 1, 1)); As, Ps, Os, Gs, Ss = st(A_), st(P_), st(OKP), st(G), st(S)

    def obj(v):
        U = v.reshape(n, k); Us = st(U)
        for j in range(k):
            Us[(j + 1) * n:(j + 2) * n, j] += h
        f = run(to_X(Us, m), As, Ps, Os, Gs, Ss).reshape(k + 1, n)
        g = ((f[1:] - f[0]) / h).T
        return f[0].sum() + 0.5 * ((U - mu) ** 2 / s2).sum(), (g + (U - mu) / s2).ravel()
    lo, hi = ULO[idx], UHI[idx]
    res = minimize(obj, np.clip(U0, lo, hi).ravel(), jac=True, method="L-BFGS-B",
                   bounds=list(zip(np.tile(lo, n), np.tile(hi, n))), options={"maxiter": 2000})
    return res.x.reshape(n, k)


def hessian(m, U, A_, P_, h=1e-2):
    """似然对 u 的海森矩阵（中心二阶差分，所有扰动点堆叠在一次前向计算中）。返回 H (n, k, k) 与众数处的 NLL。"""
    k = U.shape[1]
    pts = [np.zeros(k)]
    for j in range(k):
        e = np.zeros(k); e[j] = h; pts += [e, -e]
    pairs = [(j, l) for j in range(k) for l in range(j + 1, k)]
    for j, l in pairs:
        for sj in (1, -1):
            for sl in (1, -1):
                e = np.zeros(k); e[j] = sj * h; e[l] = sl * h; pts.append(e)
    Pn = len(pts)
    st = lambda M_: np.tile(M_, (Pn, 1))
    f = run(to_X(np.concatenate([U + p for p in pts]), m), st(A_), st(P_), st(OKP), st(G), st(S)).reshape(Pn, n)
    H = np.zeros((n, k, k)); f0 = f[0]
    for j in range(k):
        H[:, j, j] = (f[1 + 2 * j] - 2 * f0 + f[2 + 2 * j]) / h ** 2
    b0 = 1 + 2 * k
    for q, (j, l) in enumerate(pairs):
        H[:, j, l] = H[:, l, j] = (f[b0 + 4 * q] - f[b0 + 4 * q + 1] - f[b0 + 4 * q + 2] + f[b0 + 4 * q + 3]) / (4 * h * h)
    return H, f0


def posterior(H, s2):
    """似然的海森矩阵先投影为半正定（负特征值置 0，避免边界处的负曲率），再加先验精度：后验方差不会大于先验方差。"""
    w, V = np.linalg.eigh(0.5 * (H + np.transpose(H, (0, 2, 1))))
    Hp = np.einsum("nij,nj,nkj->nik", V, np.maximum(w, 0.0), V)
    w2, V2 = np.linalg.eigh(Hp + np.diag(1 / s2)[None])
    C = np.einsum("nij,nj,nkj->nik", V2, 1 / w2, V2)
    return C, np.log(w2).sum(1)


def em(m, A_, P_, X0, log, max_it=12):
    idx = FREE[m]; k = len(idx)
    U = to_U(X0, m); mu = U.mean(0); s2 = np.maximum(U.var(0), 1e-2)
    t0 = time.time()
    for it in range(max_it):
        U = map_fit(m, U, mu, s2, A_, P_)
        H, _ = hessian(m, U, A_, P_)
        C, _ = posterior(H, s2)
        mu_new = U.mean(0)
        s2_new = np.clip((U ** 2).mean(0) + np.einsum("nii->ni", C).mean(0) - mu_new ** 2, 1e-4, ((UHI[idx] - ULO[idx]) / 2) ** 2)
        ch = max(np.max(np.abs(mu_new - mu) / np.sqrt(s2)), np.max(np.abs(np.log(s2_new) - np.log(s2))))
        mu, s2 = mu_new, s2_new
        if ch < 0.02:
            break
    U = map_fit(m, U, mu, s2, A_, P_)
    H, nll = hessian(m, U, A_, P_)
    C, logdet = posterior(H, s2)
    logprior = -0.5 * (((U - mu) ** 2) / s2).sum(1) - 0.5 * np.log(2 * np.pi * s2).sum()
    LE = -nll + logprior + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdet
    log(f"  {m}：EM {it + 1} 轮，NLL {nll.sum():.1f}，证据 {LE.sum():.1f}（{time.time() - t0:.0f}s）")
    return dict(U=U, X=to_X(U, m), mu=mu, s2=s2, LE=LE, nll=nll, EM轮数=it + 1)


def dataset(name):
    """真实数据，或用 s20 的参数与种子重新生成的假数据（逐位一致）；以及 s20 的逐人最大似然解（作 EM 起点）。"""
    real = json.loads((PL.OUT / "s20_真实拟合.json").read_text(encoding="utf-8"))
    if name == "真实":
        return S20.A, S20.P, {m: np.array(real["X"][m]) for m in MODELS}, None
    Xg = np.array(real["X"][name])
    As, Ps = run(Xg, None, None, None, rng=np.random.default_rng(2000 + MODELS.index(name)))
    d = json.loads((PL.OUT / f"s20_恢复_{name}.json").read_text(encoding="utf-8"))
    X_mle = {m: np.array(d["X"][m]) for m in MODELS}
    # 核对：假数据与 s20 一致（生成模型的 MLE 解在新数据上的 BIC 与 s20 保存的相同）
    nobs = T + OKP.sum(1)
    bic = 2 * run(X_mle[name], As, Ps, OKP) + len(FREE[name]) * np.log(nobs)
    assert np.allclose(bic, np.array(d["BIC"][name]), atol=1e-6), "假数据与 s20 不一致"
    return As, Ps, X_mle, Xg


if __name__ == "__main__":
    mode = sys.argv[1]; name = "真实" if mode == "真实" else sys.argv[2] if mode == "恢复" else None
    if name is not None:
        logf = open(PL.W / "日志" / f"s21_{name}.log", "a", encoding="utf-8")
        log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
        A_, P_, X_mle, Xg = dataset(name)
        log(f"数据集 {name}")
        R = {m: em(m, A_, P_, X_mle[m], log) for m in MODELS}
        PL.save(dict(数据集=name, X真值=Xg, 结果={m: dict(X=r["X"], mu=r["mu"], s2=r["s2"], LE=r["LE"], nll=r["nll"], EM轮数=r["EM轮数"]) for m, r in R.items()}),
                f"s21_{name}.json")
        log("完成")
    else:
        D = {g: json.loads((PL.OUT / f"s21_{g}.json").read_text(encoding="utf-8")) for g in MODELS}
        real = json.loads((PL.OUT / "s21_真实.json").read_text(encoding="utf-8"))
        out = {}
        for sp, sub in SPACES.items():
            C = np.zeros((len(sub), len(sub)))
            for i, g in enumerate(sub):
                ch = np.argmax(np.column_stack([D[g]["结果"][m]["LE"] for m in sub]), 1); C[i] = np.bincount(ch, minlength=len(sub)) / n
            inv = C / np.maximum(C.sum(0, keepdims=True), 1e-12)
            fac = {}
            for f in "IFB":
                has = np.array([f in m for m in sub])
                if 0 < has.sum() < len(sub):
                    fac[f] = dict(真有时判为有=float(C[has][:, has].sum(1).mean()), 真无时判为有=float(C[~has][:, has].sum(1).mean()),
                                  选中含时为真=float(C[has][:, has].sum() / max(C[:, has].sum(), 1e-12)))
            LEr = np.column_stack([real["结果"][m]["LE"] for m in sub])
            out[sp] = dict(混淆矩阵={g: dict(zip(sub, np.round(C[i], 3).tolist())) for i, g in enumerate(sub)},
                           对角线=dict(zip(sub, np.round(np.diag(C), 3).tolist())), 反演对角线=dict(zip(sub, np.round(np.diag(inv), 3).tolist())),
                           因子恢复=fac, 真实数据_证据总和=dict(zip(sub, np.round(LEr.sum(0), 1).tolist())),
                           真实数据_逐人最优=dict(zip(sub, np.bincount(LEr.argmax(1), minlength=len(sub)).tolist())))
        par = {}
        for g in MODELS:
            Xt, Xe = np.array(D[g]["X真值"]), np.array(D[g]["结果"][g]["X"])
            par[g] = {}
            for c in FREE[g]:
                tv, ev = Xt[:, c], Xe[:, c]
                if c in (3, 5):
                    tv, ev = 1 / (1 + np.exp(-tv)), 1 / (1 + np.exp(-ev))
                par[g][NAMES[c]] = dict(Pearson=float(np.corrcoef(tv, ev)[0, 1]), Spearman=float(spearmanr(tv, ev)[0]))
        out["参数恢复"] = par
        PL.save(out, "s21_审计汇总.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
