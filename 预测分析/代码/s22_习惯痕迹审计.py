# -*- coding: utf-8 -*-
"""
第 22 步：把"一步惯性"换成会衰减的习惯痕迹，检验 MF 在群体层面的优势是否只是在替过简的惯性还账
  背景：s21 中群体层面 IFB 胜 IB（iBIC 差 240），但不依赖模型的回归显示自己 t−3 至 t−5 轮的选择仍有独立的正效应，
        而 s20 的惯性只记上一轮；慢学习率的 MF（只在去时更新 Q_去）可能在冒充长记忆的惯性。
  习惯痕迹：H ← H + α_H (c − H)，c = 本轮选择（±1），H 初值 0；选择中 w_I·c 换成 w_I·H（α_H = 1 即 s20 的一步惯性）
  12 个模型：s20 的 8 个（I = 一步惯性，α_H 固定为 1）+ H、HF、HB、HFB（H = 习惯痕迹，α_H 自由，logit 尺度）
  估计：与 s21 相同的层级估计（经验贝叶斯 EM + 拉普拉斯近似；复用 s21 的 map_fit / hessian / posterior），EM 最多 20 轮
  群体层面的模型比较：iBIC = −2 Σ_i log p(D_i | m) + 2k·log(总观测数)（Huys et al., 2011；每个参数有 μ、σ² 两个超参数）
  起点：真实数据——一步惯性的模型从 s21 的解出发；习惯痕迹的模型从对应一步模型的 s21 解出发，α_H 逐人在网格上取似然最大者
        假数据——每人从该模型真实拟合的群体均值出发（不用个人真值），先验取真实拟合的 (μ, σ²)
  后验预测检查：用 IB、IFB、HB、HFB 的真实拟合值各模拟 20 套数据，做与真实数据相同的 logistic 回归
        （合并所有人、个人固定截距；自己 t−1…t−5 的选择、t−1 与 t−2 的状态、"自己 × 状态"），比较系数
用法：python3 s22_习惯痕迹审计.py 拟合 <数据集> <模型>    数据集 = 真实，或 <生成模型>_<重复号>（用该模型的真实拟合值生成）
      python3 s22_习惯痕迹审计.py 汇总 <数据集>
      python3 s22_习惯痕迹审计.py 预测检查
输出：结果/s22_<数据集>_<模型>.json、结果/s22_<数据集>_汇总.json、结果/s22_预测检查.json
"""
import sys, json, time, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr
import pred_lib as PL


def _load(fname, mod):
    spec = importlib.util.spec_from_file_location(mod, str(pathlib.Path(__file__).with_name(fname)))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    return M


S20 = _load("s20_三成分八模型审计.py", "s20")
S21 = _load("s21_层级估计审计.py", "s21")
n, T, G, S, OKP, CROWD_PREV = S20.n, S20.T, S20.G, S20.S, S20.OKP, S20.CROWD_PREV
NAMES = S20.NAMES + ["aH"]
ONE_STEP = 50.0                                                          # logit α_H = 50 → α_H = 1（一步惯性）
MODELS = S20.MODELS + ["H", "HF", "HB", "HFB"]
FREE = {m: [0, 5, 6, 7] + ([1] if ("I" in m or "H" in m) else []) + ([8] if "H" in m else [])
        + ([2, 3] if "F" in m else []) + ([4] if "B" in m else []) for m in MODELS}
LOGW = S21.LOGW
ULO = np.r_[S21.ULO, -7.0]
UHI = np.r_[S21.UHI, 7.0]
AH_GRID = [-3.0, -1.5, 0.0, 1.0, 2.0, 3.0, 5.0, 7.0]


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """同 s20 的 run，只把上一轮的选择 c 换成习惯痕迹 H。X：(行, 9)，第 9 列为 logit α_H。"""
    rows = X.shape[0]
    G_ = G if G_ is None else G_; S_ = S if S_ is None else S_
    b, wI, wF, aF, wB = X[:, 0], X[:, 1], X[:, 2], expit(X[:, 3]), X[:, 4]
    eta, delta, sig, aH = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7]), expit(X[:, 8])
    Mn, Mc = 60 + delta / 2, 60 - delta / 2
    QG, QS = np.full(rows, 0.5), np.full(rows, 0.35)
    H = np.zeros(rows); nll = np.zeros(rows)
    if rng is not None:
        A_ = np.zeros((rows, T)); P_ = np.zeros((rows, T))
    for t in range(T):
        sp = CROWD_PREV[t]
        Mp = Mc if sp else Mn
        p = ndtr((60.5 - Mp) / sig)
        z = b + wI * H + wF * (QG - QS) + wB * (1.7 * p - 0.7)
        if rng is not None:
            A_[:, t] = (rng.random(rows) < expit(z)).astype(float)
            P_[:, t] = np.round(Mp + sig * rng.standard_normal(rows))
        a = A_[:, t]
        if rng is None:
            nll += np.logaddexp(0, z) - a * z
            nll += OKP_[:, t] * (0.5 * np.log(2 * np.pi) + np.log(sig) + 0.5 * ((P_[:, t] - Mp) / sig) ** 2)
        if sp:
            Mc = Mc + eta * (S20.N[t] - Mc)
        else:
            Mn = Mn + eta * (S20.N[t] - Mn)
        r = a * G_[:, t] + (1 - a) * 0.7 * S_[:, t]
        QG = QG + aF * a * (r - QG); QS = QS + aF * (1 - a) * (r - QS)
        H = H + aH * (2 * a - 1 - H)
    return (A_, P_) if rng is not None else nll


def to_X(U, m):
    X = np.zeros((U.shape[0], 9)); X[:, 8] = ONE_STEP
    for j, col in enumerate(FREE[m]):
        X[:, col] = np.exp(U[:, j]) if col in LOGW else U[:, j]
    return X


def to_U(X, m):
    idx = FREE[m]
    U = np.column_stack([np.log(np.maximum(X[:, c], np.exp(-8))) if c in LOGW else X[:, c] for c in idx])
    return np.clip(U, ULO[idx], UHI[idx])


# s21 的 map_fit / hessian / posterior 在模块全局中查找前向模型与参数表：换成本步的版本
S21.run, S21.FREE, S21.ULO, S21.UHI, S21.to_X, S21.to_U = run, FREE, ULO, UHI, to_X, to_U


def em(m, A_, P_, U, mu, s2, log, max_it=20):
    """与 s21.em 相同，只是起点 (U, μ, σ²) 由调用者给出。"""
    idx = FREE[m]; k = len(idx); t0 = time.time()
    for it in range(max_it):
        U = S21.map_fit(m, U, mu, s2, A_, P_)
        Hs, _ = S21.hessian(m, U, A_, P_)
        C, _ = S21.posterior(Hs, s2)
        mu_new = U.mean(0)
        s2_new = np.clip((U ** 2).mean(0) + np.einsum("nii->ni", C).mean(0) - mu_new ** 2, 1e-4, ((UHI[idx] - ULO[idx]) / 2) ** 2)
        ch = max(np.max(np.abs(mu_new - mu) / np.sqrt(s2)), np.max(np.abs(np.log(s2_new) - np.log(s2))))
        mu, s2 = mu_new, s2_new
        if ch < 0.02:
            break
    U = S21.map_fit(m, U, mu, s2, A_, P_)
    Hs, nll = S21.hessian(m, U, A_, P_)
    C, logdet = S21.posterior(Hs, s2)
    logprior = -0.5 * (((U - mu) ** 2) / s2).sum(1) - 0.5 * np.log(2 * np.pi * s2).sum()
    LE = -nll + logprior + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdet
    log(f"  {m}：EM {it + 1} 轮，NLL {nll.sum():.1f}，证据 {LE.sum():.1f}（{time.time() - t0:.0f}s）")
    return dict(X=to_X(U, m), mu=mu, s2=s2, LE=LE, nll=nll, EM轮数=it + 1)


def pick_aH(X, A_, P_):
    """α_H 的逐人网格起点：其余参数固定，取似然最大的 logit α_H。"""
    f = []
    for a in AH_GRID:
        Xa = X.copy(); Xa[:, 8] = a; f.append(run(Xa, A_, P_, OKP))
    X = X.copy(); X[:, 8] = np.array(AH_GRID)[np.argmin(np.array(f), 0)]
    return X


def dataset(name):
    if name == "真实":
        return S20.A, S20.P, None
    g, rep = name.rsplit("_", 1)
    Xg = np.array(json.loads((PL.OUT / f"s22_真实_{g}.json").read_text(encoding="utf-8"))["X"])
    As, Ps = run(Xg, None, None, None, rng=np.random.default_rng(3000 + 100 * MODELS.index(g) + int(rep)))
    return As, Ps, Xg


def start(name, m, A_, P_):
    if name == "真实":
        base = m.replace("H", "I")
        r = json.loads((PL.OUT / "s21_真实.json").read_text(encoding="utf-8"))["结果"][base]
        X = np.column_stack([np.array(r["X"]), np.full(n, ONE_STEP)])
        if "H" in m:
            X = pick_aH(X, A_, P_)
        U = to_U(X, m)
        return U, U.mean(0), np.maximum(U.var(0), 1e-2)
    r = json.loads((PL.OUT / f"s22_真实_{m}.json").read_text(encoding="utf-8"))
    mu, s2 = np.array(r["mu"]), np.array(r["s2"])
    X = to_X(np.tile(mu, (n, 1)), m)
    if "H" in m:
        X = pick_aH(X, A_, P_)
    return to_U(X, m), mu, s2


LAG_LABELS = [f"自己 t-{k}" for k in range(1, 6)] + ["状态 t-1", "状态 t-2", "自己×状态 t-1", "自己×状态 t-2"]


def lag_coefs(A_, se=False):
    """合并所有人的 logistic 回归（个人固定截距；±1 编码）：自己 t−1…t−5 的选择、t−1 与 t−2 的状态（挤 = +1）及交互。
    逐人回归的人均系数受个别近乎完全分离的人影响太大，故用合并估计。se=True 时另返回按人聚类的 z 值。"""
    import warnings
    import statsmodels.api as sm
    L = 5; t = np.arange(L, T); rows = A_.shape[0]
    c = np.stack([2 * A_[:, t - k] - 1 for k in range(1, L + 1)], -1)
    s = np.stack([np.broadcast_to(2 * CROWD_PREV[t - k + 1] - 1, (rows, len(t))) for k in (1, 2)], -1)
    Z = np.concatenate([c, s, c[..., :2] * s], -1).reshape(-1, 9)
    D = np.kron(np.eye(rows), np.ones((len(t), 1)))
    kw = dict(cov_type="cluster", cov_kwds={"groups": np.repeat(np.arange(rows), len(t))}) if se else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sm.Logit(A_[:, t].reshape(-1), np.hstack([Z, D])).fit(disp=0, method="newton", maxiter=50, **kw)
    return (r.params[:9], r.tvalues[:9]) if se else r.params[:9]


if __name__ == "__main__":
    mode = sys.argv[1]; name = sys.argv[2] if len(sys.argv) > 2 else None
    if mode == "预测检查":
        b, z = lag_coefs(S20.A, se=True)
        res = {"真实": dict(系数=b.round(3).tolist(), z=z.round(2).tolist())}
        for m in ["IB", "IFB", "HB", "HFB"]:
            X = np.array(json.loads((PL.OUT / f"s22_真实_{m}.json").read_text(encoding="utf-8"))["X"])
            sims = [lag_coefs(run(X, None, None, None, rng=np.random.default_rng(5000 + r))[0]) for r in range(20)]
            res[m] = dict(均值=np.mean(sims, 0).round(3).tolist(), 模拟间标准差=np.std(sims, 0).round(3).tolist())
            print(m, flush=True)
        out = dict(变量=LAG_LABELS, **res)
        PL.save(out, "s22_预测检查.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
        sys.exit()
    Ntot = float((T + OKP.sum(1)).sum())
    if mode == "拟合":
        m = sys.argv[3]
        logf = open(PL.W / "日志" / f"s22_{name}_{m}.log", "a", encoding="utf-8")
        log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
        A_, P_, Xg = dataset(name)
        log(f"数据集 {name}，模型 {m}")
        R = em(m, A_, P_, *start(name, m, A_, P_), log)
        PL.save(dict(数据集=name, 模型=m, X真值=Xg, iBIC=float(-2 * R["LE"].sum() + 2 * len(FREE[m]) * np.log(Ntot)), **R),
                f"s22_{name}_{m}.json")
        log("完成")
    else:
        D = {m: json.loads((PL.OUT / f"s22_{name}_{m}.json").read_text(encoding="utf-8")) for m in MODELS}
        ib = {m: D[m]["iBIC"] for m in MODELS}
        best = min(ib, key=ib.get)
        out = dict(数据集=name, iBIC={m: round(v, 1) for m, v in ib.items()}, 相对最优={m: round(v - ib[best], 1) for m, v in ib.items()},
                   最优=best, EM轮数={m: D[m]["EM轮数"] for m in MODELS})
        par = {}
        for m in MODELS:
            X = np.array(D[m]["X"])
            q = lambda v: [round(float(x), 3) for x in np.percentile(v, [25, 50, 75])]
            par[m] = {NAMES[c]: q(expit(X[:, c]) if c in (3, 5, 8) else X[:, c]) for c in FREE[m]}
        out["参数（四分位数；α_F、η、α_H 为概率尺度）"] = par
        PL.save(out, f"s22_{name}_汇总.json")
        print(json.dumps({k: out[k] for k in ("iBIC", "相对最优", "最优", "EM轮数")}, ensure_ascii=False, indent=1))
