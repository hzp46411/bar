# -*- coding: utf-8 -*-
"""
第 29 步：对上一轮人数大小的分级反应
  背景：s27 的 ABM 复现了平均人数、挤的比例、效率、换选择率与个体差异，但人数的 ACF1 太弱（模拟 −.16，真实 −.35）、
        ACF2 / ACF4 不对；开环残差 N_t − Σp_it 与上一轮人数偏离 60 的大小负相关（HDW：r = −.26），
        且在"挤"与"不挤"各自内部都成立——人不只看挤不挤，还看挤了多少。第 4 层主模型中的 λ·lag 正是这一项。
  两种放法（lag = (N_{t−1} − 60) / 10）：
    R  只进入选择：z += λ·lag（λ < 0：越挤越不去）
    G  进入世界模型：E_t = μ_t + (δ/2)(1 − 2 s_{t−1}) + κ·lag，同时影响预测与选择
  模型：HDWR、HDWG（在 s26 的 HDW 上）；HDWLR、HDWLG（在 s26 的 HDWL 上）
  估计与比较：与 s22–s26 相同（层级 EM 最多 20 轮；iBIC）
  起点：对应模型在 s26 的解；λ 或 κ 逐人取网格
用法：python3 s29_分级反应.py 拟合 <模型> | 汇总
输出：结果/s29_真实_<模型>.json、结果/s29_真实_汇总.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s26", str(pathlib.Path(__file__).with_name("s26_两系统简化.py")))
S26 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S26)
S22, S21, S20 = S26.S22, S26.S21, S26.S20
n, T, G, S, OKP, CROWD_PREV, ONE_STEP = S26.n, S26.T, S26.G, S26.S, S26.OKP, S26.CROWD_PREV, S26.ONE_STEP
N_FULL = S20.N_full
LAG = (np.r_[N_FULL[-T - 1], S20.N[:-1]] - 60) / 10                      # 决策时已知的上一轮人数偏离
BASE = {"HDW": [0, 5, 6, 7, 1, 8, 9, 10, 4], "HDWL": [0, 5, 6, 7, 1, 8, 9, 10, 2, 3, 4]}
NEW = ["HDWR", "HDWG", "HDWLR", "HDWLG"]
FREE = {m: BASE[m[:-1]] + ([13] if m.endswith("R") else [14]) for m in NEW}
FIXED_dE = {m: (50.0 if "L" in m else -50.0) for m in NEW}
ULO = np.r_[S26.ULO, 0.0, -10.0, -40.0]                                 # 第 13 列（世界模型类型）不是自由参数，占位
UHI = np.r_[S26.UHI, 1.0, 10.0, 40.0]
GRID = {13: [-1.5, -0.75, 0.0, 0.75], 14: [-6.0, -3.0, 0.0, 3.0, 6.0]}


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """同 s26 的 run（世界模型均为 W），另加 λ·lag（第 14 列）与世界模型中的 κ·lag（第 15 列）。X：(行, 15)。"""
    rows = X.shape[0]
    G_ = G if G_ is None else G_; S_ = S if S_ is None else S_
    b, wI, wF, aF, wB = X[:, 0], X[:, 1], X[:, 2], expit(X[:, 3]), X[:, 4]
    eta, delta, sig, aH = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7]), expit(X[:, 8])
    wD, aD, dE, lam, kap = X[:, 9], aH * expit(X[:, 10]), expit(X[:, 11]), X[:, 13], X[:, 14]
    mu = np.full(rows, 60.0)
    QG, QS = np.full(rows, 0.5), np.full(rows, 0.35)
    H = np.zeros(rows); D = np.zeros(rows); nll = np.zeros(rows)
    if rng is not None:
        A_ = np.zeros((rows, T)); P_ = np.zeros((rows, T))
    for t in range(T):
        sp = CROWD_PREV[t]
        Mp = mu + delta / 2 * (1 - 2 * sp) + kap * LAG[t]
        p = ndtr((60.5 - Mp) / sig)
        z = b + wI * H + wD * D + wF * (QG - QS) + wB * (1.7 * p - 0.7) + lam * LAG[t]
        if rng is not None:
            A_[:, t] = (rng.random(rows) < expit(z)).astype(float)
            P_[:, t] = np.round(Mp + sig * rng.standard_normal(rows))
        a = A_[:, t]
        if rng is None:
            nll += np.logaddexp(0, z) - a * z
            nll += OKP_[:, t] * (0.5 * np.log(2 * np.pi) + np.log(sig) + 0.5 * ((P_[:, t] - Mp) / sig) ** 2)
        mu = mu + eta * (S20.N[t] - mu)
        QG = QG + aF * (a + (1 - a) * dE) * (G_[:, t] - QG)
        QS = QS + aF * ((1 - a) + a * dE) * (0.7 * S_[:, t] - QS)
        H = H + aH * (2 * a - 1 - H)
        D = D + aD * (2 * a - 1 - D)
    return (A_, P_) if rng is not None else nll


def to_X(U, m):
    X = np.zeros((U.shape[0], 15)); X[:, 8] = ONE_STEP; X[:, 11] = FIXED_dE[m]; X[:, 12] = 1.0
    for j, col in enumerate(FREE[m]):
        X[:, col] = np.exp(U[:, j]) if col in S22.LOGW else U[:, j]
    return X


def to_U(X, m):
    idx = FREE[m]
    U = np.column_stack([np.log(np.maximum(X[:, c], np.exp(-8))) if c in S22.LOGW else X[:, c] for c in idx])
    return np.clip(U, ULO[idx], UHI[idx])


S21.run, S21.FREE, S21.ULO, S21.UHI, S21.to_X, S21.to_U = run, FREE, ULO, UHI, to_X, to_U
S22.FREE, S22.ULO, S22.UHI, S22.to_X = FREE, ULO, UHI, to_X


def start(m):
    X13 = np.array(json.loads((PL.OUT / f"s26_真实_{m[:-1]}.json").read_text(encoding="utf-8"))["X"])
    X = np.zeros((n, 15)); X[:, :13] = X13
    col = FREE[m][-1]; f = []
    for v in GRID[col]:
        Xa = X.copy(); Xa[:, col] = v; f.append(run(Xa, S20.A, S20.P, OKP))
    X[:, col] = np.array(GRID[col])[np.argmin(np.array(f), 0)]
    U = to_U(X, m)
    return U, U.mean(0), np.maximum(U.var(0), 1e-2)


if __name__ == "__main__":
    mode = sys.argv[1]
    Ntot = float((T + OKP.sum(1)).sum())
    if mode == "拟合":
        m = sys.argv[2]
        logf = open(PL.W / "日志" / f"s29_真实_{m}.log", "a", encoding="utf-8")
        log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
        log(f"数据集 真实，模型 {m}")
        R = S22.em(m, S20.A, S20.P, *start(m), log)
        PL.save(dict(数据集="真实", 模型=m, iBIC=float(-2 * R["LE"].sum() + 2 * len(FREE[m]) * np.log(Ntot)), **R), f"s29_真实_{m}.json")
        log("完成")
    else:
        src = {"HDW": "s26_真实_HDW.json", "HDWL": "s26_真实_HDWL.json", **{m: f"s29_真实_{m}.json" for m in NEW}}
        D = {m: json.loads((PL.OUT / f).read_text(encoding="utf-8")) for m, f in src.items()}
        ib = {m: D[m]["iBIC"] for m in D}; best = min(ib, key=ib.get)
        q = lambda v: [round(float(x), 3) for x in np.percentile(v, [25, 50, 75])]
        out = dict(iBIC={m: round(v, 1) for m, v in ib.items()}, 相对最优={m: round(v - ib[best], 1) for m, v in ib.items()}, 最优=best,
                   EM轮数={m: D[m]["EM轮数"] for m in D},
                   关键比较={"HDWR − HDW": round(ib["HDWR"] - ib["HDW"], 1), "HDWG − HDW": round(ib["HDWG"] - ib["HDW"], 1),
                         "HDWLR − HDWL": round(ib["HDWLR"] - ib["HDWL"], 1), "HDWLG − HDWL": round(ib["HDWLG"] - ib["HDWL"], 1)},
                   λ与κ={m: q(np.array(D[m]["X"])[:, FREE[m][-1]]) for m in NEW})
        PL.save(out, "s29_真实_汇总.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
