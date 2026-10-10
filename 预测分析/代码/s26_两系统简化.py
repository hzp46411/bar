# -*- coding: utf-8 -*-
"""
第 26 步：把主模型简化回两个系统（习惯 + 世界模型）
  背景：s25 的最优模型 HDLB 在两个系统内部各有两部分：习惯痕迹 H + 慢漂移 D；状态预测 B + 近期信念 L。
        D 与 L 都是在探索中加入的，且都是慢变量。
  检验 1：D 是否多余 → HLB（HDLB 去掉 D）
  检验 2：B 与 L 能否合成一个世界模型 W（同时进入选择与预测，参数个数与 B 相同：η、δ、σ）
    预期人数 E_t = μ_t + (δ/2)·(1 − 2·s_{t−1})，s = 1 为上轮挤（δ > 0：反转；δ < 0：外推）
    平均水平 μ 每轮更新：μ ← μ + η (N_t − μ)，初值 60
    预测 y_t ~ N(E_t, σ²)；选择中 w_B·(1.7p − 0.7)，p = Φ((60.5 − E_t)/σ)
    （B 的 M(挤)、M(不挤) 各自只在对应状态之后更新；η = 0 时 W 与 B 完全相同）
  模型：HLB；HW = H + W；HDW = H + D + W；HWL = H + W + L（合并之后 L 是否仍然需要）；HDWL = H + D + W + L
        对照取已有结果：HDLB（s25）、HDB（s23）、HB（s22）
  估计与比较：与 s22–s25 相同（层级 EM 最多 20 轮；iBIC）
  起点：HLB、HWL 从 s25 中 HDLB 的解出发（去掉 D）；HW、HDW 分别从 s22 的 HB、s23 的 HDB 出发；含 W 的模型 η 逐人取网格；
        HDWL 从 HDW 的解出发，L 的 (w_F, a_F) 取 HWL 的解
用法：python3 s26_两系统简化.py 拟合 <模型> | 汇总
输出：结果/s26_真实_<模型>.json、结果/s26_真实_汇总.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s25", str(pathlib.Path(__file__).with_name("s25_EWA检验.py")))
S25 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S25)
S23, S22, S21, S20 = S25.S23, S25.S22, S25.S21, S25.S20
n, T, G, S, OKP, CROWD_PREV, ONE_STEP = S25.n, S25.T, S25.G, S25.S, S25.OKP, S25.CROWD_PREV, S25.ONE_STEP
NEW = ["HLB", "HW", "HDW", "HWL", "HDWL"]
FREE = {"HLB": [0, 5, 6, 7, 1, 8, 2, 3, 4], "HW": [0, 5, 6, 7, 1, 8, 4], "HDW": [0, 5, 6, 7, 1, 8, 9, 10, 4], "HWL": [0, 5, 6, 7, 1, 8, 2, 3, 4],
        "HDWL": [0, 5, 6, 7, 1, 8, 9, 10, 2, 3, 4]}
FIXED_dE = {m: (50.0 if "L" in m else -50.0) for m in NEW}               # 含 L 的模型 δ_EWA = 1
WORLD = {"HLB": 0.0, "HW": 1.0, "HDW": 1.0, "HWL": 1.0, "HDWL": 1.0}                 # 第 13 列：0 = 状态预测 B，1 = 统一世界模型 W
ULO, UHI = S25.ULO, S25.UHI
ETA_GRID = [-7.0, -5.0, -3.5, -2.0, -1.0]


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """同 s25 的 run；第 13 列为 1 的行用统一世界模型 W（平均水平每轮更新），为 0 的行用 B。X：(行, 13)。"""
    rows = X.shape[0]
    G_ = G if G_ is None else G_; S_ = S if S_ is None else S_
    b, wI, wF, aF, wB = X[:, 0], X[:, 1], X[:, 2], expit(X[:, 3]), X[:, 4]
    eta, delta, sig, aH = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7]), expit(X[:, 8])
    wD, aD, dE, W = X[:, 9], aH * expit(X[:, 10]), expit(X[:, 11]), X[:, 12] > 0.5
    Mn, Mc, mu = 60 + delta / 2, 60 - delta / 2, np.full(rows, 60.0)
    QG, QS = np.full(rows, 0.5), np.full(rows, 0.35)
    H = np.zeros(rows); D = np.zeros(rows); nll = np.zeros(rows)
    if rng is not None:
        A_ = np.zeros((rows, T)); P_ = np.zeros((rows, T))
    for t in range(T):
        sp = CROWD_PREV[t]
        Mp = np.where(W, mu + delta / 2 * (1 - 2 * sp), Mc if sp else Mn)
        p = ndtr((60.5 - Mp) / sig)
        z = b + wI * H + wD * D + wF * (QG - QS) + wB * (1.7 * p - 0.7)
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
        mu = mu + eta * (S20.N[t] - mu)
        QG = QG + aF * (a + (1 - a) * dE) * (G_[:, t] - QG)
        QS = QS + aF * ((1 - a) + a * dE) * (0.7 * S_[:, t] - QS)
        H = H + aH * (2 * a - 1 - H)
        D = D + aD * (2 * a - 1 - D)
    return (A_, P_) if rng is not None else nll


def to_X(U, m):
    X = np.zeros((U.shape[0], 13)); X[:, 8] = ONE_STEP; X[:, 11] = FIXED_dE[m]; X[:, 12] = WORLD[m]
    for j, col in enumerate(FREE[m]):
        X[:, col] = np.exp(U[:, j]) if col in S22.LOGW else U[:, j]
    return X


def to_U(X, m):
    idx = FREE[m]
    U = np.column_stack([np.log(np.maximum(X[:, c], np.exp(-8))) if c in S22.LOGW else X[:, c] for c in idx])
    return np.clip(U, ULO[idx], UHI[idx])


S21.run, S21.FREE, S21.ULO, S21.UHI, S21.to_X, S21.to_U = run, FREE, ULO, UHI, to_X, to_U
S22.FREE, S22.ULO, S22.UHI, S22.to_X = FREE, ULO, UHI, to_X


def widen(X, m):
    """把较早步骤的解补成 13 列，并设好本模型的固定列。"""
    Y = np.zeros((X.shape[0], 13)); Y[:, :X.shape[1]] = X
    if X.shape[1] < 9:
        Y[:, 8] = ONE_STEP
    Y[:, 11], Y[:, 12] = FIXED_dE[m], WORLD[m]
    return Y


def eta_grid(X):
    f = []
    for e in ETA_GRID:
        Xa = X.copy(); Xa[:, 5] = e; f.append(run(Xa, S20.A, S20.P, OKP))
    X = X.copy(); X[:, 5] = np.array(ETA_GRID)[np.argmin(np.array(f), 0)]
    return X


def start(m):
    if m == "HDWL":
        X = widen(np.array(json.loads((PL.OUT / "s26_真实_HDW.json").read_text(encoding="utf-8"))["X"]), m)
        X[:, 2:4] = np.array(json.loads((PL.OUT / "s26_真实_HWL.json").read_text(encoding="utf-8"))["X"])[:, 2:4]
        U = to_U(X, m)
        return U, U.mean(0), np.maximum(U.var(0), 1e-2)
    src = {"HLB": "s25_真实_HDLB.json", "HWL": "s25_真实_HDLB.json", "HW": "s22_真实_HB.json", "HDW": "s23_真实_HDB.json"}[m]
    X = widen(np.array(json.loads((PL.OUT / src).read_text(encoding="utf-8"))["X"]), m)
    if "D" not in m:
        X[:, 9] = 0.0
    if WORLD[m]:
        X = eta_grid(X)
    U = to_U(X, m)
    return U, U.mean(0), np.maximum(U.var(0), 1e-2)


if __name__ == "__main__":
    mode = sys.argv[1]
    Ntot = float((T + OKP.sum(1)).sum())
    if mode == "拟合":
        m = sys.argv[2]
        logf = open(PL.W / "日志" / f"s26_真实_{m}.log", "a", encoding="utf-8")
        log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
        log(f"数据集 真实，模型 {m}")
        R = S22.em(m, S20.A, S20.P, *start(m), log)
        PL.save(dict(数据集="真实", 模型=m, iBIC=float(-2 * R["LE"].sum() + 2 * len(FREE[m]) * np.log(Ntot)), **R), f"s26_真实_{m}.json")
        log("完成")
    else:
        src = {"HB": "s22_真实_HB.json", "HDB": "s23_真实_HDB.json", "HDLB": "s25_真实_HDLB.json", **{m: f"s26_真实_{m}.json" for m in NEW}}
        D = {m: json.loads((PL.OUT / f).read_text(encoding="utf-8")) for m, f in src.items()}
        ib = {m: D[m]["iBIC"] for m in D}
        best = min(ib, key=ib.get)
        k = {"HB": 7, "HDB": 9, "HDLB": 11, **{m: len(FREE[m]) for m in NEW}}
        q = lambda v: [round(float(x), 4) for x in np.percentile(v, [25, 50, 75])]
        out = dict(iBIC={m: round(v, 1) for m, v in ib.items()}, 相对最优={m: round(v - ib[best], 1) for m, v in ib.items()}, 最优=best,
                   每人参数个数=k, EM轮数={m: D[m]["EM轮数"] for m in D},
                   关键比较={"D 是否需要（HLB − HDLB）": round(ib["HLB"] - ib["HDLB"], 1),
                         "统一世界模型 对 B + L（HW − HLB）": round(ib["HW"] - ib["HLB"], 1),
                         "统一世界模型 对 状态预测（HW − HB）": round(ib["HW"] - ib["HB"], 1),
                         "合并之后 L 是否仍需要（HWL − HW）": round(ib["HWL"] - ib["HW"], 1),
                         "含 D 时：HDW − HDB": round(ib["HDW"] - ib["HDB"], 1),
                         "有 D 与 W 时 L 是否仍需要（HDWL − HDW）": round(ib["HDWL"] - ib["HDW"], 1)},
                   W的学习率η=({m: q(expit(np.array(D[m]["X"])[:, 5])) for m in ("HW", "HDW", "HWL", "HDWL")}))
        PL.save(out, "s26_真实_汇总.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
