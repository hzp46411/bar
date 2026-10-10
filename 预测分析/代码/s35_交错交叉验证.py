# -*- coding: utf-8 -*-
"""
第 35 步：分级反应 κ 是过拟合还是随时间变化？——交错区组的交叉验证，以及去掉 κ 之后的模型族
  背景：s28 的前后半程交叉验证中，去掉 κ 反而预测得更好（检验负对数似然合计好 336）；
        但 κ 的群体中位数从前半程的 +0.40 降到后半程的 +0.04，而 δ、σ 稳定。前后半程切分把"过拟合"与"不稳定"混在一起。
  交错区组：每 40 轮一组，奇数组训练、偶数组检验，再反过来（检验与训练在同一时期）
  新模型（字母同 s28；另加 T：κ 随时间线性变化 κ_t = κ + κ1·(t/T − 0.5)）：
    HDW1、HDWL、HDWF1（去掉 κ 之后的敲除与 MF 替换）；HDWLT1（κ 随时间变化）
  比较的模型：HDWLG1、HDWL1、HDWG1、HDWLG、HDWFG1、HDW1、HDWL、HDWF1、HDWLT1
    交错区组：全部；前后半程：新模型（其余取 s28）
  真实拟合：新模型从 s30 中 HDWLG124 的解出发（去掉不含的成分；F 的 α_F 逐人取网格；κ1 从 0 出发）
  交叉验证的起点：该模型真实拟合的群体均值（不用个人信息）
用法：python3 s35_交错交叉验证.py 真实 <模型> | 交错 <模型> <奇|偶> | 半程 <模型> <前半|后半> | 汇总
输出：结果/s35_*.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s28", str(pathlib.Path(__file__).with_name("s28_最终模型验证.py")))
S28 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S28)
S30, S22, S21, S20 = S28.S30, S28.S22, S28.S21, S28.S20
n, T, G, S, OKP, CROWD_PREV, ONE_STEP, LAGS = S28.n, S28.T, S28.G, S28.S, S28.OKP, S28.CROWD_PREV, S28.ONE_STEP, S28.LAGS
NEW = ["HDW1", "HDWL", "HDWF1", "HDWLT1"]
ALL = ["HDWLG1", "HDWL1", "HDWG1", "HDWLG", "HDWFG1"] + NEW
FREE = S28.FREE
for m in NEW:
    FREE[m] = S28.free(m.replace("T", "G")) + ([17] if "T" in m else [])
ULO = np.r_[S28.ULO, -40.0]
UHI = np.r_[S28.UHI, 40.0]
CW = np.ones(T)
TT = np.arange(T) / T - 0.5


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """同 s28 的 run；κ_t = κ + κ1·(t/T − 0.5)（第 18 列为 κ1）。X：(行, 18)。"""
    rows = X.shape[0]
    G_ = G if G_ is None else G_; S_ = S if S_ is None else S_
    b, wI, wF, aF, wB = X[:, 0], X[:, 1], X[:, 2], expit(X[:, 3]), X[:, 4]
    eta, delta, sig, aH = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7]), expit(X[:, 8])
    wD, aD, dE, lam1, kap, lam2, lam4, kap1 = (X[:, 9], aH * expit(X[:, 10]), expit(X[:, 11]), X[:, 13], X[:, 14],
                                               X[:, 15], X[:, 16], X[:, 17])
    mu = np.full(rows, 60.0)
    QG, QS = np.full(rows, 0.5), np.full(rows, 0.35)
    H = np.zeros(rows); D = np.zeros(rows); nll = np.zeros(rows)
    if rng is not None:
        A_ = np.zeros((rows, T)); P_ = np.zeros((rows, T))
    for t in range(T):
        sp = CROWD_PREV[t]
        Mp = mu + delta / 2 * (1 - 2 * sp) + (kap + kap1 * TT[t]) * LAGS[1][t]
        p = ndtr((60.5 - Mp) / sig)
        z = (b + wI * H + wD * D + wF * (QG - QS) + wB * (1.7 * p - 0.7)
             + lam1 * LAGS[1][t] + lam2 * LAGS[2][t] + lam4 * LAGS[4][t])
        if rng is not None:
            A_[:, t] = (rng.random(rows) < expit(z)).astype(float)
            P_[:, t] = np.round(Mp + sig * rng.standard_normal(rows))
        a = A_[:, t]
        if rng is None:
            nll += CW[t] * (np.logaddexp(0, z) - a * z)
            nll += OKP_[:, t] * (0.5 * np.log(2 * np.pi) + np.log(sig) + 0.5 * ((P_[:, t] - Mp) / sig) ** 2)
        mu = mu + eta * (S20.N[t] - mu)
        QG = QG + aF * (a + (1 - a) * dE) * (G_[:, t] - QG)
        QS = QS + aF * ((1 - a) + a * dE) * (0.7 * S_[:, t] - QS)
        H = H + aH * (2 * a - 1 - H)
        D = D + aD * (2 * a - 1 - D)
    return (A_, P_) if rng is not None else nll


def to_X(U, m):
    X = np.zeros((U.shape[0], 18)); X[:, 8] = ONE_STEP; X[:, 11] = -50.0 if "F" in m else 50.0; X[:, 12] = 1.0
    for j, col in enumerate(FREE[m]):
        X[:, col] = np.exp(U[:, j]) if col in S22.LOGW else U[:, j]
    return X


def to_U(X, m):
    idx = FREE[m]
    U = np.column_stack([np.log(np.maximum(X[:, c], np.exp(-8))) if c in S22.LOGW else X[:, c] for c in idx])
    return np.clip(U, ULO[idx], UHI[idx])


S21.run, S21.ULO, S21.UHI, S21.to_X, S21.to_U = run, ULO, UHI, to_X, to_U
S22.ULO, S22.UHI, S22.to_X = ULO, UHI, to_X
Ntot = S28.Ntot
load = S28.load


def pad(X):
    Y = np.zeros((X.shape[0], 18)); Y[:, :X.shape[1]] = X
    return Y


def real_start(m):
    X = pad(np.array(load("s30_真实_HDWLG124.json")["X"]))
    keep = set(FREE[m]) | {11, 12}
    X[:, [c for c in range(18) if c not in keep]] = 0.0
    X[:, 8] = X[:, 8] if 8 in keep else ONE_STEP
    X[:, 11] = -50.0 if "F" in m else 50.0
    if "F" in m:
        f = []
        for a in [-3.0, -1.5, 0.0, 1.5]:
            Xa = X.copy(); Xa[:, 3] = a; f.append(run(Xa, S20.A, S20.P, OKP))
        X[:, 3] = np.array([-3.0, -1.5, 0.0, 1.5])[np.argmin(np.array(f), 0)]
    U = to_U(X, m)
    return U, U.mean(0), np.maximum(U.var(0), 1e-2)


def real_file(m):
    return f"s35_真实_{m}.json" if m in NEW else f"s28_真实_{m}.json"


def mean_start(m):
    r = load(real_file(m))
    mu, s2 = np.array(r["mu"]), np.array(r["s2"])
    return np.tile(mu, (n, 1)), mu, s2


def cv(m, train, tag):
    CW[:] = train; S21.OKP = OKP * train
    R = S28.fit(m, S20.A, S20.P, *mean_start(m), tag)
    X = R["X"]
    CW[:] = ~train; choice = run(X, S20.A, S20.P, np.zeros_like(OKP))
    CW[:] = 0.0; fc = run(X, S20.A, S20.P, OKP * ~train)
    return dict(检验选择NLL=choice, 检验预测NLL=fc, X=X)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "真实":
        m = sys.argv[2]
        R = S28.fit(m, S20.A, S20.P, *real_start(m), f"s35真实_{m}")
        PL.save(dict(模型=m, iBIC=float(-2 * R["LE"].sum() + 2 * len(FREE[m]) * np.log(Ntot)), **R), f"s35_真实_{m}.json")
    elif mode == "交错":
        m, fold = sys.argv[2], sys.argv[3]
        blk = (np.arange(T) // 40) % 2 == (1 if fold == "奇" else 0)
        PL.save(dict(模型=m, 训练=fold, **cv(m, blk, f"s35交错_{m}_{fold}")), f"s35_交错_{m}_{fold}.json")
    elif mode == "半程":
        m, fold = sys.argv[2], sys.argv[3]
        train = (np.arange(T) < T // 2) if fold == "前半" else (np.arange(T) >= T // 2)
        PL.save(dict(模型=m, 训练=fold, **cv(m, train, f"s35半程_{m}_{fold}")), f"s35_交叉验证_{m}_{fold}.json")
    else:
        out = {}
        ib = {m: load(real_file(m))["iBIC"] for m in ALL}
        out["真实_iBIC_相对HDWLG1"] = {m: round(v - ib["HDWLG1"], 1) for m, v in ib.items()}
        for scheme, pre, folds in (("交错区组", "s35_交错", ("奇", "偶")), ("前后半程", None, ("前半", "后半"))):
            res = {}
            for m in ALL:
                p = pre or ("s35_交叉验证" if m in NEW else "s28_交叉验证")
                try:
                    d = [load(f"{p}_{m}_{f}.json") for f in folds]
                except FileNotFoundError:
                    continue
                res[m] = dict(选择=float(sum(np.sum(x["检验选择NLL"]) for x in d)), 预测=float(sum(np.sum(x["检验预测NLL"]) for x in d)),
                              逐折=[[round(float(np.sum(x["检验选择NLL"])), 1), round(float(np.sum(x["检验预测NLL"])), 1)] for x in d])
            if "HDWLG1" in res:
                b = res["HDWLG1"]
                out[scheme] = {m: dict(选择=round(v["选择"] - b["选择"], 1), 预测=round(v["预测"] - b["预测"], 1),
                                       合计=round(v["选择"] + v["预测"] - b["选择"] - b["预测"], 1), 逐折=v["逐折"]) for m, v in res.items()}
        if (PL.OUT / "s35_真实_HDWLT1.json").exists():
            X = np.array(load("s35_真实_HDWLT1.json")["X"])
            q = lambda v: [round(float(x), 3) for x in np.percentile(v, [25, 50, 75])]
            out["κ 随时间变化（HDWLT1）"] = dict(κ均值=q(X[:, 14]), κ1斜率=q(X[:, 17]), 前段κ=q(X[:, 14] - 0.5 * X[:, 17]), 后段κ=q(X[:, 14] + 0.5 * X[:, 17]))
        PL.save(out, "s35_汇总.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
