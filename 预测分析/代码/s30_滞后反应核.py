# -*- coding: utf-8 -*-
"""
第 30 步：选择对前几轮人数的直接反应（绕过信念的快速成分）
  背景：s29 的最优模型 HDWLG 在 ABM 中把 ACF1 从 −.16 改进到 −.28（真实 −.35），但 ACF2（模拟 −.11，真实 +.06）
        与 ACF4（模拟 ≈ 0，真实 −.15）仍不对；开环残差与 2 轮前人数偏离正相关、与 4 轮前负相关。
        第 4 层的分析发现：对上一轮偏离的分级反应 λ 不经由信念；4 轮记忆在选择里、不在信念里。
  在 HDWLG（世界模型含 κ·lag1）上，只在选择中加入 λ_k·lag_k（lag_k = (N_{t−k} − 60) / 10）：
    HDWLG1 = + λ1；HDWLG4 = + λ4；HDWLG24 = + λ2、λ4；HDWLG124 = + λ1、λ2、λ4
  估计与比较：与 s22–s29 相同（层级 EM 最多 20 轮；iBIC）；起点为 s29 中 HDWLG 的解，新增的 λ 从 0 出发
用法：python3 s30_滞后反应核.py 拟合 <模型> | 汇总
输出：结果/s30_真实_<模型>.json、结果/s30_真实_汇总.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s29", str(pathlib.Path(__file__).with_name("s29_分级反应.py")))
S29 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S29)
S26, S22, S21, S20 = S29.S26, S29.S22, S29.S21, S29.S20
n, T, G, S, OKP, CROWD_PREV, ONE_STEP = S29.n, S29.T, S29.G, S29.S, S29.OKP, S29.CROWD_PREV, S29.ONE_STEP
NFULL = S20.N_full[-T - 4:]                                              # 第 1 轮之前的 4 轮 + 400 轮
LAGS = {k: (NFULL[4 - k:4 - k + T] - 60) / 10 for k in (1, 2, 3, 4)}    # LAGS[k][t] = (N_{t−k} − 60) / 10
assert np.allclose(LAGS[1], S29.LAG)
BASE = S29.FREE["HDWLG"]
NEW = ["HDWLG1", "HDWLG4", "HDWLG24", "HDWLG124"]
COL = {1: 13, 2: 15, 4: 16}                                              # λ1 沿用 s29 的第 14 列；λ2、λ4 为第 16、17 列
FREE = {m: BASE + [COL[int(c)] for c in m[5:]] for m in NEW}
ULO = np.r_[S29.ULO, -10.0, -10.0]
UHI = np.r_[S29.UHI, 10.0, 10.0]


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """同 s29 的 run，另加 λ2·lag2（第 16 列）与 λ4·lag4（第 17 列）。X：(行, 17)。"""
    rows = X.shape[0]
    G_ = G if G_ is None else G_; S_ = S if S_ is None else S_
    b, wI, wF, aF, wB = X[:, 0], X[:, 1], X[:, 2], expit(X[:, 3]), X[:, 4]
    eta, delta, sig, aH = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7]), expit(X[:, 8])
    wD, aD, dE, lam1, kap, lam2, lam4 = X[:, 9], aH * expit(X[:, 10]), expit(X[:, 11]), X[:, 13], X[:, 14], X[:, 15], X[:, 16]
    mu = np.full(rows, 60.0)
    QG, QS = np.full(rows, 0.5), np.full(rows, 0.35)
    H = np.zeros(rows); D = np.zeros(rows); nll = np.zeros(rows)
    if rng is not None:
        A_ = np.zeros((rows, T)); P_ = np.zeros((rows, T))
    for t in range(T):
        sp = CROWD_PREV[t]
        Mp = mu + delta / 2 * (1 - 2 * sp) + kap * LAGS[1][t]
        p = ndtr((60.5 - Mp) / sig)
        z = (b + wI * H + wD * D + wF * (QG - QS) + wB * (1.7 * p - 0.7)
             + lam1 * LAGS[1][t] + lam2 * LAGS[2][t] + lam4 * LAGS[4][t])
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
    X = np.zeros((U.shape[0], 17)); X[:, 8] = ONE_STEP; X[:, 11] = 50.0; X[:, 12] = 1.0
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
    r = json.loads((PL.OUT / "s29_真实_HDWLG.json").read_text(encoding="utf-8"))
    X = np.zeros((n, 17)); X[:, :15] = np.array(r["X"])
    U = to_U(X, m)
    s2 = np.r_[np.array(r["s2"]), np.full(len(FREE[m]) - len(r["s2"]), 0.25)]
    return U, U.mean(0), s2


if __name__ == "__main__":
    mode = sys.argv[1]
    Ntot = float((T + OKP.sum(1)).sum())
    if mode == "拟合":
        m = sys.argv[2]
        logf = open(PL.W / "日志" / f"s30_真实_{m}.log", "a", encoding="utf-8")
        log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
        log(f"数据集 真实，模型 {m}")
        R = S22.em(m, S20.A, S20.P, *start(m), log)
        PL.save(dict(数据集="真实", 模型=m, iBIC=float(-2 * R["LE"].sum() + 2 * len(FREE[m]) * np.log(Ntot)), **R), f"s30_真实_{m}.json")
        log("完成")
    else:
        src = {"HDWLG": "s29_真实_HDWLG.json", **{m: f"s30_真实_{m}.json" for m in NEW}}
        D = {m: json.loads((PL.OUT / f).read_text(encoding="utf-8")) for m, f in src.items()}
        ib = {m: D[m]["iBIC"] for m in D}; best = min(ib, key=ib.get)
        q = lambda v: [round(float(x), 3) for x in np.percentile(v, [25, 50, 75])]
        lam = {}
        for m in NEW:
            X = np.array(D[m]["X"])
            lam[m] = {f"λ{k}": dict(四分位数=q(X[:, COL[k]]), 群体均值=round(float(X[:, COL[k]].mean()), 3)) for k in (1, 2, 4) if COL[k] in FREE[m]}
        out = dict(iBIC={m: round(v, 1) for m, v in ib.items()}, 相对最优={m: round(v - ib[best], 1) for m, v in ib.items()}, 最优=best,
                   EM轮数={m: D[m]["EM轮数"] for m in D}, λ=lam)
        PL.save(out, "s30_真实_汇总.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
