# -*- coding: utf-8 -*-
"""
第 25 步：剩余的弱成分是 MF（只为所选的行动、按自己得到的收益记账），还是从全部反馈中学习（信念学习）？
  背景：s23 中加入慢漂移后，MF 的边际贡献仍在（HDFB − HDB = −32 iBIC），且 a_F 变快（中位数约 .12）；
        后验预测中它在 t−1 的状态效应与"自己 × 状态"交互上更接近真实数据。
  EWA 的 δ（Camerer & Ho, 1999）：未选的行动也用它本该得到的收益更新，权重为 δ
    Q_去 ← Q_去 + α_F·[a + (1 − a)·δ]·(G − Q_去)            G = 若去会不挤（反事实，任何选择下都有定义）
    Q_留 ← Q_留 + α_F·[(1 − a) + a·δ]·(0.7·S − Q_留)        S = 若不去酒吧会挤
    δ = 0：纯 MF（s20–s23 的 F）；δ = 1：两个行动每轮都按公共结果更新，Q_去 − Q_留 是近期"去是否划算"的加权平均，
    不依赖自己的选择，即一个不分状态、偏重近期的信念学习者（属于世界层）
  模型（都含习惯痕迹 H、慢漂移 D、世界模型 B）：
    HDFB  δ = 0（取 s23 的结果）
    HDLB  δ = 1（参数个数与 HDFB 相同：两者的 iBIC 之差就是"MF 对信念学习"的直接比较）
    HDEB  δ 自由（logit 尺度）
  估计与比较：与 s22 / s23 相同（层级 EM 最多 20 轮；iBIC）
  起点：从 s23 中 HDFB 的解出发；HDLB 的 a_F、HDEB 的 (a_F, δ) 逐人在网格上取似然最大者
  后验预测检查：HDLB、HDEB 各模拟 20 套，回归与 s22 相同（其余模型取 s23 的结果）
用法：python3 s25_EWA检验.py 拟合 <模型> | 汇总 | 预测检查
输出：结果/s25_真实_<模型>.json、结果/s25_真实_汇总.json、结果/s25_预测检查.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s23", str(pathlib.Path(__file__).with_name("s23_慢漂移检验.py")))
S23 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S23)
S22, S21, S20 = S23.S22, S23.S21, S23.S20
n, T, G, S, OKP, CROWD_PREV, ONE_STEP = S23.n, S23.T, S23.G, S23.S, S23.OKP, S23.CROWD_PREV, S23.ONE_STEP
NAMES = S23.NAMES + ["dE"]
NEW = ["HDLB", "HDEB"]
FIXED_D = {"HDFB": -50.0, "HDLB": 50.0, "HDEB": -50.0}                 # logit δ：−50 → δ = 0，+50 → δ = 1
BASE = [0, 5, 6, 7, 1, 8, 9, 10, 2, 3, 4]
FREE = {"HDFB": BASE, "HDLB": BASE, "HDEB": BASE + [11]}
ULO = np.r_[S23.ULO, -7.0]
UHI = np.r_[S23.UHI, 7.0]
AF_GRID = [-3.0, -1.5, 0.0, 1.5]
DE_GRID = [-3.0, 0.0, 3.0]


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """同 s23 的 run，只把 MF 的更新换成 EWA 式（未选的行动以权重 δ 用反事实收益更新）。X：(行, 12)，第 12 列为 logit δ。"""
    rows = X.shape[0]
    G_ = G if G_ is None else G_; S_ = S if S_ is None else S_
    b, wI, wF, aF, wB = X[:, 0], X[:, 1], X[:, 2], expit(X[:, 3]), X[:, 4]
    eta, delta, sig, aH = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7]), expit(X[:, 8])
    wD, aD, dE = X[:, 9], aH * expit(X[:, 10]), expit(X[:, 11])
    Mn, Mc = 60 + delta / 2, 60 - delta / 2
    QG, QS = np.full(rows, 0.5), np.full(rows, 0.35)
    H = np.zeros(rows); D = np.zeros(rows); nll = np.zeros(rows)
    if rng is not None:
        A_ = np.zeros((rows, T)); P_ = np.zeros((rows, T))
    for t in range(T):
        sp = CROWD_PREV[t]
        Mp = Mc if sp else Mn
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
        QG = QG + aF * (a + (1 - a) * dE) * (G_[:, t] - QG)
        QS = QS + aF * ((1 - a) + a * dE) * (0.7 * S_[:, t] - QS)
        H = H + aH * (2 * a - 1 - H)
        D = D + aD * (2 * a - 1 - D)
    return (A_, P_) if rng is not None else nll


def to_X(U, m):
    X = np.zeros((U.shape[0], 12)); X[:, 8] = ONE_STEP; X[:, 11] = FIXED_D[m]
    for j, col in enumerate(FREE[m]):
        X[:, col] = np.exp(U[:, j]) if col in S22.LOGW else U[:, j]
    return X


def to_U(X, m):
    idx = FREE[m]
    U = np.column_stack([np.log(np.maximum(X[:, c], np.exp(-8))) if c in S22.LOGW else X[:, c] for c in idx])
    return np.clip(U, ULO[idx], UHI[idx])


S21.run, S21.FREE, S21.ULO, S21.UHI, S21.to_X, S21.to_U = run, FREE, ULO, UHI, to_X, to_U
S22.FREE, S22.ULO, S22.UHI, S22.to_X = FREE, ULO, UHI, to_X


def grid_start(X, m):
    """HDLB：a_F 取网格；HDEB：(a_F, δ) 取网格。其余参数固定，逐人取似然最大者。"""
    cand = [(a, FIXED_D[m]) for a in AF_GRID] if m == "HDLB" else [(a, d) for a in AF_GRID for d in DE_GRID]
    f = []
    for a, d in cand:
        Xa = X.copy(); Xa[:, 3], Xa[:, 11] = a, d; f.append(run(Xa, S20.A, S20.P, OKP))
    best = np.argmin(np.array(f), 0)
    X = X.copy(); X[:, 3] = np.array([c[0] for c in cand])[best]; X[:, 11] = np.array([c[1] for c in cand])[best]
    return X


def load(m):
    src = f"s23_真实_{m}.json" if m in S23.NEW else f"s25_真实_{m}.json"
    X = np.array(json.loads((PL.OUT / src).read_text(encoding="utf-8"))["X"])
    return X if X.shape[1] == 12 else np.column_stack([X, np.full(n, FIXED_D[m])]), json.loads((PL.OUT / src).read_text(encoding="utf-8"))


if __name__ == "__main__":
    mode = sys.argv[1]
    Ntot = float((T + OKP.sum(1)).sum())
    if mode == "拟合":
        m = sys.argv[2]
        logf = open(PL.W / "日志" / f"s25_真实_{m}.log", "a", encoding="utf-8")
        log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
        X0, _ = load("HDFB"); X0[:, 11] = FIXED_D[m]
        U = to_U(grid_start(X0, m), m)
        log(f"数据集 真实，模型 {m}")
        R = S22.em(m, S20.A, S20.P, U, U.mean(0), np.maximum(U.var(0), 1e-2), log)
        PL.save(dict(数据集="真实", 模型=m, iBIC=float(-2 * R["LE"].sum() + 2 * len(FREE[m]) * np.log(Ntot)), **R), f"s25_真实_{m}.json")
        log("完成")
    elif mode == "汇总":
        ms = ["HDB", "HDFB", "HDLB", "HDEB"]
        ib = {m: json.loads((PL.OUT / (f"s23_真实_{m}.json" if m in S23.NEW else f"s25_真实_{m}.json")).read_text(encoding="utf-8"))["iBIC"] for m in ms}
        best = min(ib, key=ib.get)
        XE, rE = load("HDEB")
        dE = expit(XE[:, 11])
        out = dict(iBIC={m: round(v, 1) for m, v in ib.items()}, 相对最优={m: round(v - ib[best], 1) for m, v in ib.items()}, 最优=best,
                   关键比较={"MF 对 信念学习（HDFB − HDLB，参数个数相同）": round(ib["HDFB"] - ib["HDLB"], 1),
                         "信念学习的边际贡献（HDLB − HDB）": round(ib["HDLB"] - ib["HDB"], 1),
                         "δ 自由对 δ = 1（HDEB − HDLB）": round(ib["HDEB"] - ib["HDLB"], 1)},
                   δ的分布=dict(四分位数=[round(float(x), 3) for x in np.percentile(dE, [25, 50, 75])],
                             群体均值_概率尺度=round(float(expit(rE["mu"][-1])), 3), 群体方差_logit尺度=round(float(rE["s2"][-1]), 3),
                             大于0点5的人数=int((dE > 0.5).sum())))
        par = {}
        for m in ["HDFB", "HDLB", "HDEB"]:
            X, _ = load(m)
            q = lambda v: [round(float(x), 3) for x in np.percentile(v, [25, 50, 75])]
            par[m] = {"wF": q(X[:, 2]), "aF": q(expit(X[:, 3])), "wB": q(X[:, 4]), "wI": q(X[:, 1]), "wD": q(X[:, 9])}
        out["参数（四分位数）"] = par
        PL.save(out, "s25_真实_汇总.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        old = json.loads((PL.OUT / "s23_预测检查.json").read_text(encoding="utf-8"))
        res = dict(old)
        for m in NEW:
            X, _ = load(m)
            sims = [S22.lag_coefs(run(X, None, None, None, rng=np.random.default_rng(5000 + r))[0]) for r in range(20)]
            res[m] = dict(均值=np.mean(sims, 0).round(3).tolist(), 模拟间标准差=np.std(sims, 0).round(3).tolist())
            print(m, flush=True)
        PL.save(res, "s25_预测检查.json")
        print(json.dumps(res, ensure_ascii=False, indent=1))
