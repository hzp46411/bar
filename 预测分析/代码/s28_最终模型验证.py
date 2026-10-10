# -*- coding: utf-8 -*-
"""
第 28 步：最终模型空间的验证——同一设置下的真实拟合、样本外交叉验证、群体层面的模型恢复与参数恢复
  模型用字母拼写（前向模型同 s30）：
    H 习惯痕迹（w_I、α_H）  D 慢漂移（w_D、α_D）
    W 世界模型进入选择（w_B；世界模型 μ、δ、σ、η 总在，由预测约束）  G 世界模型中的分级反应 κ·lag1
    L 近期信念（EWA δ = 1）  F 把 L 换成 MF（δ = 0，参数个数相同）
    1 / 2 / 4 选择端对 1 / 2 / 4 轮前人数的反应 λ1、λ2、λ4
  候选（主模型 HDWLG1 及其敲除）：HDWLG1；DWLG1（去 H）、HWLG1（去 D）、HDWG1（去 L）、HDLG1（去 W）、HDWL1（去 κ）、
        HDWLG（去 λ1）；HDWFG1（L 换成 MF）；HW（最简的两系统）；HDWLG124（再加 λ2、λ4）
  真实：每个候选从 s30 中 HDWLG124 的解出发（去掉不含的成分；F 的 α_F 逐人取网格），层级 EM 最多 20 轮
  交叉验证：只用前半程（或后半程）的选择与预测做层级估计，起点为该模型真实拟合的群体均值（不用个人信息）；
        在另一半程上计算选择与预测的负对数似然（状态用真实历史从第 1 轮跑起）
  恢复：用生成模型的真实拟合值生成假数据（开环：他人的选择与公布人数取真实值），每个候选都拟合；
        起点为该候选真实拟合的群体均值；群体层面按 iBIC 选模型；生成模型拟合自己的数据时报告参数恢复
用法：python3 s28_最终模型验证.py 真实 <模型>
      python3 s28_最终模型验证.py 交叉验证 <模型> <前半|后半>
      python3 s28_最终模型验证.py 恢复 <生成模型>_<重复号> <模型>
      python3 s28_最终模型验证.py 汇总
输出：结果/s28_*.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr
from scipy.stats import spearmanr
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s30", str(pathlib.Path(__file__).with_name("s30_滞后反应核.py")))
S30 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S30)
S22, S21, S20 = S30.S22, S30.S21, S30.S20
n, T, G, S, OKP, CROWD_PREV, ONE_STEP, LAGS = S30.n, S30.T, S30.G, S30.S, S30.OKP, S30.CROWD_PREV, S30.ONE_STEP, S30.LAGS
NAMES = ["b", "wI", "wF", "aF", "wB", "eta", "delta", "lsig", "aH", "wD", "aD", "dE", "W", "lam1", "kap", "lam2", "lam4"]
CAND = ["HDWLG1", "DWLG1", "HWLG1", "HDWG1", "HDLG1", "HDWL1", "HDWLG", "HDWFG1", "HW", "HDWLG124"]
GEN = ["HDWLG1", "HDWFG1", "HDWG1", "HDLG1", "HWLG1", "HDWLG124"]
ULO, UHI = S30.ULO, S30.UHI
CW = np.ones(T)                                                         # 选择似然的轮次权重（交叉验证时只用训练半程）


def free(m):
    f = [0, 5, 6, 7]
    if "H" in m: f += [1, 8]
    if "D" in m: f += ([8] if "H" not in m else []) + [9, 10]
    if "L" in m or "F" in m: f += [2, 3]
    if "W" in m: f += [4]
    if "G" in m: f += [14]
    for c, col in (("1", 13), ("2", 15), ("4", 16)):
        if c in m: f += [col]
    return f


FREE = {m: free(m) for m in CAND}


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """同 s30 的 run；选择似然乘以轮次权重 CW。"""
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
            nll += CW[t] * (np.logaddexp(0, z) - a * z)
            nll += OKP_[:, t] * (0.5 * np.log(2 * np.pi) + np.log(sig) + 0.5 * ((P_[:, t] - Mp) / sig) ** 2)
        mu = mu + eta * (S20.N[t] - mu)
        QG = QG + aF * (a + (1 - a) * dE) * (G_[:, t] - QG)
        QS = QS + aF * ((1 - a) + a * dE) * (0.7 * S_[:, t] - QS)
        H = H + aH * (2 * a - 1 - H)
        D = D + aD * (2 * a - 1 - D)
    return (A_, P_) if rng is not None else nll


def to_X(U, m):
    X = np.zeros((U.shape[0], 17)); X[:, 8] = ONE_STEP; X[:, 11] = -50.0 if "F" in m else 50.0; X[:, 12] = 1.0
    for j, col in enumerate(FREE[m]):
        X[:, col] = np.exp(U[:, j]) if col in S22.LOGW else U[:, j]
    return X


def to_U(X, m):
    idx = FREE[m]
    U = np.column_stack([np.log(np.maximum(X[:, c], np.exp(-8))) if c in S22.LOGW else X[:, c] for c in idx])
    return np.clip(U, ULO[idx], UHI[idx])


S21.run, S21.FREE, S21.ULO, S21.UHI, S21.to_X, S21.to_U = run, FREE, ULO, UHI, to_X, to_U
S22.FREE, S22.ULO, S22.UHI, S22.to_X = FREE, ULO, UHI, to_X
Ntot = float((T + OKP.sum(1)).sum())


def load(f):
    return json.loads((PL.OUT / f).read_text(encoding="utf-8"))


def real_start(m):
    X = np.array(load("s30_真实_HDWLG124.json")["X"])
    keep = set(FREE[m]) | {11, 12}
    X[:, [c for c in range(17) if c not in keep]] = 0.0
    X[:, 8] = X[:, 8] if 8 in keep else ONE_STEP
    X[:, 11] = -50.0 if "F" in m else 50.0
    if "F" in m:
        f = []
        for a in [-3.0, -1.5, 0.0, 1.5]:
            Xa = X.copy(); Xa[:, 3] = a; f.append(run(Xa, S20.A, S20.P, OKP))
        X[:, 3] = np.array([-3.0, -1.5, 0.0, 1.5])[np.argmin(np.array(f), 0)]
    U = to_U(X, m)
    return U, U.mean(0), np.maximum(U.var(0), 1e-2)


def mean_start(m):
    r = load(f"s28_真实_{m}.json")
    mu, s2 = np.array(r["mu"]), np.array(r["s2"])
    return np.tile(mu, (n, 1)), mu, s2


def fit(m, A_, P_, U, mu, s2, logname):
    logf = open(PL.W / "日志" / f"s28_{logname}.log", "a", encoding="utf-8")
    log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
    log(f"{logname}")
    return S22.em(m, A_, P_, U, mu, s2, log)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "真实":
        m = sys.argv[2]
        R = fit(m, S20.A, S20.P, *real_start(m), f"真实_{m}")
        PL.save(dict(模型=m, iBIC=float(-2 * R["LE"].sum() + 2 * len(FREE[m]) * np.log(Ntot)), **R), f"s28_真实_{m}.json")
    elif mode == "交叉验证":
        m, fold = sys.argv[2], sys.argv[3]
        train = (np.arange(T) < T // 2) if fold == "前半" else (np.arange(T) >= T // 2)
        CW[:] = train; S21.OKP = OKP * train
        R = fit(m, S20.A, S20.P, *mean_start(m), f"交叉验证_{m}_{fold}")
        X = R["X"]
        CW[:] = ~train; choice = run(X, S20.A, S20.P, np.zeros_like(OKP))
        CW[:] = 0.0; fc = run(X, S20.A, S20.P, OKP * ~train)
        PL.save(dict(模型=m, 训练=fold, 检验选择NLL=choice, 检验预测NLL=fc, X=X), f"s28_交叉验证_{m}_{fold}.json")
    elif mode == "恢复":
        name, m = sys.argv[2], sys.argv[3]
        g, rep = name.rsplit("_", 1)
        Xg = np.array(load(f"s28_真实_{g}.json")["X"])
        As, Ps = run(Xg, None, None, None, rng=np.random.default_rng(9000 + 100 * GEN.index(g) + int(rep)))
        R = fit(m, As, Ps, *mean_start(m), f"恢复_{name}_{m}")
        PL.save(dict(生成模型=g, 模型=m, X真值=Xg, iBIC=float(-2 * R["LE"].sum() + 2 * len(FREE[m]) * np.log(Ntot)), **R),
                f"s28_恢复_{name}_{m}.json")
    else:
        out = {}
        ib = {m: load(f"s28_真实_{m}.json")["iBIC"] for m in CAND}; best = min(ib, key=ib.get)
        out["真实_iBIC_相对最优"] = {m: round(v - ib[best], 1) for m, v in ib.items()}
        cv = {}
        for m in CAND:
            try:
                d = [load(f"s28_交叉验证_{m}_{f}.json") for f in ("前半", "后半")]
                cv[m] = dict(选择=float(sum(np.sum(x["检验选择NLL"]) for x in d)), 预测=float(sum(np.sum(x["检验预测NLL"]) for x in d)))
            except FileNotFoundError:
                pass
        if cv:
            bc = min(v["选择"] for v in cv.values()); bf = min(v["预测"] for v in cv.values())
            out["交叉验证（检验半程的负对数似然，相对最优）"] = {m: dict(选择=round(v["选择"] - bc, 1), 预测=round(v["预测"] - bf, 1),
                                                         合计=round(v["选择"] + v["预测"], 1)) for m, v in cv.items()}
        rec = {}
        for g in GEN:
            for rep in (1, 2, 3):
                fs = {m: PL.OUT / f"s28_恢复_{g}_{rep}_{m}.json" for m in CAND}
                if not all(f.exists() for f in fs.values()):
                    continue
                D = {m: json.loads(f.read_text(encoding="utf-8")) for m, f in fs.items()}
                ibr = {m: D[m]["iBIC"] for m in CAND}; br = min(ibr, key=ibr.get)
                key = f"{g}_{rep}"
                rec[key] = dict(选中=br, 生成模型相对最优=round(ibr[g] - ibr[br], 1) if g in ibr else None,
                                相对最优={m: round(v - ibr[br], 1) for m, v in ibr.items()})
                if g in D:
                    Xt, Xe = np.array(D[g]["X真值"]), np.array(D[g]["X"])
                    pr = {}
                    for c in FREE[g]:
                        tv, ev = Xt[:, c], Xe[:, c]
                        if c in (3, 5, 8, 10):
                            tv, ev = expit(tv), expit(ev)
                        pr[NAMES[c]] = round(float(spearmanr(tv, ev)[0]), 3)
                    rec[key]["参数恢复_Spearman"] = pr
        out["模型恢复"] = rec
        PL.save(out, "s28_汇总.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
