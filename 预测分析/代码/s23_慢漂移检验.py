# -*- coding: utf-8 -*-
"""
第 23 步：MF 的剩余信号是不是"很长的记忆 / 缓慢漂移"
  背景：s22 中把一步惯性换成习惯痕迹后，HFB 只比 HB 好 1.7 iBIC，但"真值无 MF"的模拟中 HFB 平均输 65，
        真实数据仍有约 67 的剩余；后验预测检查中所有模型都低估了自己 t−5 的效应，HFB 在任何行为标志上都不比 HB 好。
        猜测：F 的慢痕迹（α_F 约 .07）在吸收一个没有建模的、比习惯更慢的选择倾向漂移。
  慢漂移痕迹 D：D ← D + α_D (c − D)，c = 本轮选择（±1），D 初值 0；选择中加 w_D·D
    α_D = α_H · expit(a_D)，保证 D 比习惯痕迹慢（两条痕迹不会互换标签）；w_D 可正可负
  模型：HB、HFB（与 s22 相同，直接取 s22 的结果）；HDB、HDFB（新拟合）
  估计：与 s22 相同（s21 的层级 EM，最多 20 轮；iBIC 比较）
  起点：真实数据——从 s22 中 HB / HFB 的解出发，(w_D, a_D) 逐人在网格上取似然最大者
        假数据——每人从该模型真实拟合的群体均值出发，(w_D, a_D) 同样取网格
  后验预测检查：HDB、HDFB 各模拟 20 套，回归与 s22 相同（HB、HFB 取 s22 的结果）
用法：python3 s23_慢漂移检验.py 拟合 <数据集> <模型>    数据集 = 真实，或 <生成模型>_<重复号>
      python3 s23_慢漂移检验.py 汇总
      python3 s23_慢漂移检验.py 预测检查
输出：结果/s23_<数据集>_<模型>.json、结果/s23_真实_汇总.json、结果/s23_预测检查.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr
import pred_lib as PL


def _load(fname, mod):
    spec = importlib.util.spec_from_file_location(mod, str(pathlib.Path(__file__).with_name(fname)))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    return M


S22 = _load("s22_习惯痕迹审计.py", "s22")
S20, S21 = S22.S20, S22.S21
n, T, G, S, OKP, CROWD_PREV, ONE_STEP = S22.n, S22.T, S22.G, S22.S, S22.OKP, S22.CROWD_PREV, S22.ONE_STEP
NAMES = S22.NAMES + ["wD", "aD"]
OLD = ["HB", "HFB"]                                                     # 取 s22 的结果
NEW = ["HDB", "HDFB"]
FREE = {m: [0, 5, 6, 7, 1, 8] + ([9, 10] if "D" in m else []) + ([2, 3] if "F" in m else []) + [4] for m in OLD + NEW}
ULO = np.r_[S22.ULO, -10.0, -7.0]
UHI = np.r_[S22.UHI, 10.0, 7.0]
D_GRID = [(w, a) for w in (-1.0, -0.5, 0.5, 1.0, 2.0) for a in (-3.0, -1.5, 0.0)]   # a_D：α_D / α_H ≈ .05、.18、.5


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """同 s22 的 run，另加慢漂移痕迹 D。X：(行, 11)，第 10、11 列为 w_D 与 a_D。"""
    rows = X.shape[0]
    G_ = G if G_ is None else G_; S_ = S if S_ is None else S_
    b, wI, wF, aF, wB = X[:, 0], X[:, 1], X[:, 2], expit(X[:, 3]), X[:, 4]
    eta, delta, sig, aH = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7]), expit(X[:, 8])
    wD, aD = X[:, 9], aH * expit(X[:, 10])
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
        r = a * G_[:, t] + (1 - a) * 0.7 * S_[:, t]
        QG = QG + aF * a * (r - QG); QS = QS + aF * (1 - a) * (r - QS)
        H = H + aH * (2 * a - 1 - H)
        D = D + aD * (2 * a - 1 - D)
    return (A_, P_) if rng is not None else nll


def to_X(U, m):
    X = np.zeros((U.shape[0], 11)); X[:, 8] = ONE_STEP
    for j, col in enumerate(FREE[m]):
        X[:, col] = np.exp(U[:, j]) if col in S22.LOGW else U[:, j]
    return X


def to_U(X, m):
    idx = FREE[m]
    U = np.column_stack([np.log(np.maximum(X[:, c], np.exp(-8))) if c in S22.LOGW else X[:, c] for c in idx])
    return np.clip(U, ULO[idx], UHI[idx])


# s21 的 map_fit / hessian / posterior 与 s22 的 em 在各自模块全局中查找前向模型与参数表：换成本步的版本
S21.run, S21.FREE, S21.ULO, S21.UHI, S21.to_X, S21.to_U = run, FREE, ULO, UHI, to_X, to_U
S22.FREE, S22.ULO, S22.UHI, S22.to_X = FREE, ULO, UHI, to_X


def pick_D(X, A_, P_):
    """(w_D, a_D) 的逐人网格起点：其余参数固定，取似然最大的一组。"""
    f = []
    for w, a in D_GRID:
        Xa = X.copy(); Xa[:, 9], Xa[:, 10] = w, a; f.append(run(Xa, A_, P_, OKP))
    best = np.argmin(np.array(f), 0)
    X = X.copy(); X[:, 9] = np.array([g[0] for g in D_GRID])[best]; X[:, 10] = np.array([g[1] for g in D_GRID])[best]
    return X


def widen(X9):
    return np.column_stack([X9, np.zeros((X9.shape[0], 2))])


def dataset(name):
    if name == "真实":
        return S20.A, S20.P, None
    g, rep = name.rsplit("_", 1)
    src = f"s23_真实_{g}.json" if g in NEW else f"s22_真实_{g}.json"
    X = np.array(json.loads((PL.OUT / src).read_text(encoding="utf-8"))["X"])
    Xg = X if X.shape[1] == 11 else widen(X)
    As, Ps = run(Xg, None, None, None, rng=np.random.default_rng(7000 + 100 * (OLD + NEW).index(g) + int(rep)))
    return As, Ps, Xg


def start(name, m, A_, P_):
    if name == "真实":
        r = json.loads((PL.OUT / f"s22_真实_{m.replace('D', '')}.json").read_text(encoding="utf-8"))
        X = pick_D(widen(np.array(r["X"])), A_, P_)
        U = to_U(X, m)
        return U, U.mean(0), np.maximum(U.var(0), 1e-2)
    src = f"s23_真实_{m}.json" if m in NEW else f"s22_真实_{m}.json"
    r = json.loads((PL.OUT / src).read_text(encoding="utf-8"))
    mu, s2 = np.array(r["mu"]), np.array(r["s2"])
    X = to_X(np.tile(mu, (n, 1)), m)
    if "D" in m:
        X = pick_D(X, A_, P_)
    return to_U(X, m), mu, s2


if __name__ == "__main__":
    mode = sys.argv[1]
    Ntot = float((T + OKP.sum(1)).sum())
    if mode == "拟合":
        name, m = sys.argv[2], sys.argv[3]
        logf = open(PL.W / "日志" / f"s23_{name}_{m}.log", "a", encoding="utf-8")
        log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
        A_, P_, Xg = dataset(name)
        log(f"数据集 {name}，模型 {m}")
        R = S22.em(m, A_, P_, *start(name, m, A_, P_), log)
        PL.save(dict(数据集=name, 模型=m, X真值=Xg, iBIC=float(-2 * R["LE"].sum() + 2 * len(FREE[m]) * np.log(Ntot)), **R),
                f"s23_{name}_{m}.json")
        log("完成")
    elif mode == "汇总":
        D = {m: json.loads((PL.OUT / (f"s23_真实_{m}.json" if m in NEW else f"s22_真实_{m}.json")).read_text(encoding="utf-8")) for m in OLD + NEW}
        ib = {m: D[m]["iBIC"] for m in D}
        best = min(ib, key=ib.get)
        par = {}
        for m in D:
            X = np.array(D[m]["X"]); X = X if X.shape[1] == 11 else widen(X)
            q = lambda v: [round(float(x), 3) for x in np.percentile(v, [25, 50, 75])]
            par[m] = {NAMES[c]: q(expit(X[:, c]) if c in (3, 5, 8) else expit(X[:, 8]) * expit(X[:, 10]) if c == 10 else X[:, c]) for c in FREE[m]}
        out = dict(iBIC={m: round(v, 1) for m, v in ib.items()}, 相对最优={m: round(v - ib[best], 1) for m, v in ib.items()}, 最优=best,
                   MF的边际贡献={"无慢漂移：HFB−HB": round(ib["HFB"] - ib["HB"], 1), "有慢漂移：HDFB−HDB": round(ib["HDFB"] - ib["HDB"], 1)},
                   EM轮数={m: D[m]["EM轮数"] for m in D}, **{"参数（四分位数；α 为概率尺度，aD 列为 α_D）": par})
        PL.save(out, "s23_真实_汇总.json")
        print(json.dumps({k: v for k, v in out.items() if not k.startswith("参数")}, ensure_ascii=False, indent=1))
    else:
        old = json.loads((PL.OUT / "s22_预测检查.json").read_text(encoding="utf-8"))
        res = dict(变量=old["变量"], 真实=old["真实"], **{m: old[m] for m in ("IB", "HB", "HFB")})
        for m in NEW:
            X = np.array(json.loads((PL.OUT / f"s23_真实_{m}.json").read_text(encoding="utf-8"))["X"])
            sims = [S22.lag_coefs(run(X, None, None, None, rng=np.random.default_rng(5000 + r))[0]) for r in range(20)]
            res[m] = dict(均值=np.mean(sims, 0).round(3).tolist(), 模拟间标准差=np.std(sims, 0).round(3).tolist())
            print(m, flush=True)
        PL.save(res, "s23_预测检查.json")
        print(json.dumps(res, ensure_ascii=False, indent=1))
