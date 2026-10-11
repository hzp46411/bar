# -*- coding: utf-8 -*-
"""
第 44 步：联合估计本身能否恢复 γ、σ（3.5.3）
  真值：s38 中 HB 的联合估计（个体参数 + γ、σ）；另做 γ = σ = 0 的零假设数据（检查假阳性与偏差）
  生成（开环，状态取真实历史，与 s22 的假数据相同）：
    z_it = d_it(θ_i) − γ·(N_{t−1} − 60)/10 + σ·ε_t，ε_t ~ N(0, 1)（同一轮所有人共享）；选择 ~ Bernoulli(expit z)；
    预测 ~ round(M + σ_P·N(0, 1))，只在真实数据中有效预测的轮次计入似然（与 s22 相同）
  估计（与真实数据完全相同的流程）：
    1 不含共同成分的 HB 层级拟合（s22 的起点与 em）
    2 两步法：个体参数固定，估计 γ、σ（s36 的 estimate，含似然比检验）
    3 联合估计：s38 的条件期望最大化（ε 后验压缩为 K = 8 个分位点；最多 6 轮）
  报告：两步法与联合估计的 γ、σ（均值、SD、偏差）；零假设数据中的估计值与两步法似然比检验的拒绝率；
        个体参数 δ、w_B、w_I 的恢复（Spearman）
用法：python3 s44_联合估计恢复.py 拟合 <真|零> <编号>  |  汇总
输出：结果/s44_<真|零>_<编号>.json、结果/s44_联合估计恢复.json
"""
import sys, json, time, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr
from scipy.stats import spearmanr
import pred_lib as PL


def _load(fname, mod):
    spec = importlib.util.spec_from_file_location(mod, str(pathlib.Path(__file__).with_name(fname)))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    return M


S38 = _load("s38_共同因素联合估计.py", "s38")
S22, S36, S20 = S38.S22, S38.S36, S38.S20
T, n, G, S, CROWD_PREV, LAG, K, OFF = S38.T, S38.n, S38.G, S38.S, S38.CROWD_PREV, S38.LAG, S38.K, S38.OFF
M = "HB"


def forward(X, C=None, A_=None, rng=None):
    """HB 的前向计算（公式同 s22 的 run）。rng 不为空时生成 (A, P)，C 为每轮的共同偏移；否则返回给定 A_ 下的个体 logit (行, T)。"""
    rows = X.shape[0]
    b, wI, wF, aF, wB = X[:, 0], X[:, 1], X[:, 2], expit(X[:, 3]), X[:, 4]
    eta, delta, sig, aH = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7]), expit(X[:, 8])
    Mn, Mc = 60 + delta / 2, 60 - delta / 2
    QG, QS = np.full(rows, 0.5), np.full(rows, 0.35)
    H = np.zeros(rows); Z = np.zeros((rows, T))
    if rng is not None:
        A_ = np.zeros((rows, T)); P_ = np.zeros((rows, T))
    for t in range(T):
        sp = CROWD_PREV[t]
        Mp = Mc if sp else Mn
        p = ndtr((60.5 - Mp) / sig)
        z = b + wI * H + wF * (QG - QS) + wB * (1.7 * p - 0.7)
        Z[:, t] = z
        if rng is not None:
            A_[:, t] = (rng.random(rows) < expit(z + C[t])).astype(float)
            P_[:, t] = np.round(Mp + sig * rng.standard_normal(rows))
        a = A_[:, t]
        if sp:
            Mc = Mc + eta * (S20.N[t] - Mc)
        else:
            Mn = Mn + eta * (S20.N[t] - Mn)
        r = a * G[:, t] + (1 - a) * 0.7 * S[:, t]
        QG = QG + aF * a * (r - QG); QS = QS + aF * (1 - a) * (r - QS)
        H = H + aH * (2 * a - 1 - H)
    return (A_, P_) if rng is not None else Z


def posterior_offsets(d, A_, g, s):
    """同 s38 的 posterior_offsets，但选择矩阵由参数给出。"""
    z = d[None] - g * LAG + s * S36.NODES[:, None, None]
    L = (A_[None] * z - np.logaddexp(0, z)).sum(1)
    L = L - L.max(0); w = S36.W[:, None] * np.exp(L); w = w / w.sum(0)
    cdf = np.cumsum(w, 0); qs = (np.arange(K) + 0.5) / K
    eps = np.array([[S36.NODES[np.searchsorted(cdf[:, t], q)] for q in qs] for t in range(T)])
    return -g * LAG[:, None] + s * eps


def fit(kind, rep):
    tag = f"{kind}_{rep}"
    logf = open(PL.W / "日志" / f"s44_{tag}.log", "a", encoding="utf-8")
    log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
    tr = json.loads((PL.OUT / "s38_联合估计_HB.json").read_text(encoding="utf-8"))
    Xt = np.array(tr["X"]); g0, s0 = (tr["γ"], tr["σ"]) if kind == "真" else (0.0, 0.0)
    rng = np.random.default_rng((7000 if kind == "真" else 7500) + rep)
    C = -g0 * LAG + s0 * rng.standard_normal(T)
    A_, P_ = forward(Xt, C=C, rng=rng)
    log(f"{tag}：真值 γ {g0:.4f}，σ {s0:.4f}；去的比例 {A_.mean():.3f}")
    t0 = time.time()
    OFF[:] = 0.0
    U, mu, s2 = S22.start("假", M, A_, P_)
    R = S22.em(M, A_, P_, U, mu, s2, log)
    X9, mu, s2 = R["X"], R["mu"], R["s2"]; U = S22.to_U(X9, M); X_two = X9.copy()
    E = S36.estimate(forward(X9, A_=A_), A_)
    two = dict(γ=E["γ+σ"][0], σ=E["γ+σ"][1], 似然比检验=S36.lr_tests(E))
    g, s = two["γ"], two["σ"]
    log(f"  两步法：γ {g:.4f}，σ {s:.4f}")
    hist = [dict(轮=0, γ=g, σ=s)]
    for it in range(1, 7):
        OFF[:] = posterior_offsets(forward(X9, A_=A_), A_, g, s)
        R = S22.em(M, A_, P_, U, mu, s2, log, max_it=10)
        X9, mu, s2 = R["X"], R["mu"], R["s2"]; U = S22.to_U(X9, M)
        e = S36.estimate(forward(X9, A_=A_), A_)["γ+σ"]
        dg, ds = abs(e[0] - g), abs(e[1] - s); g, s = e[0], e[1]
        hist.append(dict(轮=it, γ=g, σ=s)); log(f"  联合第 {it} 轮：γ {g:.4f}，σ {s:.4f}")
        if dg < 0.005 and ds < 0.005:
            break
    rec = lambda Xe: {nm: round(float(spearmanr(Xt[:, c], Xe[:, c])[0]), 3) for nm, c in (("δ", 6), ("w_B", 4), ("w_I", 1), ("b", 0))}
    PL.save(dict(数据=tag, 真值=dict(γ=g0, σ=s0), 两步法=two, 联合=dict(γ=g, σ=s, 轮数=len(hist) - 1), 历史=hist,
                 个体参数恢复=dict(两步法=rec(X_two), 联合=rec(X9)), 用时秒=round(time.time() - t0)), f"s44_{tag}.json")
    log("完成")


def summary():
    out = {}
    for kind in ("真", "零"):
        rs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(PL.OUT.glob(f"s44_{kind}_*.json"))]
        if not rs:
            continue
        tv = rs[0]["真值"]; row = dict(套数=len(rs), 真值=tv)
        for meth in ("两步法", "联合"):
            for par in ("γ", "σ"):
                v = np.array([r[meth][par] for r in rs])
                row[f"{meth}_{par}"] = dict(均值=round(float(v.mean()), 4), SD=round(float(v.std(ddof=1)), 4),
                                            偏差=round(float(v.mean() - tv[par]), 4),
                                            范围=[round(float(v.min()), 4), round(float(v.max()), 4)])
        for par in ("γ（在 σ 之上）", "σ（在 γ 之上）"):
            row[f"两步法似然比检验_{par}_拒绝率"] = round(float(np.mean([r["两步法"]["似然比检验"][par]["p"] < 0.05 for r in rs])), 3)
        row["个体参数恢复_联合（中位数）"] = {k: round(float(np.median([r["个体参数恢复"]["联合"][k] for r in rs])), 3)
                                    for k in rs[0]["个体参数恢复"]["联合"]}
        out[kind] = row
    PL.save(out, "s44_联合估计恢复.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "拟合":
        fit(sys.argv[2], int(sys.argv[3]))
    else:
        summary()
