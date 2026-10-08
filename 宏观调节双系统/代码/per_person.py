# -*- coding: utf-8 -*-
"""
群体层面 vs 逐人层面：4 种宏观状态 × 2 个系统（信念、惯性）
基线 = HRG（精修后）去掉该宏观状态的全部作用；每人单独估计 (δB_i, δH_i)：
    z = 其余部分 + 信念部分·e^{δB·M} + 惯性部分·e^{δH·M}
报告：全体共用 δ（群体层面）；逐人 δ 的中位数、方向比例、Wilcoxon；个体差异 LR（逐人 vs 共用，df = 2×99）。
零分布：用 HRG（人人相同的 θ）开环模拟 20 套，同样计算个体差异 LR 与单人显著人数，用来校准。
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from scipy.stats import wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

FIT = json.loads((L.OUT / "拟合" / "HRG.json").read_text(encoding="utf-8"))
SP = L.Spec(**FIT["spec"]); X = np.array(FIT["X"]); SH = dict(FIT["shared"]); STD = L.load_std()
N_SIM = 20


def parts(A, m):
    """去掉宏观状态 m 的全部作用后，返回 (其余部分, 信念部分, 惯性部分, 标准化的 M)，形状 (人, 轮)。"""
    sh = dict(SH); sh[f"θR_{m}"] = 0.0; sh[f"θG_{m}"] = 0.0
    phi = np.array([sh[k] for k in SP.names()])
    run = lambda Xv: L.run(Xv, SP, phi, A, L.G_REAL, L.S_REAL, L.ATT, out="z")
    z = run(X); Xb = X.copy(); Xb[:, 1] = 0; Xk = X.copy(); Xk[:, 2] = 0
    bp = z - run(Xb); hp = z - run(Xk)
    if m == "rel":
        Mz = (L.run(X, SP, phi, A, L.G_REAL, L.S_REAL, L.ATT, out="rel") - STD["rel"][0]) / STD["rel"][1]
    else:
        pm, _ = L.public_mods(L.ATT); Mz = np.broadcast_to((pm[m] - STD[m][0]) / STD[m][1], A.shape)
    return z - bp - hp, bp, hp, Mz


def nll(d, rest, bp, hp, M, a):
    z = rest + bp * np.exp(d[0] * M) + hp * np.exp(d[1] * M)
    return float(np.logaddexp(0, z).sum() - a @ z)


def analyze(A, m):
    rest, bp, hp, M = parts(A, m)
    n = len(A)
    f0 = np.array([nll([0, 0], rest[i], bp[i], hp[i], M[i], A[i]) for i in range(n)])
    D = np.zeros((n, 2)); fi = np.zeros(n)
    for i in range(n):
        r = minimize(nll, [0, 0], args=(rest[i], bp[i], hp[i], M[i], A[i]), method="L-BFGS-B", bounds=[(-3, 3)] * 2)
        D[i], fi[i] = r.x, r.fun
    tot = lambda d: sum(nll(d, rest[i], bp[i], hp[i], M[i], A[i]) for i in range(n))
    rc = minimize(tot, [0, 0], method="L-BFGS-B", bounds=[(-3, 3)] * 2)
    return dict(共用δ信念=float(rc.x[0]), 共用δ惯性=float(rc.x[1]), 共用LR=float(2 * (f0.sum() - rc.fun)),
                个体差异LR=float(2 * (rc.fun - fi.sum())), 单人显著=int((2 * (f0 - fi) > 5.99).sum()), D=D)


def sim_job(k):
    rng = np.random.default_rng(900 + k); eps = np.random.default_rng(950 + k).standard_normal(L.T)
    A = np.zeros(L.A_REAL.shape)
    L.run(X, SP, np.array([SH[q] for q in SP.names()]), A, L.G_REAL, L.S_REAL, L.ATT, rng=rng, sigma=FIT["sigma"], eps=eps)
    return {m: {kk: v for kk, v in analyze(A, m).items() if kk != "D"} for m in L.MODS}


if __name__ == "__main__":
    out = {}
    for m in L.MODS:
        r = analyze(L.A_REAL, m); D = r.pop("D")
        sgn = lambda v: dict(中位数=round(float(np.median(v)), 3), 为负比例=round(float(np.mean(v < 0)), 2), Wilcoxon_p=float(wilcoxon(v).pvalue))
        out[m] = dict(r, 逐人δ信念=sgn(D[:, 0]), 逐人δ惯性=sgn(D[:, 1]), δ_i=D)
        print(L.LABEL[m], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    with Pool(2) as pool:
        sims = pool.map(sim_job, range(N_SIM))
    for m in L.MODS:
        het = np.array([s[m]["个体差异LR"] for s in sims]); sig = np.array([s[m]["单人显著"] for s in sims])
        out[m]["零分布"] = dict(个体差异LR_均值=float(het.mean()), 个体差异LR_95=float(np.percentile(het, 95)),
                             个体差异_p=float((1 + (het >= out[m]["个体差异LR"]).sum()) / (N_SIM + 1)),
                             单人显著_均值=float(sig.mean()), 单人显著_95=float(np.percentile(sig, 95)))
    L.save_json(out, L.OUT / "群体与逐人.json")
    for m in L.MODS:
        o = out[m]
        print(f"\n{L.LABEL[m]}：共用 δ信念 {o['共用δ信念']:+.3f} δ惯性 {o['共用δ惯性']:+.3f}（LR {o['共用LR']:.1f}）",
              f"| 逐人 信念 中位 {o['逐人δ信念']['中位数']:+.3f} 负 {o['逐人δ信念']['为负比例']:.2f} p {o['逐人δ信念']['Wilcoxon_p']:.3f}",
              f"| 惯性 中位 {o['逐人δ惯性']['中位数']:+.3f} 负 {o['逐人δ惯性']['为负比例']:.2f} p {o['逐人δ惯性']['Wilcoxon_p']:.3f}",
              f"| 个体差异 LR {o['个体差异LR']:.0f}（零 均值 {o['零分布']['个体差异LR_均值']:.0f}，95% {o['零分布']['个体差异LR_95']:.0f}，p {o['零分布']['个体差异_p']:.3f}）",
              f"| 单人显著 {o['单人显著']}（零 均值 {o['零分布']['单人显著_均值']:.1f}）")
