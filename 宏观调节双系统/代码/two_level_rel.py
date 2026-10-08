# -*- coding: utf-8 -*-
"""
双层联合约束 · 可靠性仲裁的强度
  第三环发现：闭环里仍对不上的三个宏观量（长期稳定后的回调 φs、换人率自相关、换人 → 下轮偏离）
  都随"可靠性仲裁"变强而单调靠近真实值；真实数据一步预测也显示信念通道的宏观足迹约为模型的 1.5–2 倍。
  这里沿可靠性强度倍数 k（θB_relB、θH_relH 同乘 k；以及只乘信念一侧）计算：
    个体层：含 σ 的对数似然（两步：个体参数与其余共用参数固定）
    宏观层：多元正态合成似然（Wood 2010），统计量 = 7 个闭环宏观量，每格 B 次闭环
输出：结果/双层约束_可靠性_<模型>.json
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.stats import multivariate_normal
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import abm_arb as Ab
import third_arrow as TA

MODEL = sys.argv[1] if len(sys.argv) > 1 else "HRGPRS"; B = int(sys.argv[2]) if len(sys.argv) > 2 else 300
FIT = json.loads((L.OUT / "拟合" / f"{MODEL}.json").read_text(encoding="utf-8"))
SP = L.Spec(**FIT["spec"]); X = np.array(FIT["X"]); SH = dict(FIT["shared"]); NAMES = SP.names()
A, G, S, N = L.A_REAL, L.G_REAL, L.S_REAL, L.ATT
STD = L.load_std()
STATS = ["φ0", "φs", "φn", "sq_acf1", "换人率_sd", "换人率_acf1", "换人→下轮偏离"]
GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]


def parts(zero):
    sh = dict(SH)
    for k in zero:
        sh[k] = 0.0
    phi = np.array([sh[k] for k in NAMES])
    run = lambda Xv: L.run(Xv, SP, phi, A, G, S, N, out="z")
    z = run(X); Xb = X.copy(); Xb[:, 1] = 0; Xk = X.copy(); Xk[:, 2] = 0
    bp = z - run(Xb); hp = z - run(Xk)
    return z - bp - hp, bp, hp


def job(args):
    lab, kB, kH, seed = args
    N_, A_, _ = Ab.simulate(MODEL, {"phi_scale": {"θB_relB": kB, "θH_relH": kH}}, seed)
    s = TA.structure(N_, A_)
    return lab, [s[k] for k in STATS]


if __name__ == "__main__":
    phi = np.array([SH[k] for k in NAMES])
    MB = (L.run(X, SP, phi, A, G, S, N, out="relB") - STD["relB"][0]) / STD["relB"][1]
    MH = (L.run(X, SP, phi, A, G, S, N, out="relH") - STD["relH"][0]) / STD["relH"][1]
    rest, bp, hp = parts(["θB_relB", "θH_relH"])
    grid = [(f"同乘×{k}", k, k) for k in GRID] + [(f"只乘信念×{k}", k, 1.0) for k in GRID if k != 1.0]
    micro = {}
    for lab, kB, kH in grid:
        z = rest + bp * np.exp(kB * SH["θB_relB"] * MB) + hp * np.exp(kH * SH["θH_relH"] * MH)
        micro[lab] = L.marginal_sigma(z, A)[1]
    with Pool(4) as pool:
        res = pool.map(job, [(lab, kB, kH, 990000 + 7919 * i + 101 * j) for j, (lab, kB, kH) in enumerate(grid) for i in range(B)], chunksize=20)
    obs = TA.structure(N, A); o = np.array([obs[k] for k in STATS])
    m0 = micro["同乘×1.0"]
    rows = []
    for lab, kB, kH in grid:
        V = np.array([v for l, v in res if l == lab])
        mu, C = V.mean(0), np.cov(V.T)
        macro = float(multivariate_normal(mu, C, allow_singular=True).logpdf(o))
        rows.append(dict(变体=lab, 信念倍数=kB, 惯性倍数=kH, 个体层对数似然_相对原估计=round(micro[lab] - m0, 2),
                         宏观合成对数似然=round(macro, 2), **{f"{k}_均值": round(float(mu[j]), 3) for j, k in enumerate(STATS)}))
    mref = [r for r in rows if r["变体"] == "同乘×1.0"][0]["宏观合成对数似然"]
    for r in rows:
        r["宏观_相对原估计"] = round(r["宏观合成对数似然"] - mref, 2)
        r["两层合计_相对原估计"] = round(r["宏观_相对原估计"] + r["个体层对数似然_相对原估计"], 2)
        print(r, flush=True)
    L.save_json(dict(模型=MODEL, B=B, 真实={k: obs[k] for k in STATS}, 网格=rows), L.OUT / f"双层约束_可靠性_{MODEL}.json")
