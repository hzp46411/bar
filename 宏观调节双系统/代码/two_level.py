# -*- coding: utf-8 -*-
"""双层联合约束：沿稳定推力 ψ 的大小，计算（1）个体层含 σ 的对数似然（两步，样条分级反应，其余参数固定）；
（2）宏观层 φs 的合成似然（闭环 200 次，正态近似）。看两层是否存在共同可接受的 ψ。"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import robust as R
import abm_arb as Ab
import state_ar as SA

Q = np.array(json.loads((L.OUT / "闭环第三环.json").read_text(encoding="utf-8"))["样条"]["系数"])
rest, bp, hp = R.parts(["ψ_stab"])
rest0 = rest - R.SH["lam"] * R.LAG[None] + (Q @ R.basis)[None]
GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
real_phis = SA.coefs(L.ATT)[1]
orig = Ab.fingerprint
Ab.fingerprint = lambda N, A_: {"phis": float(SA.coefs(N)[1])}

def job(args):
    psi, k = args
    Ab.VARIANTS["TL"] = ("HRGPR", {"lam_coef": Q, "psi_stab_i": np.full(R.n, psi)})
    return psi, Ab.abm_one(("TL", 970000 + k))[1]["phis"]

if __name__ == "__main__":
    micro = {}
    for psi in GRID:
        _, ll = L.marginal_sigma(rest0 + bp + hp + psi * R.c * R.Mst, R.A); micro[psi] = ll
    with Pool(4) as pool:
        res = pool.map(job, [(psi, k) for psi in GRID for k in range(200)])
    rows = []
    m0 = max(micro.values())
    for psi in GRID:
        v = np.array([r for p, r in res if p == psi])
        macro_ll = float(norm(v.mean(), v.std()).logpdf(real_phis))
        rows.append(dict(ψ=psi, 个体层对数似然_相对最优=round(micro[psi] - m0, 2), 宏观φs均值=round(float(v.mean()), 3),
                         真实φs分位=round(float(np.mean(v <= real_phis)), 3), 宏观合成对数似然=round(macro_ll, 2)))
        print(rows[-1], flush=True)
    L.save_json(dict(真实φs=real_phis, 网格=rows), L.OUT / "双层约束_稳定推力.json")
