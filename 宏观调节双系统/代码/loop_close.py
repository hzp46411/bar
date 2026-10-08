# -*- coding: utf-8 -*-
"""
闭环第 ③ 环：补上两个微观成分，看能否生成真实宏观的状态依赖（φs：稳定之后的回拉；φn：大偏离的非线性）
  1 分级反应改为分段样条（两步：HRGPR 其余参数固定）
  2 稳定推力因人而异：逐人 ψ_i，再经验贝叶斯收缩
  然后闭环模拟（每种 300 次）：HRGPR / +样条 / +逐人ψ / +两者 / 关宏观调节，报告 φ0、φs、φn 与宏观指纹
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import robust as R
import abm_arb as Ab
import state_ar as SA

A = R.A; out = {}
# ---------- 1 样条分级反应 ----------
rest, bp, hp = R.parts([])
rest0 = rest - R.SH["lam"] * R.LAG[None]
spl = lambda q: rest0 + (q @ R.basis)[None] + bp + hp
q, ll_s = R.fit(spl, 5, np.r_[R.SH["lam"], 0, 0, 0, 0])
_, ll_l = L.marginal_sigma(rest + bp + hp, A)
out["样条"] = dict(系数=np.round(q, 4).tolist(), 相对线性_LR_df4=round(2 * (ll_s - ll_l), 2), p=float(chi2.sf(max(2 * (ll_s - ll_l), 0), 4)))
print("样条", out["样条"], flush=True)

# ---------- 2 逐人稳定推力（两步 + 经验贝叶斯收缩）----------
rest, bp, hp = R.parts(["ψ_stab"])
base = rest + bp + hp; F = R.c * R.Mst
psi_i = np.zeros(R.n); se_i = np.zeros(R.n)
for i in range(R.n):
    nl = lambda p: float(np.logaddexp(0, base[i] + p * F[i]).sum() - A[i] @ (base[i] + p * F[i]))
    r = minimize_scalar(nl, bounds=(-3, 3), method="bounded"); psi_i[i] = r.x
    h = 1e-3; curv = (nl(r.x + h) - 2 * r.fun + nl(r.x - h)) / h ** 2
    se_i[i] = 1 / np.sqrt(max(curv, 1e-6))
mu = np.average(psi_i, weights=1 / se_i ** 2)
tau2 = max(np.var(psi_i) - np.mean(se_i ** 2), 1e-6)
psi_eb = mu + tau2 / (tau2 + se_i ** 2) * (psi_i - mu)
kap = R.X[:, 2]
out["逐人ψ"] = dict(均值=round(float(mu), 4), 个体SD=round(float(np.sqrt(tau2)), 4), 收缩后中位数=round(float(np.median(psi_eb)), 4),
                  为正比例=round(float(np.mean(psi_eb > 0)), 2), 重复型均值=round(float(psi_eb[kap > 0].mean()), 4), 交替型均值=round(float(psi_eb[kap < 0].mean()), 4))
print("逐人ψ", out["逐人ψ"], flush=True)

# ---------- 3 闭环 ----------
Ab.VARIANTS["L_HRGPR"] = ("HRGPR", {})
Ab.VARIANTS["L_样条"] = ("HRGPR", {"lam_coef": q})
Ab.VARIANTS["L_逐人ψ"] = ("HRGPR", {"psi_stab_i": psi_eb})
Ab.VARIANTS["L_样条+逐人ψ"] = ("HRGPR", {"lam_coef": q, "psi_stab_i": psi_eb})
Ab.VARIANTS["L_关宏观调节"] = ("HRGPR", {"no_macro": True})
Ab.VARIANTS["L_样条+关宏观调节"] = ("HRGPR", {"lam_coef": q, "no_macro": True})
orig_fp = Ab.fingerprint
def fp_with_N(N, A_):
    f = orig_fp(N, A_); f["phi"] = SA.coefs(N).tolist(); return f
Ab.fingerprint = fp_with_N

def job(args):
    return Ab.abm_one(args)

if __name__ == "__main__":
    real = orig_fp(L.ATT, L.A_REAL); real_phi = SA.coefs(L.ATT)
    out["真实"] = dict(φ0=round(real_phi[0], 3), φs=round(real_phi[1], 3), φn=round(real_phi[2], 3), sd=round(real["sd"], 2),
                     acf1=round(real["acf1"], 3), sq_acf1=round(real["sq_acf1"], 3), switch=round(real["switch"], 3))
    names = ["L_HRGPR", "L_样条", "L_逐人ψ", "L_样条+逐人ψ", "L_关宏观调节", "L_样条+关宏观调节"]
    with Pool(4) as pool:
        res = pool.map(job, [(nm, 950000 + 13 * k + 7 * i) for i, nm in enumerate(names) for k in range(300)])
    for nm in names:
        fps = [f for n_, f in res if n_ == nm]
        P = np.array([f["phi"] for f in fps])
        row = {}
        for j, lab in enumerate(["φ0", "φs", "φn"]):
            row[lab] = dict(均值=round(float(P[:, j].mean()), 3), 真实分位=round(float(np.mean(P[:, j] <= real_phi[j])), 3))
        for k_ in ("sd", "acf1", "sq_acf1", "switch"):
            v = np.array([f[k_] for f in fps]); row[k_] = dict(均值=round(float(v.mean()), 3), 真实分位=round(float(np.mean(v <= real[k_])), 3))
        out[nm] = row
        print(nm, row, flush=True)
    print("真实", out["真实"])
    L.save_json(dict(out, 逐人ψ值=psi_eb), L.OUT / "闭环第三环.json")
