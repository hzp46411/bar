# -*- coding: utf-8 -*-
"""可靠性仲裁检验的模拟校准（快速、理想条件：个体参数与其他共用参数取生成真值，只估计两个可靠性系数）
  零：用 HRGP−rel（无可靠性仲裁）生成 40 套开环数据 → LR（df 2）分布、假阳性率
  检验力 / 恢复：用 HRGPR 生成 20 套 → LR 分布、检出率、θB_relB 与 θH_relH 的估计"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

FIT = lambda n: json.loads((L.OUT / "拟合" / f"{n}.json").read_text(encoding="utf-8"))
F1 = FIT("HRGPR"); SP1 = L.Spec(**F1["spec"]); NAMES = SP1.names()
iB, iH = NAMES.index("θB_relB"), NAMES.index("θH_relH")


def job(args):
    kind, k = args
    src = FIT("HRGP-rel") if kind == "零" else F1
    sp_src = L.Spec(**src["spec"]); X = np.array(src["X"])
    if kind == "零":                                         # HRGP−rel 的参数放进 HRGPR 的参数向量（可靠性系数 = 0）
        sh = dict(src["shared"]); sh["θB_relB"] = sh["θH_relH"] = 0.0
    else:
        sh = dict(src["shared"])
    phi_true = np.array([sh[q] for q in NAMES])
    A = np.zeros(L.A_REAL.shape)
    L.run(X, SP1, phi_true, A, L.G_REAL, L.S_REAL, L.ATT, rng=np.random.default_rng(7000 + k + (0 if kind == "零" else 500)),
          sigma=src["sigma"], eps=np.random.default_rng(8000 + k + (0 if kind == "零" else 500)).standard_normal(L.T))
    def phi_of(d):
        p = phi_true.copy(); p[iB], p[iH] = d; return p
    f = lambda d: float(L.run(X, SP1, phi_of(d), A, L.G_REAL, L.S_REAL, L.ATT).sum())
    r = minimize(f, [0.0, 0.0], method="L-BFGS-B", bounds=[(-1, 1)] * 2)
    _, ll1 = L.marginal_sigma(L.run(X, SP1, phi_of(r.x), A, L.G_REAL, L.S_REAL, L.ATT, out="z"), A)
    _, ll0 = L.marginal_sigma(L.run(X, SP1, phi_of([0, 0]), A, L.G_REAL, L.S_REAL, L.ATT, out="z"), A)
    return kind, float(2 * (ll1 - ll0)), r.x.tolist()


if __name__ == "__main__":
    jobs = [("零", k) for k in range(40)] + [("真值", k) for k in range(20)]
    with Pool(2) as pool:
        res = pool.map(job, jobs)
    out = {}
    for kind in ("零", "真值"):
        lr = np.array([r[1] for r in res if r[0] == kind]); est = np.array([r[2] for r in res if r[0] == kind])
        out[kind] = dict(套数=len(lr), LR均值=round(float(lr.mean()), 2), LR_95分位=round(float(np.percentile(lr, 95)), 2),
                        p小于05比例=round(float(np.mean(chi2.sf(np.maximum(lr, 0), 2) < .05)), 3),
                        θB_relB均值=round(float(est[:, 0].mean()), 4), θB_relB_SD=round(float(est[:, 0].std()), 4),
                        θH_relH均值=round(float(est[:, 1].mean()), 4), θH_relH_SD=round(float(est[:, 1].std()), 4))
    out["真值参数"] = dict(θB_relB=F1["shared"]["θB_relB"], θH_relH=F1["shared"]["θH_relH"])
    out["真实数据_LR"] = 28.85
    L.save_json(dict(out, 原始=res), L.OUT / "可靠性_模拟校准.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))
