# -*- coding: utf-8 -*-
"""双重分离检验（两步：个体参数固定在 HRG 精修值，只重估共用参数；含 σ 的边际似然）
对每个宏观状态 m：信念指数 θB = θG + θR/2，惯性指数 θH = θG − θR/2。
  约束 A：θB_m = 0（信念不随 m 变）；约束 B：θH_m = 0（惯性不随 m 变）；约束 C：θB_m = θH_m（纯增益，比例不变）
近似 95% 区间：由 LR 反推标准误 |θ| / sqrt(LR)。"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

fit = json.loads((L.OUT / "拟合" / "HRG.json").read_text(encoding="utf-8"))
sp = L.Spec(**fit["spec"]); X = np.array(fit["X"]); names = sp.names()
phi_hat = np.array([fit["shared"][k] for k in names]); args = (L.A_REAL, L.G_REAL, L.S_REAL, L.ATT)
f_full = float(L.run(X, sp, phi_hat, *args).sum()); _, ll_full = L.marginal_sigma(L.run(X, sp, phi_hat, *args, out="z"), L.A_REAL)
out = {}
for m in L.MODS:
    iR, iG = names.index(f"θR_{m}"), names.index(f"θG_{m}")
    tB, tH = phi_hat[iG] + phi_hat[iR] / 2, phi_hat[iG] - phi_hat[iR] / 2
    row = dict(θ信念=round(float(tB), 4), θ惯性=round(float(tH), 4))
    for kind in ("A", "B", "C"):
        free = [j for j in range(len(names)) if j != iG]
        def build(p, kind=kind):
            phi = phi_hat.copy(); phi[free] = p
            phi[iG] = {"A": -phi[iR] / 2, "B": phi[iR] / 2}.get(kind, phi[iG])
            if kind == "C": phi[iR] = 0.0
            return phi
        if kind == "C":
            free = [j for j in range(len(names)) if j != iR]
        r = minimize(lambda p: float(L.run(X, sp, build(p), *args).sum()), phi_hat[free], method="L-BFGS-B", bounds=[sp.bounds()[j] for j in free])
        _, ll = L.marginal_sigma(L.run(X, sp, build(r.x), *args, out="z"), L.A_REAL)
        LR = max(2 * (ll_full - ll), 1e-9)
        row[{"A": "信念不变_LR", "B": "惯性不变_LR", "C": "纯增益_LR"}[kind]] = round(LR, 2)
        row[{"A": "信念不变_p", "B": "惯性不变_p", "C": "纯增益_p"}[kind]] = float(chi2.sf(LR, 1))
    seB = abs(tB) / np.sqrt(row["信念不变_LR"]); seH = abs(tH) / np.sqrt(row["惯性不变_LR"])
    row["信念95%区间"] = [round(float(tB - 1.96 * seB), 3), round(float(tB + 1.96 * seB), 3)]
    row["惯性95%区间"] = [round(float(tH - 1.96 * seH), 3), round(float(tH + 1.96 * seH), 3)]
    out[L.LABEL[m]] = row
    print(L.LABEL[m], row, flush=True)
L.save_json(out, L.OUT / "双重分离_两步.json")
