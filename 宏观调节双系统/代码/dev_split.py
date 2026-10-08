# -*- coding: utf-8 -*-
"""偏离对两个系统的作用分开检验（快速两步：个体参数固定在 HRG 估计，只重估共用参数）。
信念指数 = θG + θR/2，惯性指数 = θG − θR/2。约束：
  A 信念不随偏离变（θG_dev = −θR_dev/2）  B 惯性不随偏离变（θG_dev = θR_dev/2）  C 两者都不变（θR_dev = θG_dev = 0）"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

fit = json.loads((L.OUT / "拟合" / "HRG.json").read_text(encoding="utf-8"))
sp = L.Spec(**fit["spec"]); X = np.array(fit["X"]); names = sp.names()
phi_hat = np.array([fit["shared"][k] for k in names])
iR, iG = names.index("θR_dev"), names.index("θG_dev")
args = (L.A_REAL, L.G_REAL, L.S_REAL, L.ATT)


def fit_constrained(kind):
    free = [j for j in range(len(names)) if j != iG] if kind in ("A", "B") else [j for j in range(len(names)) if j not in (iR, iG)]
    def build(p):
        phi = phi_hat.copy(); phi[free] = p
        if kind == "A": phi[iG] = -phi[iR] / 2
        elif kind == "B": phi[iG] = phi[iR] / 2
        elif kind == "C": phi[iR] = phi[iG] = 0.0
        return phi
    f = lambda p: float(L.run(X, sp, build(p), *args).sum())
    r = minimize(f, phi_hat[free], method="L-BFGS-B", bounds=[sp.bounds()[j] for j in free])
    return build(r.x), r.fun


f_full = float(L.run(X, sp, phi_hat, *args).sum())
s_full, ll_full = L.marginal_sigma(L.run(X, sp, phi_hat, *args, out="z"), L.A_REAL)
out = {"完整HRG": dict(NLL=round(f_full, 3), θ信念_dev=round(float(phi_hat[iG] + phi_hat[iR] / 2), 4), θ惯性_dev=round(float(phi_hat[iG] - phi_hat[iR] / 2), 4))}
for kind, lab, df in (("A", "信念不随偏离变", 1), ("B", "惯性不随偏离变", 1), ("C", "两者都不随偏离变", 2)):
    phi, fc = fit_constrained(kind)
    s, ll = L.marginal_sigma(L.run(X, sp, phi, *args, out="z"), L.A_REAL)
    LRs = 2 * (ll_full - ll)
    out[lab] = dict(NLL=round(fc, 3), LR条件=round(2 * (fc - f_full), 2), LR含σ=round(LRs, 2), df=df, p=float(chi2.sf(max(LRs, 0), df)),
                    约束下_θ信念_dev=round(float(phi[iG] + phi[iR] / 2), 4), 约束下_θ惯性_dev=round(float(phi[iG] - phi[iR] / 2), 4))
# 具体权重变化：上一轮偏离 1 人 → 10 人
std = L.load_std()
zlo, zhi = (0.1 - std["dev"][0]) / std["dev"][1], (1.0 - std["dev"][0]) / std["dev"][1]
tB, tH = phi_hat[iG] + phi_hat[iR] / 2, phi_hat[iG] - phi_hat[iR] / 2
out["偏离1人→10人"] = dict(信念权重倍数=round(float(np.exp(tB * (zhi - zlo))), 3), 惯性权重倍数=round(float(np.exp(tH * (zhi - zlo))), 3),
                           信念比惯性倍数=round(float(np.exp((tB - tH) * (zhi - zlo))), 3))
L.save_json(out, L.OUT / "偏离_分系统检验_两步.json")
print(json.dumps(out, ensure_ascii=False, indent=1))
