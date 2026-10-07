# -*- coding: utf-8 -*-
"""偏离对惯性的调节：是每个人身上都有，还是少数人/群体层面的？
在 HRG 拟合上把"偏离对比例的作用"设为 0 作为基线，再给每个人单独估计一个交互项 δ_i·c_it·dev_z(t)
（δ_i < 0 = 上一轮偏离越大，这个人自己的惯性越弱），并检验 δ_i 是否人人相同。"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import chi2, wilcoxon
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

fit = json.loads((L.OUT / "拟合" / "HRG.json").read_text(encoding="utf-8"))
sp = L.Spec(**fit["spec"]); X = np.array(fit["X"]); sh = dict(fit["shared"])
sh0 = dict(sh); sh0["θR_dev"] = 0.0
phi0 = np.array([sh0[k] for k in sp.names()])
z0 = L.run(X, sp, phi0, L.A_REAL, L.G_REAL, L.S_REAL, L.ATT, out="z")
A = L.A_REAL
aH = float(expit(sh["logit_aH"]))
H = np.full(A.shape[0], .5); c = np.zeros(A.shape)
for t in range(L.T):
    c[:, t] = 2 * H - 1; H += aH * (A[:, t] - H)
std = L.load_std(); pm, _ = L.public_mods(L.ATT)
devz = (pm["dev"] - std["dev"][0]) / std["dev"][1]
F = c * devz[None]                                                     # 每人自己的习惯 × 公共的偏离
nll = lambda z, a: float(np.logaddexp(0, z).sum() - a @ z)
d_i, lr_i = np.zeros(len(A)), np.zeros(len(A))
for i in range(len(A)):
    f0 = nll(z0[i], A[i]); r = minimize_scalar(lambda d: nll(z0[i] + d * F[i], A[i]), bounds=(-3, 3), method="bounded")
    d_i[i], lr_i[i] = r.x, 2 * (f0 - r.fun)
# 全体共用一个 δ
tot = lambda d: sum(nll(z0[i] + d * F[i], A[i]) for i in range(len(A)))
r = minimize_scalar(tot, bounds=(-3, 3), method="bounded"); d_common = r.x
lr_common = 2 * (tot(0.0) - r.fun)
lr_het = 2 * (r.fun - sum(nll(z0[i] + d_i[i] * F[i], A[i]) for i in range(len(A))))
kap = X[:, 2]; beta = X[:, 1]
grp = lambda m: dict(人数=int(m.sum()), δ中位数=round(float(np.median(d_i[m])), 3), δ为负的比例=round(float(np.mean(d_i[m] < 0)), 2))
out = dict(
    共用δ=round(float(d_common), 4), 共用δ的LR_df1=round(float(lr_common), 2), p=float(chi2.sf(lr_common, 1)),
    逐人δ中位数=round(float(np.median(d_i)), 4), δ为负的人数=int((d_i < 0).sum()), 逐人Wilcoxon_p=float(wilcoxon(d_i).pvalue),
    个体差异检验_LR=round(float(lr_het), 1), 个体差异_df=len(A) - 1, 个体差异_p=float(chi2.sf(lr_het, len(A) - 1)),
    单人显著_LR大于3_84=int((lr_i > 3.84).sum()),
    分组={"惯性为正（重复型）": grp(kap > 0), "惯性为负（交替型）": grp(kap < 0),
          "惯性较强（|κ| ≥ 中位数）": grp(np.abs(kap) >= np.median(np.abs(kap))), "惯性较弱": grp(np.abs(kap) < np.median(np.abs(kap))),
          "外推者 β>0": grp(beta > 0), "逆向者 β<0": grp(beta < 0)})
L.save_json(dict(out, δ_i=d_i, LR_i=lr_i), L.OUT / "偏离调节_逐人.json")
print(json.dumps(out, ensure_ascii=False, indent=1))
