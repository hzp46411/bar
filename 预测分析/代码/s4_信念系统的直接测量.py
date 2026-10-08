# -*- coding: utf-8 -*-
"""第 4 步：预测作为信念系统的直接测量，与 BBL（选择模型）对接
  1 类型的汇聚效度：由预测得到的个人类型（P−60 对上一轮挤 / 不挤与偏离大小的反应）与由选择估计的 BBL β 的符号是否一致
      BBL：β > 0 外推（上一轮不挤 → 相信不挤 → 去）；β < 0 逆向。由预测：上一轮挤之后预测更高 = 外推。
  2 人内：模型信念 V_it（由 β 定向后）与陈述的预测是否同步
  3 符号还是大小：预测对上一轮人数的反应主要来自"挤 / 不挤"还是"挤多少"（逐人 F 检验）
  4 分级反应 λ 是否经由信念：在选择模型（HRGPR 两步，个体参数固定）中加入陈述的预测类别，λ 是否变小
输出：结果/s4_信念系统的直接测量.json
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import pred_lib as PL
L = PL.L
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
ms = PL.model_states("HRGPR"); beta = ms["beta"]; V = ms["V"]
NP = PL.NPREV; sl = slice(1, None)
d = (NP - 60)[sl] / 10; sg = np.sign(NP - 60.5)[sl]
out = {}
# ---------- 1 类型 ----------
sgn_coef = np.zeros(n); mag_coef = np.zeros(n); F_sign = np.zeros(n); F_mag = np.zeros(n); p_sign = np.zeros(n)
for i in range(n):
    m = OK[i, 1:]; y = P[i, 1:][m] - 60
    X = np.column_stack([np.ones(m.sum()), sg[m], d[m]])
    b, res, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b; s2 = e @ e / (len(y) - 3)
    cov = s2 * np.linalg.inv(X.T @ X); sgn_coef[i], mag_coef[i] = b[1], b[2]
    p_sign[i] = 2 * stats.t.sf(abs(b[1] / np.sqrt(cov[1, 1])), len(y) - 3)
    # 单独的符号模型 vs 单独的大小模型的拟合（R²）
    for j, (Xa) in enumerate([np.column_stack([np.ones(m.sum()), sg[m]]), np.column_stack([np.ones(m.sum()), d[m]])]):
        bb, *_ = np.linalg.lstsq(Xa, y, rcond=None); r2 = 1 - np.var(y - Xa @ bb) / np.var(y)
        (F_sign if j == 0 else F_mag)[i] = r2
pred_type = np.sign(sgn_coef)                               # +1：上一轮挤之后预测更高（外推）
choice_type = np.sign(beta)                                  # +1：BBL 外推
agree = (pred_type == choice_type)
strong = p_sign < .05
out["类型的汇聚效度"] = {
    "预测外推的人数（上一轮挤后预测更高）": int((pred_type > 0).sum()), "预测逆向的人数": int((pred_type < 0).sum()),
    "选择 BBL β > 0 的人数": int((choice_type > 0).sum()),
    "类型一致的比例（全部 100 人）": float(agree.mean()),
    "类型一致的比例（预测类型显著的人）": float(agree[strong].mean()), "预测类型显著的人数": int(strong.sum()),
    "Fisher 精确检验 p（全部）": float(stats.fisher_exact(np.array([[((pred_type > 0) & (choice_type > 0)).sum(), ((pred_type > 0) & (choice_type < 0)).sum()],
                                                               [((pred_type < 0) & (choice_type > 0)).sum(), ((pred_type < 0) & (choice_type < 0)).sum()]]))[1]),
    "Spearman(预测的符号系数, β)": [float(x) for x in stats.spearmanr(sgn_coef, beta)]}
# ---------- 2 人内同步 ----------
# BBL 的"去的期望收益"V 越大 → 越相信不挤；对外推者（β>0）应对应预测越低。用 β 的符号定向：Vb = sign(β)·V
r_i = np.array([stats.pearsonr(np.sign(beta[i]) * V[i, 1:][OK[i, 1:]], P[i, 1:][OK[i, 1:]])[0] for i in range(n)])
out["人内同步：corr(定向后的模型信念, 预测)"] = {"中位数": float(np.median(r_i)), "为负（方向正确）的人数": int((r_i < 0).sum()),
                                     "Wilcoxon p": float(stats.wilcoxon(r_i).pvalue)}
# ---------- 3 符号还是大小 ----------
out["符号还是大小"] = {"逐人：只用符号的 R² 中位数": float(np.median(F_sign)), "逐人：只用大小的 R² 中位数": float(np.median(F_mag)),
                  "符号模型更好的人数": int((F_sign > F_mag).sum()),
                  "逐人系数中位数：符号（±1）": float(np.median(sgn_coef)), "逐人系数中位数：每 10 人": float(np.median(mag_coef)),
                  "|符号系数| 中位数": float(np.median(np.abs(sgn_coef))), "|大小系数| 中位数": float(np.median(np.abs(mag_coef)))}
# ---------- 4 λ 是否经由信念（两步：HRGPR 个体参数与其他共用参数固定，只重估 λ 与新加项；含 σ） ----------
f, sp, X, phi = PL.fit("HRGPR")
names = sp.names(); lam0 = phi[names.index("lam")]
z_full = ms["z"]
LAG = PL.LAG
rest = z_full - lam0 * LAG[None]                              # 去掉 λ 项
Pc = np.where(OK, P, 60.5)
belief_cat = np.where(OK, (Pc <= 60).astype(float) - 0.5, 0.0)      # 陈述的信念：不挤 +0.5 / 挤 −0.5
belief_mag = np.where(OK, (60.5 - Pc) / 10, 0.0)
def fitz(fz, k):
    nll = lambda q: float(np.sum(np.logaddexp(0, fz(q)) - A * fz(q)))
    r = minimize(nll, np.zeros(k), method="L-BFGS-B"); _, ll = L.marginal_sigma(fz(r.x), A); return r.x, ll
x0, ll0 = fitz(lambda q: rest + q[0] * LAG[None], 1)
x1, ll1 = fitz(lambda q: rest + q[0] * LAG[None] + q[1] * belief_cat, 2)
x2, ll2 = fitz(lambda q: rest + q[0] * LAG[None] + q[1] * belief_cat + q[2] * belief_mag, 3)
out["λ 是否经由信念"] = {"只有 λ：λ": float(x0[0]), "加陈述信念（挤 / 不挤）：λ": float(x1[0]), "陈述信念系数": float(x1[1]),
                     "LR（加挤 / 不挤）": float(2 * (ll1 - ll0)), "再加预测大小：λ": float(x2[0]), "大小系数（每 10 人）": float(x2[2]),
                     "LR（再加大小）": float(2 * (ll2 - ll1)),
                     "说明": "预测在选择之后，含合理化成分，所以'陈述信念'的系数会偏大；这里只看 λ 是否因之消失"}
PL.save(dict(out, 逐人符号系数=sgn_coef, 逐人大小系数=mag_coef, 逐人人内相关=r_i), "s4_信念系统的直接测量.json")
import json; print(json.dumps(out, ensure_ascii=False, indent=1))
