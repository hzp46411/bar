# -*- coding: utf-8 -*-
"""第 13 步：ACF4 的缺口——预测与选择是否受 4 轮前人数的影响
  背景：真实人数的 ACF4 = −0.155，第 4 层所有模型都复现不了（p ≤ .008）。界面只显示上一轮人数，
        所以若信念或选择对 N(t−4) 有反应，只能来自记忆（或某种周期性的个人策略）。
  1 群体：人数的 ACF 与 PACF（1–8）；N_t 对 N_{t−1…t−6} 的回归
  2 预测：P_it − 60 对 (N_{t−k} − 60)/10（k = 1…6）的回归（个人固定效应、按人聚类）；全体 / 如实报告者 / 控制本轮选择
  3 选择：去（线性概率，个人固定效应）对同样的滞后；全体 / 控制习惯痕迹
  4 逐人：对 N(t−4) 显著的人数（预测、选择）；两者是否是同一批人
输出：结果/s13_四轮前与ACF4.json
"""
import json
import numpy as np
from scipy import stats
from scipy.special import expit
import pred_lib as PL
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
ms = PL.model_states("HRGPR"); C = ms["c"]
d10 = json.loads((PL.OUT / "s10_合理化联合模型_ρ逐人.json").read_text(encoding="utf-8"))
rho_f = expit(np.array(d10["X"])[:, 7]); honest = rho_f < 0.05
out = {}
x = N - N.mean()
acf = [float(np.corrcoef(x[:-k], x[k:])[0, 1]) for k in range(1, 9)]
pacf = []
for k in range(1, 9):
    Y = x[k:]; Xl = np.column_stack([x[k - j:len(x) - j] for j in range(1, k + 1)])
    b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(Y)), Xl]), Y, rcond=None); pacf.append(float(b[-1]))
K = 6
Y = x[K:]; Xl = np.column_stack([np.ones(len(Y))] + [x[K - j:len(x) - j] for j in range(1, K + 1)])
b, *_ = np.linalg.lstsq(Xl, Y, rcond=None); e = Y - Xl @ b; s2 = e @ e / (len(Y) - Xl.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(Xl.T @ Xl)))
out["群体"] = {"ACF 1–8": acf, "PACF 1–8": pacf, "N_t 对 N_{t−1…t−6}": {f"滞后{j}": dict(b=float(b[j]), t=float(b[j] / se[j])) for j in range(1, K + 1)}}
print(out["群体"], flush=True)
# 个体回归的样本
t0 = K
idx = [(i, t) for i in range(n) for t in range(t0, T) if OK[i, t]]
ii = np.array([q[0] for q in idx]); tt = np.array([q[1] for q in idx])
L = np.column_stack([(N[tt - j] - 60) / 10 for j in range(1, K + 1)])
SG1 = np.sign(N[tt - 1] - 60.5)
nm = [f"N(t−{j})" for j in range(1, K + 1)]
y = P[ii, tt] - 60.0; a = A[ii, tt]; c = C[ii, tt]
res = {}
h = honest[ii]
res["预测：全体"] = PL.ols_fe(y, np.column_stack([L, SG1]), ii, nm + ["上轮挤±1"])
res["预测：全体，控制本轮选择"] = PL.ols_fe(y, np.column_stack([L, SG1, a]), ii, nm + ["上轮挤±1", "本轮去"])
res["预测：如实报告者"] = PL.ols_fe(y[h], np.column_stack([L, SG1])[h], ii[h], nm + ["上轮挤±1"])
res["选择：全体"] = PL.ols_fe(a, np.column_stack([L, SG1]), ii, nm + ["上轮挤±1"])
res["选择：控制习惯痕迹"] = PL.ols_fe(a, np.column_stack([L, SG1, c]), ii, nm + ["上轮挤±1", "习惯痕迹c"])
# 主模型漏掉了什么：选择减去 HRGPR 的预测概率（残差）对滞后的回归
pz = ms["p"][ii, tt]
res["选择残差（a − HRGPR 概率）"] = PL.ols_fe(a - pz, np.column_stack([L, SG1]), ii, nm + ["上轮挤±1"])
# 预测中"选择之外"的部分：只用如实报告者、并控制本轮选择
res["预测：如实报告者，控制本轮选择"] = PL.ols_fe(y[h], np.column_stack([L, SG1, a])[h], ii[h], nm + ["上轮挤±1", "本轮去"])
for k_, v in res.items():
    print(k_, {kk: (round(vv["b"], 3), round(vv["t"], 2)) for kk, vv in v.items()}, flush=True)
out["个体回归"] = res
# 逐人：对 N(t−4) 的系数
bp = np.zeros(n); tp = np.zeros(n); ba = np.zeros(n); ta = np.zeros(n)
for i in range(n):
    m = ii == i
    for yy, bb, tv in ((y, bp, tp), (a, ba, ta)):
        X = np.column_stack([np.ones(m.sum()), L[m], SG1[m]]); q, *_ = np.linalg.lstsq(X, yy[m], rcond=None)
        e = yy[m] - X @ q; s2 = e @ e / (m.sum() - X.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        bb[i] = q[4]; tv[i] = q[4] / se[4]
out["逐人 N(t−4)"] = {"预测：显著为正 / 为负": [int((tp > 1.96).sum()), int((tp < -1.96).sum())],
                    "选择：显著为正 / 为负": [int((ta > 1.96).sum()), int((ta < -1.96).sum())],
                    "预测系数与选择系数的 Spearman": [float(v) for v in stats.spearmanr(bp, ba)],
                    "说明": "随机情况下 5% 显著 ≈ 每个方向 2.5 人"}
print(out["逐人 N(t−4)"])
PL.save(dict(out, 逐人预测系数=bp, 逐人选择系数=ba), "s13_四轮前与ACF4.json")
