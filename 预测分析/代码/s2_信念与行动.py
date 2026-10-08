# -*- coding: utf-8 -*-
"""第 2 步：信念与行动
  A 一致性：若预测本轮总人数 P ≤ 60，"去"是最优反应（去且不挤 1 分 > 不去 0 分）；P ≥ 61 则"不去"最优（0.7 > 0）。
    一致率、与独立基线比较、逐人分布、心理测量曲线（去的概率对 60.5 − P 的 logistic 斜率）
  B 选择会不会改变预测（预测在选择之后）：P_it − 60 = α_i + γ·a_it + 控制 + e
    - OLS：γ 混合了"信念 → 选择"（信不挤才去）与"选择 → 预测"（合理化 / 投射 / 把自己算进去）
    - 工具变量：用不经过信念的选择变化识别 γ
        IV1 习惯痕迹 c_it（自己过去的选择史；排除限制：它只通过本轮选择影响预测）
        IV2 宏观调节的习惯推力 稳定×c、偏离×c（c 本身作控制；排除限制更弱：投射若存在，不应随稳定程度变化）
    - 控制：上一轮人数（大小与挤 / 不挤）、自己上一轮与上上轮的预测、稳定、偏离、轮次
    γ < 0：选了"去"之后把人数说少（自我合理化 / 选择支持偏差）；γ > 0：投射或"把自己算进去"（机械 +1）
输出：结果/s2_信念与行动.json
"""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
import pred_lib as PL
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
ms = PL.model_states("HRGPR"); C = ms["c"]
out = {}
# ---------- A 一致性 ----------
br = (P <= 60).astype(float)                                   # 最优反应：去
cons = np.where(OK, (A == br), np.nan)
pg = np.nanmean(A); pb = np.nanmean(br[OK])
out["一致性"] = {"一致率": float(np.nanmean(cons)), "独立基线（同样的边际比例）": float(pg * pb + (1 - pg) * (1 - pb)),
              "去的人中预测 ≤ 60 的比例": float(np.nanmean(br[(A == 1) & OK])), "不去的人中预测 ≥ 61 的比例": float(np.nanmean(1 - br[(A == 0) & OK])),
              "逐人一致率 中位数 [四分位]": [float(np.nanmedian(np.nanmean(cons, 1))), *[float(q) for q in np.nanpercentile(np.nanmean(cons, 1), [25, 75])]],
              "逐人一致率 < 0.6 的人数": int((np.nanmean(cons, 1) < 0.6).sum()),
              "去的人平均预测": float(np.nanmean(P[A == 1])), "不去的人平均预测": float(np.nanmean(P[(A == 0) & OK]))}
# 心理测量曲线：逐人 logistic a ~ k_i + s_i·(60.5 − P)/10
s_i = np.full(n, np.nan)
for i in range(n):
    x = (60.5 - P[i][OK[i]]) / 10; a = A[i][OK[i]]
    f = lambda q: float(np.sum(np.logaddexp(0, q[0] + q[1] * x) - a * (q[0] + q[1] * x)))
    s_i[i] = minimize(f, [0, 0], method="L-BFGS-B", bounds=[(-10, 10), (-20, 20)]).x[1]
out["心理测量"] = {"逐人斜率 中位数（每 10 人）": float(np.median(s_i)), "斜率 > 0 的人数": int((s_i > 0).sum()),
               "斜率 > 1 的人数（预测差 10 人，去的 logit 差 > 1）": int((s_i > 1).sum())}
# 按预测分组的去的比例
bins = [(0, 50), (50, 55), (55, 58), (58, 60), (60, 61), (61, 63), (63, 66), (66, 71), (71, 101)]
out["按预测分组"] = [dict(预测=f"{lo}–{hi - 1}", 观测数=int(((P >= lo) & (P < hi)).sum()), 去的比例=float(A[(P >= lo) & (P < hi)].mean())) for lo, hi in bins]
# ---------- B 选择 → 预测 ----------
t0 = 2
idx = [(i, t) for i in range(n) for t in range(t0, T) if OK[i, t] and OK[i, t - 1] and OK[i, t - 2]]
ii = np.array([x[0] for x in idx]); tt = np.array([x[1] for x in idx])
L = PL.L
STD = L.load_std()
z = lambda m, v: (v - STD[m][0]) / STD[m][1]
stab = z("stab", PL.PM["stab"]); dev = z("dev", PL.PM["dev"]); tim = z("time", PL.PM["time"])
y = P[ii, tt] - 60; a = A[ii, tt]; c = C[ii, tt]
dprev = (N[tt - 1] - 60) / 10; sg = np.sign(N[tt - 1] - 60.5)
Pp1 = (P[ii, tt - 1] - 60) / 10; Pp2 = (P[ii, tt - 2] - 60) / 10
Xc = np.column_stack([dprev, sg, Pp1, Pp2, stab[tt], dev[tt], tim[tt]])
names = ["上轮偏离/10", "上轮挤(±1)", "自己上轮预测/10", "自己上上轮预测/10", "稳定", "偏离", "轮次"]
out["选择→预测"] = {}
out["选择→预测"]["OLS 无控制"] = PL.ols_fe(y, a[:, None], ii, ["去"])["去"]
out["选择→预测"]["OLS 有控制"] = PL.ols_fe(y, np.column_stack([a, Xc]), ii, ["去"] + names)
r1 = PL.iv_fe(y, a, Xc, c[:, None], ii, names)
r2 = PL.iv_fe(y, a, np.column_stack([Xc, c]), np.column_stack([stab[tt] * c, dev[tt] * c]), ii, names + ["习惯痕迹c"])
out["选择→预测"]["IV1 习惯痕迹"] = r1
out["选择→预测"]["IV2 稳定×c、偏离×c（c 作控制）"] = r2
# 机械 +1 的参照：若预测含自己且无任何偏差，去的人应比同信念的不去的人多报 1 人
PL.save(dict(out, 逐人心理测量斜率=s_i), "s2_信念与行动.json")
import json; print(json.dumps(out, ensure_ascii=False, indent=1))
