# -*- coding: utf-8 -*-
"""第 1 步：信念怎样形成（描述与回归）
  1 准确性与偏差：个人 / 平均预测的 MAE，与几种基准比较（上一轮人数、常数 60、样本均值、真实 AR(1) 最优预测）
  2 预测的"省力"迹象：恰为 60、5 的倍数、与自己上一轮预测相同、与上一轮人数相同
  3 锚定与外推 / 反转：P − 60 对 N_{t−1} − 60 的斜率（全体、逐人、分区块）；符号（挤 / 不挤）与大小（偏离多少）分开
  4 群体：平均预测与本轮人数、上一轮人数的关系；预测分歧（人间 SD）
输出：结果/s1_信念形成.json
"""
import numpy as np
from scipy import stats
import pred_lib as PL
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
NP = PL.NPREV
out = {}
sl = slice(1, None)                                           # 第 1 轮没有上一轮
# ---------- 1 准确性 ----------
e_ind = np.abs(P - N[None]); mae_ind_person = np.nanmean(e_ind[:, sl], 1)
Pm = np.nanmean(P, 0)
phi = np.polyfit(N[:-1] - N.mean(), N[1:] - N.mean(), 1)[0]
bench = {"上一轮人数": NP, "常数 60": np.full(T, 60.0), "样本均值 60.4": np.full(T, N.mean()),
         f"AR(1) 最优（φ = {phi:.3f}，事后）": N.mean() + phi * (NP - N.mean())}
out["准确性"] = {"个人预测 MAE（逐人均值的中位数）": float(np.median(mae_ind_person)),
              "个人预测 MAE（全部）": float(np.nanmean(e_ind[:, sl])),
              "平均预测 MAE": float(np.mean(np.abs(Pm - N)[sl])),
              **{f"{k} MAE": float(np.nanmean(np.abs(v - N)[sl])) for k, v in bench.items()},
              "比'常数 60'更准的人数": int((mae_ind_person < np.mean(np.abs(60 - N[sl]))).sum()),
              "比'上一轮人数'更准的人数": int((mae_ind_person < np.nanmean(np.abs(NP - N)[sl])).sum())}
out["偏差"] = {"平均预测 − 实际": float(np.nanmean(P - N[None])), "平均预测": float(np.nanmean(P)), "实际均值": float(N.mean()),
             "预测的人间 SD（每轮，均值）": float(np.nanmean(np.nanstd(P, 0))), "平均预测的轮间 SD": float(Pm.std()), "实际人数 SD": float(N.std())}
# ---------- 2 省力迹象 ----------
prevP = np.c_[np.full((n, 1), np.nan), P[:, :-1]]
out["省力迹象"] = {"恰为 60": float(np.nanmean(P == 60)), "5 的倍数": float(np.nanmean(P % 5 == 0)),
               "与自己上一轮预测相同": float(np.nanmean((P == prevP)[:, sl])), "与上一轮人数相同": float(np.nanmean((P == NP[None])[:, sl])),
               "每人最常用一个值的占比（中位数）": float(np.median([np.max(np.unique(P[i][OK[i]], return_counts=True)[1]) / OK[i].sum() for i in range(n)])),
               "每人用过的不同数值个数（中位数）": float(np.median([len(np.unique(P[i][OK[i]])) for i in range(n)]))}
# ---------- 3 锚定与外推 ----------
d = (NP - 60)[sl]; y = (P - 60)[:, sl]
Y = y.ravel(); D = np.tile(d, n); ok = ~np.isnan(Y)
b = np.polyfit(D[ok], Y[ok], 1)
sg = np.sign(D + 0.5)                                         # 上一轮挤（N ≥ 61）为 +1
Xm = np.column_stack([np.ones(ok.sum()), sg[ok], D[ok] / 10])
bb, *_ = np.linalg.lstsq(Xm, Y[ok], rcond=None)
w_i = np.array([np.polyfit(d[OK[i, 1:]], y[i][OK[i, 1:]], 1)[0] for i in range(n)])
se_i = []
for i in range(n):
    r = stats.linregress(d[OK[i, 1:]], y[i][OK[i, 1:]]); se_i.append(r.stderr)
se_i = np.array(se_i)
blocks = []
for k in range(4):
    s = slice(k * 100, (k + 1) * 100)
    dk = (NP - 60)[s]; yk = (P - 60)[:, s]; m = ~np.isnan(dk)
    Yk = yk[:, m].ravel(); Dk = np.tile(dk[m], n); okk = ~np.isnan(Yk)
    blocks.append(dict(区块=f"{k*100+1}–{(k+1)*100}", 斜率=float(np.polyfit(Dk[okk], Yk[okk], 1)[0]),
                       个人MAE=float(np.nanmean(np.abs(P[:, s] - N[s][None]))), 平均预测MAE=float(np.mean(np.abs(Pm[s] - N[s])))))
out["锚定与外推"] = {"全体斜率 w（P−60 对 N_{t−1}−60）": float(b[0]), "截距": float(b[1]),
                 "符号与大小分开：截距、上轮挤（±1）、每 10 人": [float(x) for x in bb],
                 "逐人 w 中位数": float(np.median(w_i)), "w > 0（外推）人数": int((w_i > 0).sum()), "w < 0（预期反转）人数": int((w_i < 0).sum()),
                 "逐人显著为正": int((w_i / se_i > 1.96).sum()), "逐人显著为负": int((w_i / se_i < -1.96).sum()),
                 "真实人数的斜率（N_t−60 对 N_{t−1}−60）": float(np.polyfit(N[:-1] - 60, N[1:] - 60, 1)[0]),
                 "分区块": blocks}
# ---------- 4 群体 ----------
r1 = stats.pearsonr(Pm[sl], N[sl]); r0 = stats.pearsonr(Pm[sl], NP[sl])
Xg = np.column_stack([np.ones(T - 1), (Pm - 60)[sl], (NP - 60)[sl]])
bg, *_ = np.linalg.lstsq(Xg, (N - 60)[sl], rcond=None)
eg = (N - 60)[sl] - Xg @ bg; cov = np.linalg.inv(Xg.T @ Xg) * eg.var(ddof=3)
disp = np.nanstd(P, 0)
out["群体"] = {"corr(平均预测_t, 本轮人数_t)": [float(r1[0]), float(r1[1])], "corr(平均预测_t, 上一轮人数)": [float(r0[0]), float(r0[1])],
             "本轮人数−60 = a + b1·(平均预测−60) + b2·(上一轮−60)": dict(b1=float(bg[1]), b1_t=float(bg[1] / np.sqrt(cov[1, 1])), b2=float(bg[2]), b2_t=float(bg[2] / np.sqrt(cov[2, 2]))),
             "corr(预测分歧_t, |本轮人数−60|)": [float(x) for x in stats.pearsonr(disp[sl], np.abs(N - 60)[sl])],
             "corr(平均预测_t, 上一轮平均预测)": float(np.corrcoef(Pm[1:], Pm[:-1])[0, 1])}
PL.save(dict(out, 逐人w=w_i, 逐人MAE=mae_ind_person), "s1_信念形成.json")
import json; print(json.dumps(out, ensure_ascii=False, indent=1))
