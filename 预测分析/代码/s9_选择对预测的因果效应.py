# -*- coding: utf-8 -*-
"""第 9 步：选择对预测的因果效应（γ）——用"个人的习惯推力"作工具变量
  问题：去的人比不去的人少报 4.4 人。这里面有多少是"信不挤才去"（信念 → 选择），多少是"去了之后把人数说少"（选择 → 预测）？
  联合结构模型（s7）里，类别写法的 γ 被"整数预测 + 60/61 台阶"卡在 ±0.5 以内，不能直接读成合理化的大小，所以这里用不依赖函数形式的识别。
  工具：Z_it = κ̂_i · c_it —— 习惯痕迹 c 乘以"这个人对习惯的依赖程度" κ̂_i。
        κ̂_i 用交叉拟合（前 200 轮估计、用于后 200 轮，反之亦然），避免同一观测既用来估计 κ̂ 又用来识别 γ。
        逐人 logit：a_t ~ 1 + c_t + 上轮挤(±1) + 上轮偏离/10
  控制：个人固定效应、c 本身、上轮挤、上轮偏离、自己上一轮与上上轮的预测、"信念痕迹"（过去预测的指数平均，速率同 α_H）、稳定、偏离、轮次
  排除限制：在控制了过去的预测之后，过去的选择史只通过本轮选择影响本轮预测。
  结果变量：P − 60（人数）与 1[P ≤ 60]（报告"不挤"）。
  参照：只是把自己算进去 → γ = +1、对 1[P ≤ 60] 的效应约 −0.1；合理化 → γ < 0、对"不挤"的效应 > 0。
  稳健性：工具换成主模型 HRGPR 的习惯部分（全样本估计）；κ̂ 改用奇偶轮交叉拟合；不控制信念痕迹。
输出：结果/s9_选择对预测的因果效应.json
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import pred_lib as PL
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
L = PL.L; STD = L.load_std()
ms = PL.model_states("HRGPR"); C = ms["c"]; HAB = ms["hab"]
f4 = PL.fit("HRGPR"); aH = float(expit(f4[0]["shared"]["logit_aH"]))
NP = PL.NPREV; SGN = np.sign(NP - 60.5); DMV = (NP - 60) / 10
zs = lambda m, v: (v - STD[m][0]) / STD[m][1]
stab = zs("stab", PL.PM["stab"]); dev = zs("dev", PL.PM["dev"]); tim = zs("time", PL.PM["time"])
# 信念痕迹：过去预测的指数平均（缺失时保持）
BT = np.zeros((n, T)); h = np.full(n, 60.0)
for t in range(T):
    BT[:, t] = h; pt = np.where(OK[:, t], P[:, t], h); h = h + aH * (pt - h)
BT = (BT - 60) / 10


def kappa_hat(train):
    """逐人 logit，返回 κ̂_i（训练轮次 train 为布尔向量）。"""
    k = np.zeros(n)
    for i in range(n):
        m = train & OK[i] & ~np.isnan(SGN)
        X = np.column_stack([np.ones(m.sum()), C[i, m], SGN[m], DMV[m]]); a = A[i, m]
        f = lambda q: float(np.sum(np.logaddexp(0, X @ q) - a * (X @ q))) + 0.01 * q @ q
        k[i] = minimize(f, np.zeros(4), method="L-BFGS-B", bounds=[(-10, 10)] * 4).x[1]
    return k


tix = np.arange(T)
half = tix < T // 2
K_half = np.zeros((n, T)); kA = kappa_hat(half); kB = kappa_hat(~half)
K_half[:, ~half] = kA[:, None]; K_half[:, half] = kB[:, None]
odd = tix % 2 == 1
K_par = np.zeros((n, T)); kO = kappa_hat(odd); kE = kappa_hat(~odd)
K_par[:, ~odd] = kO[:, None]; K_par[:, odd] = kE[:, None]

t0 = 2
idx = [(i, t) for i in range(n) for t in range(t0, T) if OK[i, t] and OK[i, t - 1] and OK[i, t - 2]]
ii = np.array([x[0] for x in idx]); tt = np.array([x[1] for x in idx])
yN = P[ii, tt] - 60.0; yK = (P[ii, tt] <= 60).astype(float); a = A[ii, tt]; c = C[ii, tt]
Xc = np.column_stack([c, SGN[tt], DMV[tt], (P[ii, tt - 1] - 60) / 10, (P[ii, tt - 2] - 60) / 10, BT[ii, tt], stab[tt], dev[tt], tim[tt]])
nm = ["习惯痕迹c", "上轮挤(±1)", "上轮偏离/10", "自己上轮预测/10", "自己上上轮预测/10", "信念痕迹/10", "稳定", "偏离", "轮次"]
out = {"说明": "γ = 本轮选择'去'对本轮预测的因果效应（人数）；只是把自己算进去 γ = +1；合理化 γ < 0",
       "κ̂ 前后半相关": float(np.corrcoef(kA, kB)[0, 1]), "κ̂ 奇偶相关": float(np.corrcoef(kO, kE)[0, 1]),
       "κ̂ > 0 的人数（前半 / 后半）": [int((kA > 0).sum()), int((kB > 0).sum())], "观测数": int(len(ii))}
out["OLS（参照）"] = {"人数": PL.ols_fe(yN, np.column_stack([a, Xc]), ii, ["去"] + nm)["去"],
                   "不挤": PL.ols_fe(yK, np.column_stack([a, Xc]), ii, ["去"] + nm)["去"]}
specs = {
    "主：κ̂（前后半交叉拟合）× c": (K_half[ii, tt] * c, Xc, nm),
    "κ̂（奇偶交叉拟合）× c": (K_par[ii, tt] * c, Xc, nm),
    "主模型 HRGPR 的习惯部分（全样本）": (HAB[ii, tt], Xc, nm),
    "主，不控制信念痕迹": (K_half[ii, tt] * c, np.delete(Xc, 5, 1), [x for x in nm if x != "信念痕迹/10"]),
}
for lab, (z, Xe, nme) in specs.items():
    r = {}
    for yl, y in (("人数", yN), ("不挤", yK)):
        q = PL.iv_fe(y, a, Xe, z[:, None], ii, nme)
        r[yl] = dict(γ=q["内生变量"], 第一阶段F=q["第一阶段F"])
    rf = PL.ols_fe(yN, np.column_stack([z, Xe]), ii, ["工具"] + nme)["工具"]
    fs = PL.ols_fe(a, np.column_stack([z, Xe]), ii, ["工具"] + nme)["工具"]
    r["简化式（工具 → 预测）"] = rf; r["第一阶段（工具 → 去）"] = fs
    out[lab] = r
    print(lab, {k: (round(v["γ"]["b"], 3), round(v["γ"]["se"], 3), round(v["第一阶段F"], 1)) for k, v in r.items() if "γ" in v}, flush=True)
# 与 +1、0 的检验（主规格）
g = out["主：κ̂（前后半交叉拟合）× c"]["人数"]["γ"]
out["主规格检验"] = {"γ = +1 的 z": (g["b"] - 1) / g["se"], "γ = 0 的 z": g["b"] / g["se"]}
# 按"信念 / 惯性主导"分组（主模型的信念份额中位数切分）
share = np.nanmean(np.abs(ms["bel"]), 1) / (np.nanmean(np.abs(ms["bel"]), 1) + np.nanmean(np.abs(ms["hab"]), 1))
grp = {}
for lab, sel in (("惯性主导（份额低于中位数）", share[ii] < np.median(share)), ("信念主导", share[ii] >= np.median(share))):
    q = PL.iv_fe(yN[sel], a[sel], Xc[sel], (K_half[ii, tt] * c)[sel][:, None], ii[sel], nm)
    grp[lab] = dict(γ=q["内生变量"], 第一阶段F=q["第一阶段F"])
out["分组"] = grp; print(grp, flush=True)
PL.save(out, "s9_选择对预测的因果效应.json")
print(out["主规格检验"])
