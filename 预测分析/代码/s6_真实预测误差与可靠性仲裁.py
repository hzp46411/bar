# -*- coding: utf-8 -*-
"""第 6 步：用真实的预测误差检验可靠性仲裁（Lee, Shimojo & O'Doherty, 2014 的"状态预测误差"）
  信念系统的可靠性（只用 t 之前的信息）：
    R1 判断对错：自己的预测（挤 / 不挤）与实际是否一致，指数加权（速率 0.2）
    R2 绝对误差：|P − N| 的指数加权，取负号（越大越可靠）
  习惯系统的可靠性：若照"重复上一轮"做会不会赢，指数加权（速率 0.2）——与第 4 层的 relH 同义，但只用数据
  问题：两个系统冲突时（自己的预测建议 ≠ 上一轮的选择），选择听从自己预测的概率，是否随信念可靠性上升、随习惯可靠性下降？
  对照：第 4 层模型推出的 relB（信念建议成绩）。
输出：结果/s6_真实预测误差与可靠性仲裁.json
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import pred_lib as PL
L = PL.L; STD = L.load_std()
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
G, S = L.G_REAL, L.S_REAL
ms = PL.model_states("HRGPR"); C = ms["c"]
rate = 0.2
Pf = np.where(OK, P, 60.5)
R1 = np.full((n, T), np.nan); R2 = np.full((n, T), np.nan); RH = np.full((n, T), np.nan)
r1 = np.full(n, .5); r2 = np.full(n, -5.0); rh = np.full(n, .5)
prevA = np.c_[np.full((n, 1), np.nan), A[:, :-1]]
for t in range(T):
    R1[:, t], R2[:, t], RH[:, t] = r1, r2, rh
    hit = ((Pf[:, t] <= 60) == (N[t] <= 60)).astype(float)
    r1 = np.where(OK[:, t], (1 - rate) * r1 + rate * hit, r1)
    r2 = np.where(OK[:, t], (1 - rate) * r2 + rate * (-np.abs(Pf[:, t] - N[t])), r2)
    if t > 0:
        habit_win = prevA[:, t] * G[:, t] + (1 - prevA[:, t]) * S[:, t]    # 若重复上一轮会不会赢
        rh = (1 - rate) * rh + rate * habit_win
zsc = lambda M, m: (M - np.nanmean(M[m])) / np.nanstd(M[m])
br = (P <= 60).astype(float)
ok = OK & ~np.isnan(prevA); ok[:, :5] = False
conflict = ok & (br != prevA)
follow_belief = (A == br)
stab = np.broadcast_to((PL.PM["stab"] - STD["stab"][0]) / STD["stab"][1], (n, T))
dev = np.broadcast_to((PL.PM["dev"] - STD["dev"][0]) / STD["dev"][1], (n, T))
tim = np.broadcast_to((PL.PM["time"] - STD["time"][0]) / STD["time"][1], (n, T))
relB_model = (ms["sB"] - STD["relB"][0]) / STD["relB"][1]
def logit_fe(y, Xs, mask, names):
    ii = np.nonzero(mask); Y = y[mask].astype(float); X = np.column_stack([x[mask] for x in Xs]); g = ii[0]; Gn = n; K = X.shape[1]
    def nll(q):
        eta = q[:Gn][g] + X @ q[Gn:]; return float(np.sum(np.logaddexp(0, eta) - Y * eta))
    def grad(q):
        eta = q[:Gn][g] + X @ q[Gn:]; r = 1 / (1 + np.exp(-eta)) - Y
        return np.r_[np.bincount(g, weights=r, minlength=Gn), X.T @ r]
    res = minimize(nll, np.zeros(Gn + K), jac=grad, method="L-BFGS-B"); q = res.x
    eta = q[:Gn][g] + X @ q[Gn:]; p = 1 / (1 + np.exp(-eta)); w = p * (1 - p); Xd = X.copy()
    for k in range(Gn):
        m = g == k
        if m.any(): Xd[m] -= np.average(X[m], axis=0, weights=w[m] + 1e-12)
    Hi = np.linalg.inv((Xd * w[:, None]).T @ Xd); meat = np.zeros((K, K))
    for k in range(Gn):
        m = g == k; s_ = Xd[m].T @ (Y[m] - p[m]); meat += np.outer(s_, s_)
    se = np.sqrt(np.diag(Hi @ meat @ Hi)); b = q[Gn:]
    return {nm: dict(b=float(b[j]), se=float(se[j]), z=float(b[j] / se[j]), p=float(2 * norm.sf(abs(b[j] / se[j])))) for j, nm in enumerate(names)}, float(res.fun)
ctrl = [stab, dev, tim]; cn = ["稳定", "偏离", "轮次"]
out = {}
out["描述"] = {"冲突情境观测数": int(conflict.sum()), "冲突时听从自己预测的比例": float(follow_belief[conflict].mean()),
             "R1（判断对错，指数加权）均值": float(np.nanmean(R1[ok])), "corr(R1, 模型 relB)": float(np.corrcoef(R1[ok], ms['sB'][ok])[0, 1]),
             "corr(习惯可靠性, 模型 relH)": float(np.corrcoef(RH[ok], ms['sH'][ok])[0, 1])}
res = {}
res["真实预测误差：判断对错 R1 + 习惯可靠性"], f1 = logit_fe(follow_belief, [zsc(R1, ok), zsc(RH, ok)] + ctrl, conflict, ["信念可靠性 R1", "习惯可靠性"] + cn)
res["真实预测误差：绝对误差 R2 + 习惯可靠性"], f2 = logit_fe(follow_belief, [zsc(R2, ok), zsc(RH, ok)] + ctrl, conflict, ["信念可靠性 R2（−|误差|）", "习惯可靠性"] + cn)
res["模型推出的 relB + 习惯可靠性"], f3 = logit_fe(follow_belief, [relB_model, zsc(RH, ok)] + ctrl, conflict, ["模型 relB", "习惯可靠性"] + cn)
res["三者一起"], f4 = logit_fe(follow_belief, [zsc(R1, ok), relB_model, zsc(RH, ok)] + ctrl, conflict, ["信念可靠性 R1", "模型 relB", "习惯可靠性"] + cn)
out["冲突时听从自己预测的概率（logistic，个人固定效应，按人聚类）"] = res
out["负对数似然（同一样本）"] = {"R1": f1, "R2": f2, "模型 relB": f3, "三者": f4}
# 人内 / 人间：R1 的人内成分
R1w = R1 - np.nanmean(np.where(ok, R1, np.nan), 1, keepdims=True)
out["人内成分"], _ = logit_fe(follow_belief, [zsc(R1w, ok), zsc(RH, ok)] + ctrl, conflict, ["信念可靠性 R1（人内）", "习惯可靠性"] + cn)
PL.save(out, "s6_真实预测误差与可靠性仲裁.json")
import json; print(json.dumps(out, ensure_ascii=False, indent=1))
