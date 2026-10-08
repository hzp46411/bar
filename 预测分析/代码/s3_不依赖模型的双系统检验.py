# -*- coding: utf-8 -*-
"""第 3 步：不依赖模型的双系统检验——用被试自己陈述的信念，而不是模型推出的信念
  "信念一致的选择"：预测 ≤ 60 而去，或预测 ≥ 61 而不去。
  "违背信念"的选择分两种：朝习惯方向（与上一轮选择相同）或朝相反方向（换选）。
  双系统的预测：
    1 违背信念的选择应多数倒向习惯（重复上一轮），而不是随机
    2 局面越稳定，"违背信念、跟随习惯"越多（稳定 → 推向习惯）
    3 上一轮偏离越大，"违背信念、跟随习惯"越少（意外打断习惯）
    4 越往后（经验）越少（信念加强）
    5 自己的习惯越强（习惯痕迹 |2H−1| 越大），越多
  检验：逐观测的 logistic 回归（个人固定效应以个人截距实现），因变量 = "违背信念且跟随习惯"；也对"违背信念"本身做同样的回归。
输出：结果/s3_不依赖模型的双系统检验.json
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, chi2
import pred_lib as PL
L = PL.L; STD = L.load_std()
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
ms = PL.model_states("HRGPR"); C = ms["c"]
z = lambda m, v: (v - STD[m][0]) / STD[m][1]
prevA = np.c_[np.full((n, 1), np.nan), A[:, :-1]]
br = (P <= 60).astype(float)
ok = OK & ~np.isnan(prevA); ok[:, :2] = False
cons = (A == br); viol = ok & ~cons
rep = (A == prevA)
habit_follow = viol & rep                    # 违背信念、跟随习惯（重复上一轮）
habit_against = viol & ~rep                  # 违背信念、换选
out = {}
out["描述"] = {"观测数": int(ok.sum()), "违背信念的比例": float(viol[ok].mean()),
             "违背信念中重复上一轮的比例": float(rep[viol].mean()), "信念一致中重复上一轮的比例": float(rep[ok & cons].mean()),
             "全部选择中重复上一轮的比例": float(rep[ok].mean())}
# 若违背信念的选择与习惯无关，它们重复上一轮的概率应与"同一信念下的选择"无关；更直接的对照：
# 信念建议与上一轮选择相同（信念与习惯一致）vs 不同（冲突）时的违背率
agree = (br == prevA)
out["信念与习惯一致 / 冲突"] = {"一致时：违背信念的比例": float(viol[ok & agree].mean()), "冲突时：违背信念的比例（即跟随习惯）": float(viol[ok & ~agree].mean()),
                          "一致时观测数": int((ok & agree).sum()), "冲突时观测数": int((ok & ~agree).sum())}
# ---------- 回归：冲突情境（信念建议 ≠ 上一轮选择）中，选择跟随习惯而不是信念的概率 ----------
stab = np.broadcast_to(z("stab", PL.PM["stab"]), (n, T)); dev = np.broadcast_to(z("dev", PL.PM["dev"]), (n, T)); tim = np.broadcast_to(z("time", PL.PM["time"]), (n, T))
hs = np.abs(C)                                # 习惯强度（模型的习惯痕迹）
hsz = (hs - hs[ok].mean()) / hs[ok].std()
conf = (np.abs(P - 60.5) / 10)                # 信念的坚定程度：预测离 60.5 多远（每 10 人）
def logit_fe(y, Xs, mask, names):
    ii = np.nonzero(mask)
    Y = y[mask].astype(float); X = np.column_stack([x[mask] for x in Xs]); g = ii[0]
    G = n; K = X.shape[1]
    def nll(q):
        a = q[:G][g]; eta = a + X @ q[G:]
        return float(np.sum(np.logaddexp(0, eta) - Y * eta))
    def grad(q):
        a = q[:G][g]; eta = a + X @ q[G:]; r = 1 / (1 + np.exp(-eta)) - Y
        ga = np.bincount(g, weights=r, minlength=G); return np.r_[ga, X.T @ r]
    res = minimize(nll, np.zeros(G + K), jac=grad, method="L-BFGS-B")
    q = res.x; a = q[:G][g]; eta = a + X @ q[G:]; p = 1 / (1 + np.exp(-eta)); w = p * (1 - p)
    # 聚类稳健标准误（对斜率；把个人截距 partial out 的近似：用个人内去均值后的加权信息矩阵）
    Xd = X.copy()
    for k in range(G):
        m = g == k
        if m.sum() > 0:
            Xd[m] -= np.average(X[m], axis=0, weights=w[m] + 1e-12)
    H = (Xd * w[:, None]).T @ Xd; Hi = np.linalg.inv(H)
    meat = np.zeros((K, K))
    for k in range(G):
        m = g == k; s = Xd[m].T @ (Y[m] - p[m]); meat += np.outer(s, s)
    V = Hi @ meat @ Hi; se = np.sqrt(np.diag(V))
    b = q[G:]
    return {nm: dict(b=float(b[j]), se=float(se[j]), z=float(b[j] / se[j]), p=float(2 * norm.sf(abs(b[j] / se[j])))) for j, nm in enumerate(names)}, float(res.fun)
names = ["稳定", "偏离", "轮次", "习惯强度", "信念坚定程度"]
Xs = [stab, dev, tim, hsz, conf]
mask_conf = ok & ~agree                       # 冲突情境
r_conf, _ = logit_fe(viol, Xs, mask_conf, names)
out["冲突情境中跟随习惯（而不是自己的信念）的概率"] = r_conf
r_all, _ = logit_fe(habit_follow, Xs, ok, names)
out["全部观测：违背信念且跟随习惯"] = r_all
r_v, _ = logit_fe(viol & agree, Xs, ok & agree, names)
out["信念与习惯一致时仍违背信念（换选）——对照"] = r_v
# 按稳定程度、偏离分组的描述
rows = []
for k in range(5):
    m = mask_conf & (np.broadcast_to(PL.PM["stab"], (n, T)) == k)
    rows.append(dict(稳定=k, 冲突观测数=int(m.sum()), 跟随习惯的比例=float(viol[m].mean()) if m.sum() else None))
out["按稳定分组（冲突情境）"] = rows
devraw = np.broadcast_to(PL.PM["dev"] * 10, (n, T)); rows = []
for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 99)):
    m = mask_conf & (devraw >= lo) & (devraw < hi)
    rows.append(dict(上轮偏离=f"{lo}–{hi-1 if hi < 99 else '+'} 人", 冲突观测数=int(m.sum()), 跟随习惯的比例=float(viol[m].mean())))
out["按偏离分组（冲突情境）"] = rows
rows = []
for k in range(4):
    m = mask_conf.copy(); m[:, :k * 100] = False; m[:, (k + 1) * 100:] = False
    rows.append(dict(区块=f"{k*100+1}–{(k+1)*100}", 跟随习惯的比例=float(viol[m].mean())))
out["按轮次分组（冲突情境）"] = rows
PL.save(out, "s3_不依赖模型的双系统检验.json")
import json; print(json.dumps(out, ensure_ascii=False, indent=1))
