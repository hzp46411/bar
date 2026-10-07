# -*- coding: utf-8 -*-
"""习惯累积与宏观状态的关系（HRG 拟合）：
(1) 真实数据：各宏观状态下的习惯强度 |2H − 1|，以及"习惯方向与信念建议是否一致"
(2) 反事实开环模拟：同一批人（HRG 参数）放进不同的宏观环境，比较能累积多少习惯
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

fit = json.loads((L.OUT / "拟合" / "HRG.json").read_text(encoding="utf-8"))
sp = L.Spec(**fit["spec"]); X = np.array(fit["X"]); phi = np.array([fit["shared"][k] for k in sp.names()])
aH = float(expit(fit["shared"]["logit_aH"])); kap = X[:, 2]; beta = X[:, 1]; rho = expit(X[:, 0])
strong = np.abs(kap) >= np.median(np.abs(kap))                       # 惯性权重较大的一半人


def habit_trace(A):
    H = np.full(A.shape[0], .5); out = np.zeros(A.shape)
    for t in range(A.shape[1]):
        out[:, t] = 2 * H - 1                                         # 第 t 轮决策前的习惯（+1 = 习惯去，−1 = 习惯不去）
        H += aH * (A[:, t] - H)
    return out


def belief_rec(A, G, S):
    BL = np.full(A.shape[0], 1 / 3); BH = np.full(A.shape[0], 1 / 3); out = np.zeros(A.shape)
    for t in range(A.shape[1]):
        out[:, t] = np.sign(beta * (BL - 0.7 * BH))                   # 信念建议：+1 去，−1 不去
        BL += rho * (G[:, t] - BL); BH += rho * (S[:, t] - BH)
    return out


res = {"习惯痕迹速率": aH}
C = habit_trace(L.A_REAL); R = belief_rec(L.A_REAL, L.G_REAL, L.S_REAL)
pm, lag = L.public_mods(L.ATT)
rows = []
for lo, hi, lab in ((0, 1, "刚翻转"), (1, 2, "持续 1 轮"), (2, 3, "持续 2 轮"), (3, 9, "持续 ≥3 轮")):
    m = (pm["stab"] >= lo) & (pm["stab"] < hi); m[:3] = False
    rows.append(dict(状态=lab, 轮数=int(m.sum()), 习惯强度=round(float(np.abs(C[:, m]).mean()), 3),
                     惯性较强者习惯强度=round(float(np.abs(C[strong][:, m]).mean()), 3),
                     习惯与信念建议同向比例=round(float((np.sign(C[:, m]) == R[:, m]).mean()), 3)))
res["真实数据_按状态持续轮数"] = rows

# —— 反事实环境：他人人数序列 O_t（G = O ≤ 59，S = O ≥ 61，公布人数 = O）——
rng0 = np.random.default_rng(1)
envs = {"真实人数序列": L.ATT.copy(),
        "打乱时间顺序（同分布、无时间结构）": rng0.permutation(L.ATT),
        "每轮必翻转（挤/不挤交替）": np.where(np.arange(L.T) % 2 == 0, 66.0, 54.0),
        "每 5 轮翻转一次": np.where((np.arange(L.T) // 5) % 2 == 0, 66.0, 54.0)}
sim = {}
for name, O in envs.items():
    G = np.tile((O <= L.CAP - 1).astype(float), (L.N_SUBJ, 1)); S = np.tile((O >= L.CAP + 1).astype(float), (L.N_SUBJ, 1))
    hs, rep = [], []
    for k in range(20):
        A = np.zeros(L.A_REAL.shape)
        L.run(X, sp, phi, A, G, S, O, rng=np.random.default_rng(100 + k), sigma=fit["sigma"], eps=np.random.default_rng(200 + k).standard_normal(L.T))
        Ck = habit_trace(A)
        hs.append(np.abs(Ck[:, 5:]).mean()); rep.append((A[:, 1:] == A[:, :-1]).mean())
    sim[name] = dict(平均习惯强度=round(float(np.mean(hs)), 3), 重复上一轮比例=round(float(np.mean(rep)), 3))
res["反事实开环模拟"] = sim
L.save_json(res, L.OUT / "习惯与宏观.json")
print(json.dumps(res, ensure_ascii=False, indent=1))
