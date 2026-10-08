# -*- coding: utf-8 -*-
"""大偏离之后被试靠什么做决定？用 HRG（精修后）把每人每轮的 logit 拆成四部分，按上一轮偏离分组：
  偏好 b、分级反应 λ·LAG（人数离 60 多远，带方向）、信念部分（BBL：只记"挤/不挤"）、惯性部分（习惯痕迹）
另报：模型无关的换选择方向（上一轮挤之后由去改不去 / 由不去改去）、决策确定性 |p − 0.5|。"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

fit = json.loads((L.OUT / "拟合" / "HRG.json").read_text(encoding="utf-8"))
sp = L.Spec(**fit["spec"]); X = np.array(fit["X"]); phi = np.array([fit["shared"][k] for k in sp.names()])
A = L.A_REAL
run = lambda Xv, ph=phi: L.run(Xv, sp, ph, A, L.G_REAL, L.S_REAL, L.ATT, out="z")
z = run(X); Xb = X.copy(); Xb[:, 1] = 0; Xk = X.copy(); Xk[:, 2] = 0
belief = z - run(Xb); habit = z - run(Xk)
pm, lag = L.public_mods(L.ATT)
graded = np.broadcast_to(fit["shared"]["lam"] * lag, z.shape)
bias = np.broadcast_to(X[:, 3:4], z.shape)
p = expit(z)
dev = np.abs(np.r_[0.0, L.ATT[:-1] - L.CAP]); crowd_prev = np.r_[False, L.ATT[:-1] > L.CAP]
prev = np.c_[np.full((len(A), 1), np.nan), A[:, :-1]]
bins = [(0, 2.5, "≤2 人"), (2.5, 5.5, "3–5 人"), (5.5, 8.5, "6–8 人"), (8.5, 99, "≥9 人")]
rows = []
for lo, hi, lab in bins:
    m = (dev > lo) & (dev <= hi) if lo > 0 else (dev <= hi); m[:3] = False
    seg = lambda x: float(np.abs(x[:, m]).mean())
    tot = seg(bias) + seg(graded) + seg(belief) + seg(habit)
    mm = np.broadcast_to(m, A.shape)
    rows.append(dict(上轮偏离=lab, 轮数=int(m.sum()),
                     偏好_b=round(seg(bias), 3), 分级反应_λ=round(seg(graded), 3), 信念_BBL=round(seg(belief), 3), 惯性_习惯=round(seg(habit), 3),
                     分级反应占比=round(seg(graded) / tot, 3), 信念占比=round(seg(belief) / tot, 3), 惯性占比=round(seg(habit) / tot, 3),
                     决策确定性=round(float(np.abs(p[:, m] - .5).mean()), 3),
                     实际换选择=round(float((A[mm] != prev[mm]).mean()), 3)))
# 换选择的方向：上一轮挤 / 不挤，按偏离大小
dirs = []
for crowded in (True, False):
    for lo, hi, lab in ((0, 3.5, "小（≤3 人）"), (5.5, 99, "大（≥6 人）")):
        m = (crowd_prev == crowded) & (dev > lo if lo > 0 else dev >= 0) & (dev <= hi); m[:3] = False
        mm = np.broadcast_to(m, A.shape)
        went = (prev == 1) & mm; stayed = (prev == 0) & mm
        dirs.append(dict(上一轮=("挤" if crowded else "不挤") + "，偏离" + lab, 轮数=int(m.sum()),
                         去过的人改为不去=round(float((A[went] == 0).mean()), 3), 没去的人改为去=round(float((A[stayed] == 1).mean()), 3),
                         去的比例=round(float(A[mm].mean()), 3)))
out = dict(按偏离的决策成分=rows, 换选择方向=dirs, λ=fit["shared"]["lam"])
L.save_json(out, L.OUT / "大偏离后的决策成分.json")
print(json.dumps(out, ensure_ascii=False, indent=1))
