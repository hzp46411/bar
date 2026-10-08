# -*- coding: utf-8 -*-
"""第 11 步：合理化会不会制造出前面的结论？——按 ρ_i（s10 ρ逐人）把人分成"如实报告者"（ρ_i < 0.05）与"合理化者"（ρ_i > 0.5），逐项重做
  1 类型汇聚（s4）：由预测得到的外推 / 逆向类型与选择模型 BBL β 的符号是否一致
  2 人内同步（s4）：定向后的模型信念 V 与预测的逐人相关
  3 经验效应（s8）：一致率与心理测量斜率的前后变化；以及工具变量下"选择 → 报告不挤"的效应前后两半是否不同（合理化随经验增强？）
  4 冲突时跟随习惯（s3）：信念（预测类别）与习惯方向冲突时，跟随习惯的比例及其随稳定程度的变化
输出：结果/s11_合理化对前面结论的影响.json
"""
import json
import numpy as np
from scipy import stats
from scipy.special import expit
from scipy.optimize import minimize
import pred_lib as PL
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
L = PL.L; STD = L.load_std()
ms = PL.model_states("HRGPR"); beta = ms["beta"]; V = ms["V"]; C = ms["c"]
d10 = json.loads((PL.OUT / "s10_合理化联合模型_ρ逐人.json").read_text(encoding="utf-8"))
rho = expit(np.array(d10["X"])[:, 7])
G = {"如实报告者 ρ_i<0.05": rho < 0.05, "中间 0.05–0.5": (rho >= 0.05) & (rho <= 0.5), "合理化者 ρ_i>0.5": rho > 0.5}
NP = PL.NPREV; sl = slice(1, None); sg = np.sign(NP - 60.5)[sl]; dd = (NP - 60)[sl] / 10
out = {"人数": {k: int(v.sum()) for k, v in G.items()}}
# 1 / 2
sgn = np.zeros(n); r_i = np.zeros(n)
for i in range(n):
    m = OK[i, 1:]; y = P[i, 1:][m] - 60; X = np.column_stack([np.ones(m.sum()), sg[m], dd[m]])
    sgn[i] = np.linalg.lstsq(X, y, rcond=None)[0][1]
    r_i[i] = stats.pearsonr(np.sign(beta[i]) * V[i, 1:][m], P[i, 1:][m])[0]
agree = np.sign(sgn) == np.sign(beta)
cons = np.where(OK, A == (P <= 60), np.nan)
h = T // 2
def slope(i, sel):
    m = OK[i] & sel; x = (60.5 - P[i][m]) / 10; a = A[i][m]
    f = lambda q: float(np.sum(np.logaddexp(0, q[0] + q[1] * x) - a * (q[0] + q[1] * x)))
    return minimize(f, [0, 0], method="L-BFGS-B", bounds=[(-10, 10), (-20, 20)]).x[1]
tix = np.arange(T)
s_early = np.array([slope(i, tix < h) for i in range(n)]); s_late = np.array([slope(i, tix >= h) for i in range(n)])
# 4 冲突：信念方向 = 预测类别（不挤 → 去），习惯方向 = sign(c)；冲突时跟随习惯
stab = (PL.PM["stab"] - STD["stab"][0]) / STD["stab"][1]
for k, g in G.items():
    r = {}
    r["类型一致率"] = float(agree[g].mean())
    r["人内同步 为负（方向正确）的比例"] = float((r_i[g] < 0).mean())
    r["一致率 前半 → 后半"] = [float(np.nanmean(cons[g][:, :h])), float(np.nanmean(cons[g][:, h:]))]
    r["心理测量斜率中位数 前半 → 后半"] = [float(np.median(s_early[g])), float(np.median(s_late[g]))]
    bdir = np.where(P <= 60, 1, 0); hdir = (C > 0).astype(int)
    conf = OK & (bdir != hdir) & (np.abs(C) > 0.2) & g[:, None]
    fh = (A == hdir)
    r["冲突时跟随习惯的比例"] = float(fh[conf].mean())
    q = np.nanpercentile(stab, [33, 67]); tt = np.broadcast_to(stab[None], A.shape)
    r["冲突时跟随习惯：稳定 低 / 中 / 高"] = [float(fh[conf & (tt <= q[0])].mean()), float(fh[conf & (tt > q[0]) & (tt <= q[1])].mean()), float(fh[conf & (tt > q[1])].mean())]
    out[k] = r; print(k, r, flush=True)
# 3b 工具变量：选择 → 报告"不挤"，前后两半
src = open(PL.W / "代码" / "s9b_γ稳健性与报告分布.py", encoding="utf-8").read()
ns = {}; exec(src.split("out = {}")[0], ns)                           # 复用 s9b 的数据准备（工具、控制、样本）
ii, tt_, a, Xc, Z, nm = ns["ii"], ns["tt"], ns["a"], ns["Xc"], ns["Z"], ns["nm"]
yk = (P[ii, tt_] <= 60).astype(float)
iv = {}
for lab, sel in (("前半", tt_ < h), ("后半", tt_ >= h)):
    q = PL.iv_fe(yk[sel], a[sel], Xc[sel], Z[sel], ii[sel], nm); iv[lab] = dict(效应=q["内生变量"], F=q["第一阶段F"])
for lab, gsel in G.items():
    sel = gsel[ii]
    if sel.sum() > 1000:
        q = PL.iv_fe(yk[sel], a[sel], Xc[sel], Z[sel], ii[sel], nm); iv[lab] = dict(效应=q["内生变量"], F=q["第一阶段F"])
out["工具变量：选择 → 报告不挤"] = iv; print(json.dumps(iv, ensure_ascii=False)[:800])
PL.save(out, "s11_合理化对前面结论的影响.json")
