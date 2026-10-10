# -*- coding: utf-8 -*-
"""第 15 步：内部模型"普查"——每个人用的是什么内部模型？
  把前面各步对每个人的估计汇成一张"内部模型卡片"，再做交叉表：
    预测部分：方向（外推 / 反转 / 无方向，s4 的逐人符号系数是否显著）、记忆长度（s12 的 ρᴾ：快 > 0.9、慢 < 0.1、中间）、
              锚定（预测落在 58–62 的比例）、主要预测规则（s16 的后验份额最大者，若已完成）
    决策部分：信念—行动的真实一致率（s10 模型推断，去掉合理化后）、习惯方向（HRGPR 的 κ 符号）、合理化（s10 的 ρ_f）、
              当下信念对选择的额外作用（J1 的 s_i）、常客（去的比例 > 0.8）
  交叉表：预测方向 × 决策类型（如实报告者 / 中间 / 合理化者）；预测方向 × 选择中的 BBL 方向；记忆长度 × 合理化
输出：结果/s15_内部模型普查.json
"""
import json
import numpy as np
from scipy.special import expit
from scipy.stats import norm
import pred_lib as PL
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
J = lambda f: json.loads((PL.OUT / f).read_text(encoding="utf-8"))
ms = PL.model_states("HRGPR"); X4 = ms["X"]
# 预测方向（含显著性）
NP = PL.NPREV; sgn = np.sign(NP - 60.5); dd = (NP - 60) / 10
tval = np.zeros(n); coef = np.zeros(n)
for i in range(n):
    m = OK[i] & ~np.isnan(NP); y = P[i, m] - 60; X = np.column_stack([np.ones(m.sum()), sgn[m], dd[m]])
    q, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ q; s2 = e @ e / (len(y) - 3)
    coef[i] = q[1]; tval[i] = q[1] / np.sqrt((s2 * np.linalg.inv(X.T @ X))[1, 1])
dirn = np.where(tval > 1.96, "外推", np.where(tval < -1.96, "反转", "无方向"))
rp = np.array(J("s12_BBL信念与预测_BBL.json")["ρᴾ"])
mem = np.where(rp > 0.9, "只记上一轮", np.where(rp < 0.1, "长记忆", "中等记忆"))
anchor = np.nanmean(np.where(OK, (np.abs(P - 60) <= 2).astype(float), np.nan), 1)
d10 = J("s10_合理化联合模型_ρ逐人.json"); X10 = np.array(d10["X"]); rf = expit(X10[:, 7])
dec = np.where(rf < 0.05, "如实报告者", np.where(rf > 0.5, "合理化者", "中间"))
# 去掉合理化后的信念—行动一致率（s10 模型推断）
sys_argv = __import__("sys").argv; __import__("sys").argv = ["x", "ρ逐人"]
ns = {}; exec(open(PL.W / "代码" / "s10_合理化联合模型.py", encoding="utf-8").read().split("if __name__")[0], ns)
__import__("sys").argv = sys_argv
sh = np.array([d10["共用"][k] for k in ns["SH"]]); hon = np.zeros(n)
for i in range(n):
    mu, ws, wm, logv, b, s, kap, lr = X10[i]; _, dlt, psi, lam = sh
    m = ns["M"][i]; Bb = mu + ws * ns["SG"][m] + wm * ns["DM"][m]; q = norm.cdf((60.5 - Bb) / np.exp(0.5 * logv))
    z0 = b + kap * ns["Cc"][i, m] * np.exp(dlt * ns["DEV"][m]) + psi * ns["STAB"][m] * ns["Cc"][i, m] + lam * ns["LAG"][m]
    p1 = expit(z0 + s / 2); p0 = expit(z0 - s / 2); hon[i] = np.mean(q * p1 + (1 - q) * (1 - p0))
habit = np.where(X4[:, 2] > 0, "重复", "交替")
bbl = np.where(X4[:, 1] > 0, "β>0", "β<0")
sJ1 = np.array(J("s14_HRGPR预测联合模型_J1.json")["X"])[:, 7]
regular = A.mean(1) > 0.8
card = dict(预测方向=dirn.tolist(), 预测方向t=tval.tolist(), 记忆=mem.tolist(), ρᴾ=rp.tolist(), 锚定比例=anchor.tolist(), 决策类型=dec.tolist(), ρ_f=rf.tolist(),
            真实一致率=hon.tolist(), 习惯方向=habit.tolist(), 选择BBL方向=bbl.tolist(), 当下信念作用s=sJ1.tolist(), 常客=regular.tolist())
s16p = PL.OUT / "s16_Arthur机制检验.json"
if s16p.exists():
    d16 = J("s16_Arthur机制检验.json"); resp = np.array(d16["逐人后验份额"]); labs = ["其他"] + d16["规则"]
    card["主要预测规则"] = [labs[k] for k in resp.argmax(1)]
def xtab(a, b):
    ua, ub = sorted(set(a)), sorted(set(b))
    return {x: {y: int(np.sum((np.array(a) == x) & (np.array(b) == y))) for y in ub} for x in ua}
out = {"人数": n, "卡片": card,
       "边际分布": {k: {v: int(np.sum(np.array(card[k]) == v)) for v in sorted(set(card[k]))} for k in ("预测方向", "记忆", "决策类型", "习惯方向", "选择BBL方向")},
       "交叉表": {"预测方向 × 决策类型": xtab(dirn, dec), "预测方向 × 选择BBL方向": xtab(dirn, bbl), "记忆 × 决策类型": xtab(mem, dec),
                 "决策类型 × 习惯方向": xtab(dec, habit)},
       "按决策类型": {g: dict(人数=int((dec == g).sum()), 真实一致率=float(hon[dec == g].mean()), 锚定比例=float(anchor[dec == g].mean()),
                           当下信念作用s中位数=float(np.median(sJ1[dec == g])), 常客=int(regular[dec == g].sum())) for g in ("如实报告者", "中间", "合理化者")},
       "锚定比例中位数": float(np.median(anchor))}
if "主要预测规则" in card:
    out["边际分布"]["主要预测规则"] = {v: int(np.sum(np.array(card["主要预测规则"]) == v)) for v in sorted(set(card["主要预测规则"]))}
    out["交叉表"]["主要预测规则 × 预测方向"] = xtab(card["主要预测规则"], dirn)
PL.save(out, "s15_内部模型普查.json")
print(json.dumps({k: v for k, v in out.items() if k != "卡片"}, ensure_ascii=False, indent=0)[:3000])
