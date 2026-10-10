# -*- coding: utf-8 -*-
"""探索性核对（研究中在命令行里临时算过、报告中引用的几组数字，这里集中成可复现的脚本）
  1 宏观事实：个人去的比例的分布、常客人数、每轮预测"不挤"的比例、人数 SD 与"各自独立随机"的二项 SD、换人率
  2 预测规则本身的准确度按 100 轮区块：延续 vs 镜像（挤 / 不挤判对比例）与 ACF1
  3 预测方向类型的漂移：前后两半的类型转移表、方向系数的平均变化、按区块的平均方向系数
  4 对 4 轮前人数的逐人反应与个人参数的关系（s13 的逐人系数 × HRGPR 参数、ρ_f）
  5 工具变量按习惯方向分组（习惯者 κ̂ > 0、交替者 κ̂ < 0）：报"不挤"的效应、累积分布差、给定本轮选择时上一轮选择与预测的关系
输出：结果/探索_杂项.json
"""
import json, collections
import numpy as np
from scipy import stats
from scipy.special import expit
import pred_lib as PL
P, N, A, OK, n, T = PL.P, PL.N, PL.A, PL.OK, PL.n, PL.T
J = lambda f: json.loads((PL.OUT / f).read_text(encoding="utf-8"))
out = {}
# 1 宏观事实
g = A.mean(1); sh = np.array([np.mean(P[OK[:, t], t] <= 60) for t in range(T)])
out["1 宏观事实"] = dict(个人去的比例分位={str(q): float(np.percentile(g, q)) for q in (5, 10, 25, 50, 75, 90, 95)},
                     常客人数_去大于08=int((g > .8).sum()), 去的比例小于02=int((g < .2).sum()),
                     每轮预测不挤的比例=dict(均值=float(sh.mean()), SD=float(sh.std()), 与本轮人数相关=float(np.corrcoef(sh, N)[0, 1]),
                                       与上一轮人数相关=float(np.corrcoef(sh[1:], N[:-1])[0, 1])),
                     人数SD=float(N.std()), 独立随机的二项SD=float(np.sqrt(100 * g.mean() * (1 - g.mean()))),
                     各人按自己比例独立随机的SD=float(np.sqrt(np.sum(g * (1 - g)))), 换人率=float((A[:, 1:] != A[:, :-1]).mean()))
# 2 规则准确度按区块
Nf = N.astype(float); blk = []
for k in range(4):
    t = np.arange(max(1, k * 100), (k + 1) * 100); same = np.mean((Nf[t - 1] > 60) == (Nf[t] > 60))
    blk.append(dict(区块=f"{k * 100 + 1}–{(k + 1) * 100}", 延续判对=float(same), 镜像判对=float(1 - same), ACF1=float(np.corrcoef(Nf[t - 1], Nf[t])[0, 1])))
out["2 规则准确度按区块"] = blk
# 3 类型漂移
NP = PL.NPREV; sgn = np.sign(NP - 60.5); dd = (NP - 60) / 10; tix = np.arange(T)
def slope_t(sel):
    b = np.zeros(n); tv = np.zeros(n)
    for i in range(n):
        m = OK[i] & sel & ~np.isnan(NP); y = P[i, m] - 60; X = np.column_stack([np.ones(m.sum()), sgn[m], dd[m]])
        q, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ q; s2 = e @ e / (len(y) - 3)
        b[i] = q[1]; tv[i] = q[1] / np.sqrt((s2 * np.linalg.inv(X.T @ X))[1, 1])
    return b, tv
ty = lambda t: np.where(t > 1.96, "外推", np.where(t < -1.96, "反转", "无方向"))
b1, t1 = slope_t(tix < T // 2); b2, t2 = slope_t(tix >= T // 2)
tr = collections.Counter(zip(ty(t1), ty(t2)))
out["3 类型漂移"] = dict(转移表={f"{a}→{b}": int(v) for (a, b), v in tr.items()},
                     方向系数平均变化_后减前=float(np.mean(b2 - b1)), Wilcoxon_p=float(stats.wilcoxon(b2 - b1).pvalue),
                     按区块=[dict(区块=k + 1, 平均方向系数=float(slope_t((tix >= k * 100) & (tix < (k + 1) * 100))[0].mean())) for k in range(4)])
# 4 4 轮前反应的个体关联
d13 = J("s13_四轮前与ACF4.json"); ba = np.array(d13["逐人选择系数"])
X4 = PL.model_states("HRGPR")["X"]; rf = expit(np.array(J("s10_合理化联合模型_ρ逐人.json")["X"])[:, 7])
out["4 四轮反应的个体关联（Spearman）"] = {nm: float(stats.spearmanr(ba, v)[0]) for nm, v in
                                         (("ρ", expit(X4[:, 0])), ("β", X4[:, 1]), ("|β|", np.abs(X4[:, 1])), ("κ", X4[:, 2]), ("ρ_f", rf))}
out["4 四轮反应的分组均值"] = dict(交替者=float(ba[X4[:, 2] < 0].mean()), 习惯者=float(ba[X4[:, 2] > 0].mean()))
# 5 工具变量按习惯方向分组（复用 s9b 的数据准备）
ns = {}; exec(open(PL.W / "代码" / "s9b_γ稳健性与报告分布.py", encoding="utf-8").read().split("out = {}")[0], ns)
ii, tt, a, c, Xc, nm, kfull, yN = ns["ii"], ns["tt"], ns["a"], ns["c"], ns["Xc"], ns["nm"], ns["kfull"], ns["yN"]
Xnc = Xc[:, 1:]; grp = {}
for lab, sel in (("习惯者 κ̂ > 0", kfull[ii] > 0), ("交替者 κ̂ < 0", kfull[ii] < 0)):
    r = {}
    yk = (P[ii, tt] <= 60).astype(float); q = PL.iv_fe(yk[sel], a[sel], Xnc[sel], c[sel][:, None], ii[sel], nm[1:])
    r["报不挤的效应"] = q["内生变量"]
    r["累积分布差"] = {}
    for k in (55, 57, 59, 60, 61, 63, 65):
        yk = (P[ii, tt] <= k).astype(float); q = PL.iv_fe(yk[sel], a[sel], Xnc[sel], c[sel][:, None], ii[sel], nm[1:]); r["累积分布差"][str(k)] = q["内生变量"]["b"]
    for at in (0, 1):
        s2 = sel & (a == at)
        q = PL.ols_fe(yN[s2], np.column_stack([A[ii, tt - 1][s2], Xnc[s2]]), ii[s2], ["上轮去"] + nm[1:])["上轮去"]
        r["本轮" + ("去" if at else "留") + "：上轮去 → 预测"] = q
    grp[lab] = r
out["5 按习惯方向分组的工具变量"] = grp
PL.save(out, "探索_杂项.json")
print(json.dumps(out, ensure_ascii=False)[:4000])
