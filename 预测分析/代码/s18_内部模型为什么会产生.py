# -*- coding: utf-8 -*-
"""第 18 步：这些内部模型为什么会产生、为什么一直并存？——选择压力与频率依赖
  A 闭环中的频率依赖：保留每人的 |β_i| 与其他参数，只改变"外推者"（β > 0）在人群中的比例 f（0、0.1、…、1；每格 120 次，随机指定谁是外推者），
    记录外推者与反向者各自的平均得分、两者之差 Δ(f)，以及群体的平均人数、SD、效率。
    负的频率依赖（Δ 随 f 下降）意味着：任何一种信念方式一旦变多就吃亏——混合人群自我维持；Δ(f*) = 0 处为稳定的混合比例。
    再对习惯方向（κ > 0 的重复者 vs κ < 0 的交替者）做同样的检验。
  B 真实数据中的选择压力：按类型比较每人的平均得分与预测准确度
    类型：信念方向（预测中的外推 / 反转 / 无方向，来自 s4 的逐人符号系数及其显著性）；选择中的 BBL 方向（β 符号）；
         习惯方向（κ 符号）；合理化（s10 的 ρ_f）；常客（去的比例 > 0.8）
  C 类型的稳定性：前 200 轮与后 200 轮的逐人特征相关（预测的方向系数、去的比例、选择—预测一致率、换选率）
输出：结果/s18_内部模型为什么会产生.json
"""
import json, sys, importlib.util, pathlib
import numpy as np
from scipy import stats
from scipy.special import expit
import pred_lib as PL
spec = importlib.util.spec_from_file_location("s17", str(pathlib.Path(__file__).with_name("s17_宏观秩序的生成.py")))
S17 = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]; spec.loader.exec_module(S17)
X0, PHI0, SIG0, n, T = S17.X0, S17.PHI0, S17.SIG0, S17.n, S17.T
B = 120


def payoffs(Ns, A):
    crowd = Ns > 60
    return np.where(A == 1, (~crowd)[None] * 1.0, crowd[None] * 0.7).mean(1)          # 每人平均得分


def freq_scan(col, label):
    res = []
    mag = np.abs(X0[:, col]); rng = np.random.default_rng(11 + col)
    for f in np.round(np.linspace(0, 1, 11), 2):
        d, m, sd, eff, pp, pm = [], [], [], [], [], []
        for k in range(B):
            pos = np.zeros(n, bool); pos[rng.choice(n, int(round(f * n)), replace=False)] = True
            X = X0.copy(); X[:, col] = np.where(pos, mag, -mag)
            Ns, A = S17.simulate(X, PHI0, SIG0, 300000 + 1000 * col + k)
            py = payoffs(Ns, A)
            if 0 < pos.sum() < n: d.append(py[pos].mean() - py[~pos].mean()); pp.append(py[pos].mean()); pm.append(py[~pos].mean())
            m.append(Ns.mean()); sd.append(Ns.std()); eff.append(py.mean())
        row = dict(f=float(f), 平均人数=float(np.mean(m)), SD=float(np.mean(sd)), 效率=float(np.mean(eff)),
                   正向者得分=float(np.mean(pp)) if pp else None, 负向者得分=float(np.mean(pm)) if pm else None,
                   得分差=float(np.mean(d)) if d else None, 得分差SE=float(np.std(d) / np.sqrt(len(d))) if d else None)
        res.append(row); print(label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    xs = np.array([r["f"] for r in res if r["得分差"] is not None]); ys = np.array([r["得分差"] for r in res if r["得分差"] is not None])
    slope = float(np.polyfit(xs, ys, 1)[0])
    # Δ(f*) = 0 的位置（线性插值）
    fstar = None
    for a, b_ in zip(res[:-1], res[1:]):
        if a["得分差"] is not None and b_["得分差"] is not None and a["得分差"] * b_["得分差"] <= 0:
            fstar = a["f"] + (b_["f"] - a["f"]) * a["得分差"] / (a["得分差"] - b_["得分差"]); break
    return dict(各比例=res, 得分差对比例的斜率=slope, 得分相等的比例=fstar, 真实比例=float((X0[:, col] > 0).mean()))


if __name__ == "__main__":
    out = {}
    out["A 频率依赖：选择中的信念方向（β > 0 外推）"] = freq_scan(1, "β")
    PL.save(out, "s18_内部模型为什么会产生.json")
    out["A 频率依赖：习惯方向（κ > 0 重复）"] = freq_scan(2, "κ")
    PL.save(out, "s18_内部模型为什么会产生.json")
    # B 真实数据中的选择压力
    P, N, A, OK = PL.P, PL.N, PL.A, PL.OK
    py = payoffs(N.astype(float), A)
    mae = np.nanmean(np.where(OK, np.abs(P - N[None]), np.nan), 1)
    hit = np.nanmean(np.where(OK, ((P <= 60) == (N[None] <= 60)).astype(float), np.nan), 1)
    s4 = json.loads((PL.OUT / "s4_信念系统的直接测量.json").read_text(encoding="utf-8"))
    sg = np.array(s4["逐人符号系数"])
    # 逐人显著性：重算 t
    NP = PL.NPREV; sl = slice(1, None); sgn = np.sign(NP - 60.5)[sl]; dd = (NP - 60)[sl] / 10
    tval = np.zeros(n)
    for i in range(n):
        m = OK[i, 1:]; y = P[i, 1:][m] - 60; Xd = np.column_stack([np.ones(m.sum()), sgn[m], dd[m]])
        bb, *_ = np.linalg.lstsq(Xd, y, rcond=None); e = y - Xd @ bb; s2 = e @ e / (len(y) - 3)
        tval[i] = bb[1] / np.sqrt((s2 * np.linalg.inv(Xd.T @ Xd))[1, 1])
    btype = np.where(tval > 1.96, "外推", np.where(tval < -1.96, "反转", "无方向"))
    rf = expit(np.array(json.loads((PL.OUT / "s10_合理化联合模型_ρ逐人.json").read_text(encoding="utf-8"))["X"])[:, 7])
    gr = A.mean(1)
    groups = {"预测：外推": btype == "外推", "预测：反转": btype == "反转", "预测：无方向": btype == "无方向",
              "选择 BBL β > 0": X0[:, 1] > 0, "选择 BBL β < 0": X0[:, 1] < 0,
              "习惯 κ > 0（重复）": X0[:, 2] > 0, "习惯 κ < 0（交替）": X0[:, 2] < 0,
              "如实报告者 ρ_f < 0.05": rf < 0.05, "合理化者 ρ_f > 0.5": rf > 0.5, "常客（去 > 0.8）": gr > 0.8, "其余": gr <= 0.8}
    tab = {}
    for k, g in groups.items():
        tab[k] = dict(人数=int(g.sum()), 平均得分=float(py[g].mean()), 得分SD=float(py[g].std()), 预测MAE=float(mae[g].mean()), 挤不挤判对=float(hit[g].mean()))
    def mw(a, b): return float(stats.mannwhitneyu(py[a], py[b]).pvalue)
    out["B 真实数据：各类型的得分与准确度"] = dict(表=tab, 全体平均得分=float(py.mean()), 得分人间SD=float(py.std()),
        检验=dict(外推vs反转_得分p=mw(btype == "外推", btype == "反转"), β正vs负_得分p=mw(X0[:, 1] > 0, X0[:, 1] < 0),
                 κ正vs负_得分p=mw(X0[:, 2] > 0, X0[:, 2] < 0), 如实vs合理化_得分p=mw(rf < .05, rf > .5)))
    print(json.dumps(out["B 真实数据：各类型的得分与准确度"], ensure_ascii=False)[:1500], flush=True)
    # C 类型稳定性（前后两半）
    h = T // 2
    def feats(sel):
        Pm = np.where(OK & sel[None], P, np.nan)
        slope = np.zeros(n); g_ = np.zeros(n); cons = np.zeros(n); sw = np.zeros(n)
        for i in range(n):
            m = OK[i, 1:] & sel[1:]; y = P[i, 1:][m] - 60; Xd = np.column_stack([np.ones(m.sum()), sgn[m], dd[m]])
            slope[i] = np.linalg.lstsq(Xd, y, rcond=None)[0][1]
            g_[i] = A[i, sel].mean(); cons[i] = np.nanmean(np.where(OK[i] & sel, A[i] == (P[i] <= 60), np.nan))
            a_ = A[i, sel]; sw[i] = (a_[1:] != a_[:-1]).mean()
        return dict(预测方向系数=slope, 去的比例=g_, 一致率=cons, 换选率=sw)
    tix = np.arange(T)
    f1, f2 = feats(tix < h), feats(tix >= h)
    out["C 类型稳定性（前 200 轮 vs 后 200 轮的逐人相关）"] = {k: dict(Pearson=float(np.corrcoef(f1[k], f2[k])[0, 1]), Spearman=float(stats.spearmanr(f1[k], f2[k])[0])) for k in f1}
    out["C 预测方向的类型保持"] = dict(前半外推后半仍外推=float(np.mean(np.sign(f2["预测方向系数"][f1["预测方向系数"] > 0]) > 0)),
                                 前半反转后半仍反转=float(np.mean(np.sign(f2["预测方向系数"][f1["预测方向系数"] < 0]) < 0)))
    print(out["C 类型稳定性（前 200 轮 vs 后 200 轮的逐人相关）"], out["C 预测方向的类型保持"], flush=True)
    PL.save(dict(out, 逐人得分=py, 逐人预测类型=btype.tolist()), "s18_内部模型为什么会产生.json")
    print("完成")
