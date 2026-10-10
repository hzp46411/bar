# -*- coding: utf-8 -*-
"""
第 31 步：核心问题——人在 El Farol 中用什么内部模型（用主模型的个人参数回答）
  A 内部模型的构成（世界模型 W 的个人参数）：
    方向 δ（> 1 反转、< −1 外推、其余接近零）；平均水平的学习率 η；分级反应 κ；预测噪声 σ
  B 同一个内部模型是否同时驱动预测与选择（两条数据流的交叉检验，不依赖模型的量）：
    预测侧：上轮不挤之后的平均预测 − 上轮挤之后的平均预测（= 不依赖模型的 δ）
    选择侧：上轮挤之后去的比例 − 上轮不挤之后去的比例
    若选择由同一个预期驱动：反转者（预期挤之后不挤）在挤之后更愿意去，外推者相反 → 两者跨人正相关
    另报告：w_B（世界模型在选择中的权重）、选择侧状态效应与模型 δ 的相关
  C 两个系统的控制份额：每人在真实历史上 |习惯项|（H + D）与 |世界层各项|（W + L + λ）的平均绝对贡献之比
  D 内部模型的稳定性：前后半程分别计算不依赖模型的 δ，跨人相关
用法：python3 s31_内部模型的内容.py <模型>     模型取 s28 的真实拟合（如 HDWLG1）
输出：结果/s31_内部模型_<模型>.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr, pearsonr
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s27", str(pathlib.Path(__file__).with_name("s27_ABM.py")))
S27 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S27)
T = PL.T
A, P, OK = PL.A, PL.P, PL.OK
N = PL.N.astype(float)
CP = (np.r_[PL._raw.iloc[0, 1:].astype(float).values[-T - 1], N[:-1]] >= 61)


def state_contrast(t_mask):
    """不依赖模型的两个量（逐人）：预测侧 δ̂ 与选择侧的状态效应。"""
    fc, ch = np.full(A.shape[0], np.nan), np.full(A.shape[0], np.nan)
    for i in range(A.shape[0]):
        ok = OK[i] & t_mask
        if (ok & ~CP).sum() >= 5 and (ok & CP).sum() >= 5:
            fc[i] = P[i, ok & ~CP].mean() - P[i, ok & CP].mean()
        ch[i] = A[i, t_mask & CP].mean() - A[i, t_mask & ~CP].mean()
    return fc, ch


def corr(x, y):
    k = ~np.isnan(x) & ~np.isnan(y)
    r, p = pearsonr(x[k], y[k]); rs, ps = spearmanr(x[k], y[k])
    return dict(Pearson=round(float(r), 3), p=float(f"{p:.3g}"), Spearman=round(float(rs), 3), Spearman_p=float(f"{ps:.3g}"), n=int(k.sum()))


if __name__ == "__main__":
    m = sys.argv[1]
    S27.SRC[m] = f"s28_真实_{m}.json"; S27.FIXED[m] = (-50.0 if "F" in m else 50.0, 1.0)
    X = S27.load(m)
    delta, eta, kap, sig, wB, wI, wD = X[:, 6], expit(X[:, 5]), X[:, 14], np.exp(X[:, 7]), X[:, 4], X[:, 1], X[:, 9]
    q = lambda v: [round(float(x), 4) for x in np.percentile(v, [10, 25, 50, 75, 90])]
    out = {"模型": m}
    out["A 内部模型的构成"] = dict(
        方向δ=dict(分位数_10_25_50_75_90=q(delta), 反转_大于1=int((delta > 1).sum()), 外推_小于负1=int((delta < -1).sum()),
                 接近零=int((np.abs(delta) <= 1).sum())),
        平均水平的学习率η=dict(分位数=q(eta), 记忆轮数中位数=round(float(np.median(1 / eta)), 1)),
        分级反应κ=dict(分位数=q(kap), 大于0的人数=int((kap > 0).sum())), 预测噪声σ=dict(分位数=q(sig)),
        世界模型权重wB=dict(分位数=q(wB)))
    fc, ch = state_contrast(np.ones(T, bool))
    out["B 预测与选择的交叉检验"] = dict(
        不依赖模型的预测侧δ与模型δ=corr(fc, delta),
        预测侧δ与选择侧状态效应=corr(fc, ch),
        选择侧状态效应与模型δ=corr(ch, delta),
        说明="选择侧状态效应 = 上轮挤之后去的比例 − 上轮不挤之后去的比例；同一个预期驱动时与 δ 正相关",
        按类型的选择侧状态效应={k: round(float(np.nanmean(ch[s])), 4) for k, s in
                         (("反转", delta > 1), ("外推", delta < -1), ("接近零", np.abs(delta) <= 1))})
    terms, _ = S27.open_loop(X)
    hab = np.abs(terms["H"] + terms["D"]).mean(1); wor = np.abs(terms["B"] + terms["L"] + terms["R"]).mean(1)
    share = wor / (hab + wor)
    out["C 两个系统的控制份额"] = dict(世界层份额=dict(分位数=q(share), 均值=round(float(share.mean()), 3)),
                                习惯项平均绝对贡献=round(float(hab.mean()), 3), 世界层平均绝对贡献=round(float(wor.mean()), 3),
                                世界层为主的人数=int((share > 0.5).sum()))
    half = np.arange(T) < T // 2
    f1, c1 = state_contrast(half); f2, c2 = state_contrast(~half)
    out["D 稳定性（前后半程）"] = dict(预测侧δ=corr(f1, f2), 选择侧状态效应=corr(c1, c2))
    PL.save(out, f"s31_内部模型_{m}.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))
