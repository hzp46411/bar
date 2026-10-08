# -*- coding: utf-8 -*-
"""
闭环第三环 · 真实数据上的检验：被调节的微观能否"预测"下一轮的宏观？
  用真实历史（开环），每轮由模型算出每人去的概率 p_it（对同轮共同冲击 σ 积分）。
  水平：E[N_t] = Σ p_it；结构：E[换人率_t] = 平均(上轮去 ? 1 − p_it : p_it)。
  把某条调节通路关掉（参数置 0，个体参数不变）得到"无调节"预测，差值 Δ_t = 完整 − 无调节 就是这条通路对宏观的净贡献。
  回归：实际_t = a + b1·无调节预测_t + b2·Δ_t；b2 ≈ 1 且显著 = 这条通路在宏观上留下了真实可检验的足迹。
预期（逻辑分析）：调节作用在习惯方向 c 上，对人数相互抵消（Δ_N 小、难检验），对换人同号相加（Δ_sw 大、可检验）。
输出：结果/第三环_一步预测_<模型>.json
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import t as tdist
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

A, G, S, N = L.A_REAL, L.G_REAL, L.S_REAL, L.ATT
MODEL = sys.argv[1] if len(sys.argv) > 1 else "HRGPRS"
FIT = json.loads((L.OUT / "拟合" / f"{MODEL}.json").read_text(encoding="utf-8"))
SP = L.Spec(**FIT["spec"]); X = np.array(FIT["X"]); SH = dict(FIT["shared"]); NAMES = SP.names(); SIG = FIT["sigma"]
PATHS = {"稳定": ["θR_stab", "θG_stab", "ψ_stab"], "偏离": ["θR_dev", "θG_dev", "ψ_dev"], "可靠性": ["θB_relB", "θH_relH"],
         "宏观调节合计(稳定+偏离+可靠性)": ["θR_stab", "θG_stab", "ψ_stab", "θR_dev", "θG_dev", "ψ_dev", "θB_relB", "θH_relH"],
         "轮次(个人学习)": ["θR_time", "θG_time"]}
x_gh, w_gh = np.polynomial.hermite_e.hermegauss(20); w_gh = w_gh / w_gh.sum()


def channel_off(which, sh=None):
    """只关掉某一个通道上的宏观调节（稳定、偏离、可靠性），另一个通道保持不变。
    对调节变量 m：信念效应 eb = θG + θR/2，惯性效应 eh = θG − θR/2。
    关信念通道：θG = eh/2, θR = −eh，θB_relB = 0；关惯性通道：θG = eb/2, θR = eb，θH_relH = 0，推力 ψ = 0。"""
    sh = dict(sh or SH); new = {}
    for m in ("stab", "dev"):
        eb = sh[f"θG_{m}"] + sh[f"θR_{m}"] / 2; eh = sh[f"θG_{m}"] - sh[f"θR_{m}"] / 2
        if which == "信念":
            new[f"θG_{m}"], new[f"θR_{m}"] = eh / 2, -eh
        else:
            new[f"θG_{m}"], new[f"θR_{m}"] = eb / 2, eb; new[f"ψ_{m}"] = 0.0
    new["θB_relB" if which == "信念" else "θH_relH"] = 0.0
    return new


def preds(zero=(), setv=None):
    sh = dict(SH)
    sh.update(setv or {})
    for k in zero:
        if k in sh:
            sh[k] = 0.0
    z = L.run(X, SP, np.array([sh[k] for k in NAMES]), A, G, S, N, out="z")
    p = np.tensordot(w_gh, expit(z[None] + SIG * x_gh[:, None, None]), 1)       # 对 σ 积分后的个人概率
    EN = p.sum(0)
    prev = np.c_[np.full((A.shape[0], 1), np.nan), A[:, :-1]]
    Esw = np.nanmean(np.where(prev == 1, 1 - p, p), 0)
    return EN, Esw


def ols(y, Xm):
    b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    e = y - Xm @ b; n, k = Xm.shape
    XtXi = np.linalg.inv(Xm.T @ Xm)
    V = XtXi @ (Xm.T * e ** 2) @ Xm @ XtXi * n / (n - k)                       # 稳健（HC1）标准误
    se = np.sqrt(np.diag(V)); return b, se, e


sl = slice(5, None)
sw_real = np.r_[np.nan, (A[:, 1:] != A[:, :-1]).mean(0)]
full_N, full_sw = preds()
out = {"模型": MODEL, "完整模型": dict(人数_R2=None, 换人率_R2=None), "通路": {}}
for y, Ef, lab in ((N.astype(float), full_N, "人数"), (sw_real, full_sw, "换人率")):
    e = y[sl] - Ef[sl]; out["完整模型"][f"{lab}_R2"] = round(float(1 - e.var() / y[sl].var()), 4)
PATHS.update({"只关信念通道": channel_off("信念"), "只关惯性通道": channel_off("惯性")})
for pth, ks in PATHS.items():
    n0, s0 = preds(setv=ks) if isinstance(ks, dict) else preds(ks)
    res = {}
    for y, Ef, E0, lab in ((N.astype(float), full_N, n0, "人数"), (sw_real, full_sw, s0, "换人率")):
        d = (Ef - E0)[sl]; yy = y[sl]; Xm = np.column_stack([np.ones_like(d), E0[sl], d])
        b, se, e = ols(yy, Xm)
        e0 = yy - np.column_stack([np.ones_like(d), E0[sl]]) @ np.linalg.lstsq(np.column_stack([np.ones_like(d), E0[sl]]), yy, rcond=None)[0]
        df = len(yy) - 3
        res[lab] = dict(Δ的平均绝对值=round(float(np.abs(d).mean()), 4), Δ的SD=round(float(d.std()), 4),
                        b2=round(float(b[2]), 3), b2_SE=round(float(se[2]), 3), t=round(float(b[2] / se[2]), 2),
                        p=float(2 * tdist.sf(abs(b[2] / se[2]), df)),
                        增加的R2=round(float(e0.var() - e.var()) / float(yy.var()), 4))
    out["通路"][pth] = res
    print(pth, res, flush=True)
print(out["完整模型"])
L.save_json(out, L.OUT / f"第三环_一步预测_{MODEL}.json")
