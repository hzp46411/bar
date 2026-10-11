# -*- coding: utf-8 -*-
"""
第 36 步：主模型 HB（习惯 choice kernel + 世界模型）之上的共同因素层，及其 ABM
  背景：HB 的 ABM 复现了平均人数、挤的比例、效率、换选择率，但人数的 ACF1（模拟 −.12，真实 −.35）与 ACF4（模拟 ≈ 0，真实 −.15）
        都在模拟分布之外。按原项目第 2 阶段的设定加入共同因素层（不增加任何个人参数）：
        z_it = d_it + c_t,    c_t = −γ·(N_{t−1} − 60)/10 + σ·ε_t,    ε_t ~ N(0, 1)
        d_it：HB 在真实历史上给出的个体 logit；γ：全体对上一轮偏离的共同反应（> 0 = 共同负反馈）；σ：同一轮共享冲击的大小
  估计（两步，与原项目相同）：个体参数固定为 HB 的层级估计，对 ε_t 做 40 点高斯–埃尔米特积分得到边际似然，求 γ、σ 的最大似然
  A 四个嵌套版本（无 / 只有 γ / 只有 σ / γ + σ）与似然比检验（σ = 0 在边界上，p 值用 χ²₀ 与 χ²₁ 各半的混合）；
    分级检验：加入"上一轮挤"的二分项后，γ 是否仍然显著
  B 恢复（理想条件：个体 logit 取真值，只估计 γ、σ）：以估计值为真值 30 套；γ = σ = 0 时 100 套（假阳性）
  C ABM：HB；HB + σ；HB + γ + σ（γ 在 ABM 中作为全体共同的 λ1 = −γ）；与真实宏观指标比较；另做扰动自稳
用法：python3 s36_共同因素层.py
输出：结果/s36_共同因素层.json
"""
import json, importlib.util, pathlib
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logit
from scipy.stats import chi2
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s27", str(pathlib.Path(__file__).with_name("s27_ABM.py")))
S27 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S27)
T = S27.T
A = S27.A_REAL
LAG = (np.r_[S27.N_FULL[-T - 1], S27.N_REAL[:-1]] - 60) / 10
CROWD = (np.r_[S27.N_FULL[-T - 1], S27.N_REAL[:-1]] >= 61).astype(float)
NODES, W = np.polynomial.hermite_e.hermegauss(40)
W = W / W.sum()


def loglik(d, A_, g, s, dc=0.0):
    z = d[None] - g * LAG - dc * CROWD + s * NODES[:, None, None]
    L = (A_[None] * z - np.logaddexp(0, z)).sum(1)
    mx = L.max(0)
    return float((mx + np.log(W @ np.exp(L - mx))).sum())


def estimate(d, A_):
    E = {"无": (0.0, 0.0, loglik(d, A_, 0, 0))}
    r = minimize_scalar(lambda g: -loglik(d, A_, g, 0), bounds=(-2, 2), method="bounded"); E["只有γ"] = (r.x, 0.0, -r.fun)
    r = minimize_scalar(lambda s: -loglik(d, A_, 0, s), bounds=(0, 2), method="bounded"); E["只有σ"] = (0.0, r.x, -r.fun)
    r = minimize(lambda v: -loglik(d, A_, v[0], abs(v[1])), x0=[E["只有γ"][0], max(E["只有σ"][1], 0.05)], method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-4)); E["γ+σ"] = (r.x[0], abs(r.x[1]), -r.fun)
    return E


def lr_tests(E):
    p_mix = lambda lr: 0.5 * chi2.sf(lr, 1) if lr > 0 else 1.0
    lr_g = 2 * (E["γ+σ"][2] - E["只有σ"][2]); lr_s = 2 * (E["γ+σ"][2] - E["只有γ"][2])
    return {"γ（在 σ 之上）": dict(LR=round(lr_g, 2), p=float(f"{chi2.sf(lr_g, 1):.3g}")),
            "σ（在 γ 之上）": dict(LR=round(lr_s, 2), p=float(f"{p_mix(lr_s):.3g}"))}


if __name__ == "__main__":
    X = S27.load("HB")
    _, P = S27.open_loop(X)
    d = logit(np.clip(P, 1e-9, 1 - 1e-9))
    out = {"个体模型": "HB（s22 的层级估计）"}
    E = estimate(d, A)
    out["A 估计"] = {k: dict(γ=round(g, 4), σ=round(s, 4), 对数似然=round(ll, 2)) for k, (g, s, ll) in E.items()}
    out["A 似然比检验"] = lr_tests(E)
    g_hat, s_hat = E["γ+σ"][0], E["γ+σ"][1]
    r = minimize(lambda v: -loglik(d, A, v[0], abs(v[1]), v[2]), x0=[g_hat, s_hat, 0.0], method="Nelder-Mead")
    lr_grad = 2 * (-r.fun - loglik(d, A, 0.0, abs(r.x[1]), r.x[2]))
    out["A 分级检验（加入上一轮挤的二分项）"] = dict(γ=round(r.x[0], 4), 二分项=round(r.x[2], 4), σ=round(abs(r.x[1]), 4),
                                         γ的LR=round(lr_grad, 2), p=float(f"{chi2.sf(lr_grad, 1):.3g}"))
    slope = float((P * (1 - P)).sum(0).mean())
    out["A 效应量"] = dict(上一轮多10人时这一轮人数的共同变化=round(-g_hat * slope, 2), 共同冲击1SD对应的人数=round(s_hat * slope, 2))
    print(json.dumps(out, ensure_ascii=False), flush=True)
    rng = np.random.default_rng(36)
    rec = []
    for k in range(30):
        eps = rng.standard_normal(T)
        As = (rng.random(A.shape) < 1 / (1 + np.exp(-(d - g_hat * LAG + s_hat * eps)))).astype(float)
        e = estimate(d, As)["γ+σ"]; rec.append((e[0], e[1]))
    rec = np.array(rec)
    fp = []
    for k in range(100):
        As = (rng.random(A.shape) < 1 / (1 + np.exp(-d))).astype(float)
        e = estimate(d, As); fp.append([lr_tests(e)["γ（在 σ 之上）"]["p"], lr_tests(e)["σ（在 γ 之上）"]["p"]])
    fp = np.array(fp)
    out["B 恢复（理想条件）"] = dict(真值=[round(g_hat, 4), round(s_hat, 4)],
                                γ估计=dict(均值=round(float(rec[:, 0].mean()), 4), SD=round(float(rec[:, 0].std()), 4)),
                                σ估计=dict(均值=round(float(rec[:, 1].mean()), 4), SD=round(float(rec[:, 1].std()), 4)),
                                γ与σ为0时的假阳性率_p小于05=dict(γ=float((fp[:, 0] < .05).mean()), σ=float((fp[:, 1] < .05).mean())))
    print("B 完成", flush=True)
    real = {k: float(v[0]) for k, v in S27.metrics(S27.N_REAL[None], A[None].astype(np.int8)).items()}
    Xc = X.copy(); Xc[:, 13] = -g_hat
    out["C ABM"] = {"HB（无共同因素）": S27.summarize(S27.metrics(*S27.simulate(X, 200, 1, 0.0)), real),
                    "HB + σ": S27.summarize(S27.metrics(*S27.simulate(X, 200, 1, E["只有σ"][1])), real),
                    "HB + γ + σ": S27.summarize(S27.metrics(*S27.simulate(Xc, 200, 1, s_hat)), real)}
    gains = {}
    for k, Y, sc in (("HB + γ + σ", Xc, s_hat), ("HB + σ", X, E["只有σ"][1])):
        Ns0, A0 = S27.simulate(Y, 100, 4, sc); base = Ns0.mean()
        p_bar = A0.astype(float).mean((0, 2)); sl = float((p_bar * (1 - p_bar)).sum())
        gains[k] = {f"Δb={db:+.1f}": round(1 - (S27.simulate(Y, 100, 4, sc, db=db)[0].mean() - base) / (sl * db), 3) for db in (-1.0, -0.5, 0.5, 1.0)}
    out["C 扰动自稳"] = gains
    PL.save(out, "s36_共同因素层.json")
    keys = ("平均人数", "SD", "ACF1", "ACF2", "ACF3", "ACF4", "挤的比例", "效率", "换选择率", "个人去的比例SD")
    for k, v in out["C ABM"].items():
        print(k, {kk: (v[kk]["均值"], v[kk]["真实的百分位"]) for kk in keys}, flush=True)
    print(json.dumps({k: out[k] for k in out if k != "C ABM"}, ensure_ascii=False, indent=1))
