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
  C ABM（与原项目第 3 阶段相同的事先固定的检验）：HB；HB + γ；HB + σ；HB + γ + σ，各 1000 次闭环模拟
    （γ 在 ABM 中作为全体共同的 λ1 = −γ；γ、σ 用各自嵌套版本的估计）
    宏观指纹：人数的均值、SD、ACF1–3
      联合检验：真实指纹到模拟分布中心的马氏距离²，协方差由模拟估计 → Hotelling 预测检验 F(k, B − k)；p > .05 = 相容
      合成似然（Wood, 2010）：模拟指纹视为多元正态时真实指纹的对数密度；人群之间的差用 bootstrap 区间
      各指标的双侧蒙特卡洛 p（加 1 校正）
    个体指标：D = 上一轮拥挤后去的比例 − 上一轮不拥挤后去的比例；换选择比例；锁定（拥挤轮次占比 < 5% 或 > 95%）
    ACF4 不在指纹中，只作描述
  D 扰动自稳（HB + σ；HB + γ + σ）
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


FP_NAMES = ["均值", "SD", "ACF1", "ACF2", "ACF3"]


def fingerprint(Ns):
    """宏观指纹（原项目第 3 阶段）：每次模拟的人数均值、SD、ACF1–3。Ns：(R, T)。"""
    return np.column_stack([Ns.mean(1), Ns.std(1)] + [S27.acf(Ns, k) for k in (1, 2, 3)])


def micro(Ns, As):
    """个体指标：D（上一轮拥挤后去的比例 − 上一轮不拥挤后去的比例；全程同一侧时无定义）与换选择比例。"""
    As = As.astype(float); crowd = Ns[:, :-1] > 60; nxt = As[:, :, 1:]
    n_c = crowd.sum(1); n_n = (~crowd).sum(1)
    go_c = (nxt * crowd[:, None, :]).sum((1, 2)) / np.maximum(n_c * As.shape[1], 1)
    go_n = (nxt * ~crowd[:, None, :]).sum((1, 2)) / np.maximum(n_n * As.shape[1], 1)
    D = np.where((n_c > 0) & (n_n > 0), go_c - go_n, np.nan)
    return D, (As[:, :, 1:] != As[:, :, :-1]).mean((1, 2))


def p2(sim, ob):
    sim = sim[~np.isnan(sim)]; n = len(sim)
    return float(min(1.0, 2 * min((1 + np.sum(sim >= ob)) / (n + 1), (1 + np.sum(sim <= ob)) / (n + 1))))


def joint(OBS, obs_D, obs_sw, fp, D, sw, cs, acf4):
    from scipy.stats import f as F_dist, multivariate_normal
    B, k = fp.shape
    mu, C = fp.mean(0), np.cov(fp.T)
    d2 = float((OBS - mu) @ np.linalg.solve(C, OBS - mu))
    Fstat = d2 * B / (B + 1) * (B - k) / (k * (B - 1))
    return dict(模拟均值=dict(zip(FP_NAMES, np.round(mu, 4).tolist())), 各指标p=dict(zip(FP_NAMES, [round(p2(fp[:, j], OBS[j]), 4) for j in range(k)])),
                马氏距离平方=round(d2, 2), 联合p=float(f"{F_dist.sf(Fstat, k, B - k):.4g}"),
                合成对数似然=round(float(multivariate_normal(mu, C, allow_singular=True).logpdf(OBS)), 2),
                D=round(float(np.nanmean(D)), 4), D的p=round(p2(D, obs_D), 4), 换选择比例=round(float(sw.mean()), 4), 换选择p=round(p2(sw, obs_sw), 4),
                锁定比例=float(np.mean((cs < .05) | (cs > .95))), ACF4_描述=dict(均值=round(float(acf4.mean()), 4), SD=round(float(acf4.std()), 4)))


def synth_diff(OBS, fa, fb, B=1000):
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(2026)
    sl = lambda fp: multivariate_normal(fp.mean(0), np.cov(fp.T), allow_singular=True).logpdf(OBS)
    boot = [sl(fa[rng.integers(0, len(fa), len(fa))]) - sl(fb[rng.integers(0, len(fb), len(fb))]) for _ in range(B)]
    return dict(差=round(float(sl(fa) - sl(fb)), 2), 区间=[round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)])


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
    B_SIM = 1000
    pops = {"HB": (X, 0.0), "HB + γ": (X.copy(), 0.0), "HB + σ": (X, E["只有σ"][1]), "HB + γ + σ": (X.copy(), s_hat)}
    pops["HB + γ"][0][:, 13] = -E["只有γ"][0]; pops["HB + γ + σ"][0][:, 13] = -g_hat
    OBS = fingerprint(S27.N_REAL[None])[0]
    OBS_D, OBS_SW = micro(S27.N_REAL[None], A[None].astype(np.int8))
    SIM = {}
    for k, (Y, sc) in pops.items():
        fps, Ds, sws, css, acf4 = [], [], [], [], []
        for b in range(4):
            Ns, As = S27.simulate(Y, B_SIM // 4, 100 + 10 * b, sc)
            fps.append(fingerprint(Ns)); d_, sw_ = micro(Ns, As); Ds.append(d_); sws.append(sw_)
            css.append((Ns >= 61).mean(1)); acf4.append(S27.acf(Ns, 4))
        SIM[k] = (np.concatenate(fps), np.concatenate(Ds), np.concatenate(sws), np.concatenate(css), np.concatenate(acf4))
        print("C", k, flush=True)
    out["C ABM 联合检验"] = {"真实": dict(指纹=dict(zip(FP_NAMES, np.round(OBS, 4).tolist())), D=round(float(OBS_D[0]), 4),
                                         换选择比例=round(float(OBS_SW[0]), 4), ACF4=round(float(S27.acf(S27.N_REAL[None], 4)[0]), 4)),
                          **{k: joint(OBS, OBS_D[0], OBS_SW[0], *v) for k, v in SIM.items()}}
    out["C 合成似然之差（HB + γ + σ 减其他；bootstrap 95% 区间）"] = {k: synth_diff(OBS, SIM["HB + γ + σ"][0], SIM[k][0])
                                                               for k in SIM if k != "HB + γ + σ"}
    Xc = pops["HB + γ + σ"][0]
    gains = {}
    for k, Y, sc in (("HB + γ + σ", Xc, s_hat), ("HB + σ", X, E["只有σ"][1])):
        Ns0, A0 = S27.simulate(Y, 100, 4, sc); base = Ns0.mean()
        p_bar = A0.astype(float).mean((0, 2)); sl = float((p_bar * (1 - p_bar)).sum())
        gains[k] = {f"Δb={db:+.1f}": round(1 - (S27.simulate(Y, 100, 4, sc, db=db)[0].mean() - base) / (sl * db), 3) for db in (-1.0, -0.5, 0.5, 1.0)}
    out["D 扰动自稳"] = gains
    PL.save(out, "s36_共同因素层.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))
