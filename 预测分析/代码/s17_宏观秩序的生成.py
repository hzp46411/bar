# -*- coding: utf-8 -*-
"""第 17 步：哪些内部模型生成宏观协调？——闭环模拟中的"成分拆分"与"扰动自稳"
  问题（Arthur 的第二问）：人与人的内部模型各不相同，为什么群体反而形成稳定的秩序（人数钉在容量附近）？
  做法：用第 4 层主模型 HRGPR（每人 ρ、β、κ、b；共用 λ、仲裁项、ψ、α_H；同轮共同冲击 σ）做闭环模拟（人数由模拟的选择产生）。
  A 成分拆分（每种 300 次 × 400 轮）：完整人群；同质人群（全体取中位数参数）；打乱人群（各参数列独立置换）；
     全体外推（β = +|β|）；全体反向（β = −|β|）；去信念（β = 0）；去习惯（κ = 0、ψ = 0）；去 λ；去共同冲击（σ = 0）；偏好统一（b = 中位数）
  B 扰动自稳：给所有人的偏好 b 加同一个 Δb（−1、−0.5、0、+0.5、+1），看平均人数偏离容量多少。
     没有反馈时，Δb 会按 Σp(1−p)·Δb 平移人数；反馈越强，平移越小——"自稳增益" = 1 − 实际平移 / 无反馈平移。
     对 完整、去信念、去 λ、去信念且去 λ、去习惯 分别计算（每格 150 次）。
  指标：平均人数、|平均 − 60|、SD、ACF1、挤的轮次比例、效率（每人每轮平均得分：去且不挤 1，不去且挤 0.7）、换人率、个人去的比例的离散度、常客（去的比例 > 0.8）人数
输出：结果/s17_宏观秩序的生成.json
"""
import json, sys
import numpy as np
from scipy.special import expit
import pred_lib as PL
L = PL.L; STD = L.load_std()
f, SP, X0, PHI0 = PL.fit("HRGPR"); SIG0 = f["sigma"]
T, n = PL.T, X0.shape[0]
NAMES = SP.names()
B_A = int(sys.argv[1]) if len(sys.argv) > 1 else 300
B_B = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def simulate(X, phi, sig, seed, no_habit=False, no_lam=False):
    lam, thR, thG, aH = SP.unpack(phi); thBo, thHo = SP.unpack_only(phi); psi = SP.unpack_push(phi)
    if no_lam: lam = 0.0
    if no_habit: psi = {m: 0.0 for m in psi}
    zs = lambda m, v: (v - STD[m][0]) / STD[m][1]
    rng = np.random.default_rng(seed)
    rho, beta, kap, b = expit(X[:, 0]), X[:, 1], (0 * X[:, 2] if no_habit else X[:, 2]), X[:, 3]
    BL = np.full(n, 1 / 3); BH = np.full(n, 1 / 3); H = np.full(n, .5); c = np.zeros(n)
    sB = np.full(n, .5); sH = np.full(n, .5); Ns = np.zeros(T); A = np.zeros((n, T)); stab = 0.0
    for t in range(T):
        lag = (Ns[t - 1] - 60) / 10 if t > 0 else 0.0
        if t >= 2: stab = stab + 1 if (Ns[t - 1] > 60) == (Ns[t - 2] > 60) else 0.0
        mods = dict(stab=min(stab, 4), dev=abs(lag), time=t / T)
        rr = sum(thR[m] * zs(m, mods[m]) for m in thR); g = sum(thG[m] * zs(m, mods[m]) for m in thG)
        push = sum(psi[m] * zs(m, mods[m]) for m in psi)
        eB = thBo.get("relB", 0) * zs("relB", sB); eH = thHo.get("relH", 0) * zs("relH", sH)
        V = BL - 0.7 * BH
        z = b + lam * lag + np.exp(g) * (beta * V * np.exp(rr / 2 + eB) + kap * c * np.exp(-rr / 2 + eH)) + push * c + sig * rng.standard_normal()
        recB = (beta * V > 0).astype(float); recH = (c > 0).astype(float)
        a = (rng.random(n) < expit(z)).astype(float); A[:, t] = a; Ns[t] = a.sum()
        oth = Ns[t] - a; Gt = (oth <= 59).astype(float); St = (oth >= 61).astype(float)
        sB = (1 - L.REL_RATE) * sB + L.REL_RATE * (recB * Gt + (1 - recB) * St)
        sH = (1 - L.REL_RATE) * sH + L.REL_RATE * (recH * Gt + (1 - recH) * St)
        BL += rho * (Gt - BL); BH += rho * (St - BH); H += aH * (a - H); c = 2 * H - 1
    return Ns, A


def acf1(x):
    x = x - x.mean(); return float(x[1:] @ x[:-1] / (x @ x))


def metrics(Ns, A):
    crowd = Ns > 60
    pay = np.where(A == 1, (~crowd)[None] * 1.0, crowd[None] * 0.7)
    gr = A.mean(1)
    return dict(均值=float(Ns.mean()), 偏离容量=float(abs(Ns.mean() - 60)), SD=float(Ns.std()), ACF1=acf1(Ns), 挤的比例=float(crowd.mean()),
                效率=float(pay.mean()), 换人率=float((A[:, 1:] != A[:, :-1]).mean()), 去的比例离散度=float(gr.std()), 常客人数=float((gr > 0.8).sum()))


def run_many(X, phi, sig, B, seed0, **kw):
    ms = [metrics(*simulate(X, phi, sig, seed0 + k, **kw)) for k in range(B)]
    return {k: dict(均值=float(np.mean([m[k] for m in ms])), SD=float(np.std([m[k] for m in ms]))) for k in ms[0]}


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    out = {"真实": metrics(PL.N.astype(float), PL.A)}
    print("真实", {k: round(v, 3) for k, v in out["真实"].items()}, flush=True)
    med = np.median(X0, 0)
    Xh = np.tile(med, (n, 1))
    Xs = X0.copy()
    for j in range(4): Xs[:, j] = rng.permutation(Xs[:, j])
    Xp = X0.copy(); Xp[:, 1] = np.abs(X0[:, 1]); Xm = X0.copy(); Xm[:, 1] = -np.abs(X0[:, 1])
    X_nb = X0.copy(); X_nb[:, 1] = 0.0
    X_b = X0.copy(); X_b[:, 3] = med[3]
    phi_nl = PHI0.copy()
    scen = {"完整人群": (X0, PHI0, SIG0, {}), "同质人群（中位数参数）": (Xh, PHI0, SIG0, {}), "打乱人群（参数独立置换）": (Xs, PHI0, SIG0, {}),
            "全体外推（β = +|β|）": (Xp, PHI0, SIG0, {}), "全体反向（β = −|β|）": (Xm, PHI0, SIG0, {}), "去信念（β = 0）": (X_nb, PHI0, SIG0, {}),
            "去习惯（κ = 0，ψ = 0）": (X0, PHI0, SIG0, dict(no_habit=True)), "去 λ": (X0, PHI0, SIG0, dict(no_lam=True)),
            "去共同冲击（σ = 0）": (X0, PHI0, 0.0, {}), "偏好统一（b = 中位数）": (X_b, PHI0, SIG0, {})}
    out["A 成分拆分"] = {}
    for k_, (X, phi, sig, kw) in scen.items():
        r = run_many(X, phi, sig, B_A, 100000, **kw); out["A 成分拆分"][k_] = r
        print(k_, {m: round(v["均值"], 3) for m, v in r.items()}, flush=True)
        PL.save(out, "s17_宏观秩序的生成.json")
    # B 扰动自稳
    p_hat = PL.A.mean(1); base_shift = float(np.sum(p_hat * (1 - p_hat)))        # 无反馈时 Δb = 1 平移的人数（一阶）
    out["B 扰动自稳"] = {"无反馈的平移（每单位 Δb，人）": base_shift, "格": {}}
    conds = {"完整": (X0, {}), "去信念": (X_nb, {}), "去 λ": (X0, dict(no_lam=True)), "去信念且去 λ": (X_nb, dict(no_lam=True)), "去习惯": (X0, dict(no_habit=True))}
    for cname, (X, kw) in conds.items():
        row = {}
        for db in (-1.0, -0.5, 0.0, 0.5, 1.0):
            Xd = X.copy(); Xd[:, 3] = Xd[:, 3] + db
            ms = [metrics(*simulate(Xd, PHI0, SIG0, 200000 + k, **kw)) for k in range(B_B)]
            row[str(db)] = dict(均值=float(np.mean([m["均值"] for m in ms])), 效率=float(np.mean([m["效率"] for m in ms])), SD=float(np.mean([m["SD"] for m in ms])))
        xs = np.array([-1, -.5, 0, .5, 1.]); ys = np.array([row[str(v)]["均值"] for v in xs])
        slope = float(np.polyfit(xs, ys, 1)[0])
        out["B 扰动自稳"]["格"][cname] = dict(各Δb=row, 平移斜率=slope, 自稳增益=1 - slope / base_shift)
        print(cname, "平移斜率", round(slope, 2), "自稳增益", round(1 - slope / base_shift, 3), flush=True)
        PL.save(out, "s17_宏观秩序的生成.json")
    print("完成")
