# -*- coding: utf-8 -*-
"""第 14 步（续）：加入预测之后，BBL 的学习率 ρ 与权重 β 是否更好识别？
  用 J1 的拟合值生成数据（G、S 取真实值；选择、习惯痕迹、预测按模型生成；预测取整、截在 30–90）。
  在同一套模拟数据上：
    (a) 只用选择：HRGPR（第 4 层的估计程序 joint_fit，多起点）
    (b) 选择 + 预测：J1（s14 的估计程序）
  两者从同一起点（真实数据的 HRGPR 估计）出发；比较 ρ、β、κ、b 的恢复（逐人相关、ρ 的类别是否判对）与共用的仲裁参数。
用法：python3 s14b_联合模型恢复.py [套号]（每套一个进程，可并行）
输出：结果/s14b_联合模型恢复_套<套号>.json
"""
import sys, json, importlib.util, pathlib, time
import numpy as np
from scipy.special import expit
from scipy.stats import truncnorm, spearmanr
REP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
sys.argv = [sys.argv[0], "J1"]
spec = importlib.util.spec_from_file_location("s14", str(pathlib.Path(__file__).with_name("s14_HRGPR预测联合模型.py")))
s14 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s14)
PL, L, SP = s14.PL, s14.L, s14.SP
n, T = s14.n, s14.T
fit = json.loads((PL.OUT / "s14_HRGPR预测联合模型_J1.json").read_text(encoding="utf-8"))
XT = np.array(fit["X"]); PT = np.array([fit["共用"][k] for k in SP.names()])


def simulate(seed):
    rng = np.random.default_rng(seed); STD = s14.STD; G, S, N = s14.G, s14.S, s14.N
    zs = lambda m, v: (v - STD[m][0]) / STD[m][1]
    lam, thR, thG, aH = SP.unpack(PT); thBo, thHo = SP.unpack_only(PT); psi = SP.unpack_push(PT)
    pm, lag = L.public_mods(N)
    r_pub = sum((thR[m] * zs(m, pm[m]) for m in thR), np.zeros(T)); g_pub = sum((thG[m] * zs(m, pm[m]) for m in thG), np.zeros(T))
    p_pub = sum((psi[m] * zs(m, pm[m]) for m in psi), np.zeros(T))
    rho, beta, kap, b = expit(XT[:, 0]), XT[:, 1], XT[:, 2], XT[:, 3]
    m_, k_, sd, s, rf = XT[:, 4], XT[:, 5], np.exp(0.5 * XT[:, 6]), XT[:, 7], expit(XT[:, 8])
    BL = np.full(n, 1 / 3); BH = np.full(n, 1 / 3); H = np.full(n, 0.5); c = np.zeros(n); sB = np.full(n, 0.5); sH = np.full(n, 0.5)
    A = np.zeros((n, T)); P = np.zeros((n, T))
    for t in range(T):
        eB = thBo.get("relB", 0) * zs("relB", sB); eH = thHo.get("relH", 0) * zs("relH", sH)
        V = BL - 0.7 * BH; wB = np.exp(g_pub[t] + r_pub[t] / 2 + eB)
        Bb = m_ + k_ * V; Bv = Bb + sd * rng.standard_normal(n); Kc = (Bv <= 60.5).astype(float)
        z = b + lam * lag[t] + wB * (beta * V + s * (Kc - 0.5)) + np.exp(g_pub[t] - r_pub[t] / 2 + eH) * kap * c + p_pub[t] * c
        a = (rng.random(n) < expit(z)).astype(float)
        rep = Bv.copy()
        for i in np.where((Kc != a) & (rng.random(n) < rf))[0]:
            lo, hi = ((-np.inf, (60.5 - Bb[i]) / sd[i]) if a[i] else ((60.5 - Bb[i]) / sd[i], np.inf))
            rep[i] = Bb[i] + sd[i] * truncnorm.rvs(lo, hi, random_state=rng)
        A[:, t] = a; P[:, t] = np.clip(np.round(rep), 30, 90)
        recB = (beta * V > 0).astype(float); recH = (c > 0).astype(float)
        sB = (1 - L.REL_RATE) * sB + L.REL_RATE * (recB * G[:, t] + (1 - recB) * S[:, t])
        sH = (1 - L.REL_RATE) * sH + L.REL_RATE * (recH * G[:, t] + (1 - recH) * S[:, t])
        BL += rho * (G[:, t] - BL); BH += rho * (S[:, t] - BH); H += aH * (a - H); c = 2 * H - 1
    return A, P


def summ(Xe, phie, lab):
    rt, re = expit(XT[:, 0]), expit(Xe[:, 0])
    cls = lambda r: np.digitize(r, [0.1, 0.9])                               # 慢（<0.1）/ 中 / 快（>0.9）
    out = dict(ρ_Spearman=float(spearmanr(rt, re)[0]), ρ_logit相关=float(np.corrcoef(XT[:, 0], Xe[:, 0])[0, 1]),
               ρ_类别判对比例=float(np.mean(cls(rt) == cls(re))),
               β_相关=float(np.corrcoef(XT[:, 1], Xe[:, 1])[0, 1]), β_符号判对=float(np.mean(np.sign(XT[:, 1]) == np.sign(Xe[:, 1]))),
               κ_相关=float(np.corrcoef(XT[:, 2], Xe[:, 2])[0, 1]), b_相关=float(np.corrcoef(XT[:, 3], Xe[:, 3])[0, 1]),
               共用估计=dict(zip(SP.names(), np.round(phie, 4).tolist())))
    print(lab, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k != "共用估计"}, flush=True)
    return out


res = {"真值共用": dict(zip(SP.names(), PT.tolist())), "套": []}
for rep in (REP,):
    t1 = time.time()
    As, Ps = simulate(7000 + rep)
    data = (As, Ps, np.ones_like(As, bool), s14.G, s14.S)
    # (a) 只用选择
    Xa, pa, fa, ha = L.joint_fit(SP, As, s14.G, s14.S, s14.N, s14.X4.copy(), s14.PHI4.copy(), max_rounds=8, tol=0.5, rand_starts=2, log=lambda s_: None)
    ra = summ(Xa, pa, f"第 {rep + 1} 套 只用选择")
    # (b) 选择 + 预测
    X0 = s14.init_X(s14.X4.copy(), data, rho_f=np.full(n, 0.3))
    Xb, pb, hb = s14.estimate(X0, s14.PHI4.copy(), data, 12, log=lambda s_: None)
    rb = summ(Xb, pb, f"第 {rep + 1} 套 选择 + 预测")
    res["套"].append({"只用选择（HRGPR）": ra, "选择 + 预测（J1）": rb, "用时秒": time.time() - t1})
    PL.save(res, f"s14b_联合模型恢复_套{REP}.json")
print("完成")
