# -*- coding: utf-8 -*-
"""第 14 步（续二）：更保守的恢复检验——如果预测与选择的学习率本来就不同（真值 = J2），加入预测还能不能帮选择的 ρ？
  s14b 的数据由 J1 生成（预测与选择共用 ρ），预测对 ρ 的帮助在那里是"按构造"存在的。
  这里用 J2 的拟合值生成数据（选择用 ρ_i、预测中的信念用 ρᴾ_i，两者按真实估计各不相同），在同一套数据上拟合：
    (a) 只用选择：HRGPR
    (b) 选择 + 预测，设定正确：J2（两个学习率）
    (c) 选择 + 预测，设定错误：J1（强行共用一个学习率）
  比较选择的 ρ、β、κ、b 的恢复，以及 (b) 中 ρᴾ 的恢复。
用法：python3 s14c_两种学习率下的恢复.py [套号]
输出：结果/s14c_两种学习率下的恢复_套<套号>.json
"""
import sys, json, importlib.util, pathlib, time
import numpy as np
from scipy.special import expit
from scipy.stats import truncnorm, spearmanr
REP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
HERE = pathlib.Path(__file__)


def load(var):
    sys.argv = [sys.argv[0], var]
    spec = importlib.util.spec_from_file_location(f"s14_{var}", str(HERE.with_name("s14_HRGPR预测联合模型.py")))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M1, M2 = load("J1"), load("J2")
PL, L, SP = M2.PL, M2.L, M2.SP
n, T = M2.n, M2.T
fit = json.loads((PL.OUT / "s14_HRGPR预测联合模型_J2.json").read_text(encoding="utf-8"))
XT = np.array(fit["X"]); PT = np.array([fit["共用"][k] for k in SP.names()])


def simulate(seed):
    rng = np.random.default_rng(seed); STD = M2.STD; G, S, N = M2.G, M2.S, M2.N
    zs = lambda m, v: (v - STD[m][0]) / STD[m][1]
    lam, thR, thG, aH = SP.unpack(PT); thBo, thHo = SP.unpack_only(PT); psi = SP.unpack_push(PT)
    pm, lag = L.public_mods(N)
    r_pub = sum((thR[m] * zs(m, pm[m]) for m in thR), np.zeros(T)); g_pub = sum((thG[m] * zs(m, pm[m]) for m in thG), np.zeros(T))
    p_pub = sum((psi[m] * zs(m, pm[m]) for m in psi), np.zeros(T))
    rho, beta, kap, b = expit(XT[:, 0]), XT[:, 1], XT[:, 2], XT[:, 3]
    m_, k_, sd, s, rf, rhoP = XT[:, 4], XT[:, 5], np.exp(0.5 * XT[:, 6]), XT[:, 7], expit(XT[:, 8]), expit(XT[:, 9])
    BL = np.full(n, 1 / 3); BH = np.full(n, 1 / 3); BLp = np.full(n, 1 / 3); BHp = np.full(n, 1 / 3)
    H = np.full(n, 0.5); c = np.zeros(n); sB = np.full(n, 0.5); sH = np.full(n, 0.5)
    A = np.zeros((n, T)); P = np.zeros((n, T))
    for t in range(T):
        eB = thBo.get("relB", 0) * zs("relB", sB); eH = thHo.get("relH", 0) * zs("relH", sH)
        V = BL - 0.7 * BH; Vp = BLp - 0.7 * BHp; wB = np.exp(g_pub[t] + r_pub[t] / 2 + eB)
        Bb = m_ + k_ * Vp; Bv = Bb + sd * rng.standard_normal(n); Kc = (Bv <= 60.5).astype(float)
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
        BL += rho * (G[:, t] - BL); BH += rho * (S[:, t] - BH)
        BLp += rhoP * (G[:, t] - BLp); BHp += rhoP * (S[:, t] - BHp)
        H += aH * (a - H); c = 2 * H - 1
    return A, P


cls = lambda r: np.digitize(r, [0.1, 0.9])


def summ(Xe, phie, lab):
    rt, re = expit(XT[:, 0]), expit(Xe[:, 0])
    out = dict(ρ_Spearman=float(spearmanr(rt, re)[0]), ρ_logit相关=float(np.corrcoef(XT[:, 0], Xe[:, 0])[0, 1]),
               ρ_类别判对比例=float(np.mean(cls(rt) == cls(re))),
               β_相关=float(np.corrcoef(XT[:, 1], Xe[:, 1])[0, 1]), β_符号判对=float(np.mean(np.sign(XT[:, 1]) == np.sign(Xe[:, 1]))),
               κ_相关=float(np.corrcoef(XT[:, 2], Xe[:, 2])[0, 1]), b_相关=float(np.corrcoef(XT[:, 3], Xe[:, 3])[0, 1]),
               共用估计=dict(zip(SP.names(), np.round(phie, 4).tolist())))
    if Xe.shape[1] > 9:
        rpt, rpe = expit(XT[:, 9]), expit(Xe[:, 9])
        out.update(ρᴾ_Spearman=float(spearmanr(rpt, rpe)[0]), ρᴾ_类别判对比例=float(np.mean(cls(rpt) == cls(rpe))))
    print(lab, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if k != "共用估计"}, flush=True)
    return out


res = {"真值": "J2", "真值共用": dict(zip(SP.names(), PT.tolist())), "套": []}
t1 = time.time()
As, Ps = simulate(9100 + REP)
data = (As, Ps, np.ones_like(As, bool), M2.G, M2.S)
row = {}
Xa, pa, fa, ha = L.joint_fit(SP, As, M2.G, M2.S, M2.N, M2.X4.copy(), M2.PHI4.copy(), max_rounds=8, tol=0.5, rand_starts=2, log=lambda s_: None)
row["只用选择（HRGPR）"] = summ(Xa, pa, f"第 {REP + 1} 套 只用选择")
X0 = M2.init_X(M2.X4.copy(), data, rho_f=np.full(n, 0.3))
X0b = np.column_stack([X0, X0[:, 0]])
Xb, pb, hb = M2.estimate(X0b, M2.PHI4.copy(), data, 12, log=lambda s_: None)
row["选择 + 预测，两个学习率（J2，设定正确）"] = summ(Xb, pb, f"第 {REP + 1} 套 J2")
Xc, pc, hc = M1.estimate(X0.copy(), M1.PHI4.copy(), data, 12, log=lambda s_: None)
row["选择 + 预测，共用学习率（J1，设定错误）"] = summ(Xc, pc, f"第 {REP + 1} 套 J1")
row["用时秒"] = time.time() - t1
res["套"].append(row)
PL.save(res, f"s14c_两种学习率下的恢复_套{REP}.json")
print("完成")
