# -*- coding: utf-8 -*-
"""第 10 步（续）：合理化联合模型的参数恢复与假阳性检验
  1 恢复：用 ρ逐人 的拟合值生成数据（习惯痕迹随模拟的选择更新；预测取整、截在 30–90），重新拟合 ρ逐人，
          看 ρ_i、s_i 与共用参数能否恢复（逐人相关、中位数）。
  2 假阳性：用 ρ0（如实报告）的拟合值生成数据，再用 ρ逐人 拟合——如实报告的数据里会"找出"多少合理化？
          这是"合理化是不是设定误差的产物"的直接检验。
  每种 2 套。
用法：python3 s10b_合理化模型恢复.py
输出：结果/s10b_合理化模型恢复.json
"""
import sys, json
import numpy as np
from scipy.special import expit
from scipy.stats import truncnorm
import importlib.util, pathlib


def load(var):
    sys.argv = [sys.argv[0], var]
    spec = importlib.util.spec_from_file_location(f"s10_{var}", str(pathlib.Path(__file__).with_name("s10_合理化联合模型.py")))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


M_RI = load("ρ逐人")
PL = M_RI.PL
f4 = PL.fit("HRGPR"); aH = float(expit(f4[0]["shared"]["logit_aH"]))
n, Tm = M_RI.Ya.shape


def simulate(X, sh, rho_i, seed):
    rng = np.random.default_rng(seed); lr, dlt, psi, lam = sh
    Ya = np.zeros((n, Tm)); Yp = np.zeros((n, Tm)); Cc = np.zeros((n, Tm)); H = np.full(n, 0.5)
    for t in range(M_RI.t0):
        H += aH * (PL.A[:, t] - H)
    mu, ws, wm, logv, b, s, kap = X[:, :7].T; sd = np.exp(0.5 * logv)
    for t in range(Tm):
        c = 2 * H - 1; Cc[:, t] = c
        Bbar = mu + ws * M_RI.SG[t] + wm * M_RI.DM[t]
        B = Bbar + sd * rng.standard_normal(n); K = B <= 60.5
        z = b + s * (K - 0.5) + kap * c * np.exp(dlt * M_RI.DEV[t]) + psi * M_RI.STAB[t] * c + lam * M_RI.LAG[t]
        a = rng.random(n) < expit(z)
        rep = B.copy()
        bad = (K != a) & (rng.random(n) < rho_i)                     # 不一致且合理化：从一致一侧的信念分布中抽一个
        for i in np.where(bad)[0]:
            lo, hi = ((-np.inf, (60.5 - Bbar[i]) / sd[i]) if a[i] else ((60.5 - Bbar[i]) / sd[i], np.inf))
            rep[i] = Bbar[i] + sd[i] * truncnorm.rvs(lo, hi, random_state=rng)
        Ya[:, t] = a; Yp[:, t] = np.clip(np.round(rep), 30, 90)
        H += aH * (a - H)
    return Ya, Yp, Cc


def refit(m, data, iters=15):
    X = m.init_X(data); sh = np.array([0.0, 0.0, 0.2, -0.25])
    X, sh, hist = m.estimate(X, sh, iters, log=lambda s_: None, data=data, tol=0.5)
    return X, sh, hist[-1]


fit_RI = json.loads((PL.OUT / "s10_合理化联合模型_ρ逐人.json").read_text(encoding="utf-8"))
fit_H = json.loads((PL.OUT / "s10_合理化联合模型_ρ0.json").read_text(encoding="utf-8"))
out = {"恢复（真值 = ρ逐人 拟合）": [], "假阳性（真值 = 如实报告 ρ0 拟合）": []}
cases = [("恢复（真值 = ρ逐人 拟合）", fit_RI, True), ("假阳性（真值 = 如实报告 ρ0 拟合）", fit_H, False)]
for lab, fit, ri in cases:
    X0 = np.array(fit["X"]); sh0 = np.array([fit["共用"][k] for k in M_RI.SH])
    rho_true = expit(X0[:, 7]) if ri else np.zeros(n)
    for rep in range(2):
        Ya, Yp, Cc = simulate(X0, sh0, rho_true, 2024 + 31 * rep + (0 if ri else 500))
        data = (Ya, Yp, Cc, np.ones_like(Ya, bool))
        cons = float(np.mean(Ya == (Yp <= 60)))
        Xe, she, f = refit(M_RI, data)
        rho_e = expit(Xe[:, 7])
        row = dict(套=rep, 模拟数据一致率=cons, 共用估计=dict(zip(M_RI.SH, np.round(she, 4).tolist())),
                   ρ_i估计_分位数=np.round(np.percentile(rho_e, [10, 25, 50, 75, 90]), 3).tolist(),
                   ρ_i估计大于05的人数=int((rho_e > .5).sum()), s_i估计中位数=float(np.median(Xe[:, 5])), s_i真值中位数=float(np.median(X0[:, 5])))
        if ri:
            row.update(ρ_i相关=float(np.corrcoef(rho_true, rho_e)[0, 1]), s_i相关=float(np.corrcoef(X0[:, 5], Xe[:, 5])[0, 1]),
                       ρ_i真值大于05的人数=int((rho_true > .5).sum()), κ_i相关=float(np.corrcoef(X0[:, 6], Xe[:, 6])[0, 1]))
        out[lab].append(row); print(lab, row, flush=True)
        PL.save(out, "s10b_合理化模型恢复.json")
print("完成")
