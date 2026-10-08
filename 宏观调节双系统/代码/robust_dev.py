# -*- coding: utf-8 -*-
"""补充：偏离对习惯的两种作用（乘法：惯性权重；加法：推离习惯 ψ_dev）合并检验，在线性与样条分级反应下各做一次。"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as R          # 复用 robust.py 中的数据与函数（导入时会重跑一遍，约数分钟）
import arb_lib as L

rest, bp, hp = R.parts(["θR_dev", "θG_dev", "ψ_dev"])
rest0 = rest - R.SH["lam"] * R.LAG[None]
M, c = R.Mdev, R.c
res = {}
for lab, k_l, lamf in (("线性", 1, lambda q: (q[0] * R.LAG)[None]), ("样条", 5, lambda q: (q @ R.basis)[None])):
    def full(p):  # p = [δB, δH, ψ_dev, λ...]
        return rest0 + lamf(p[3:]) + bp * np.exp(p[0] * M) + hp * np.exp(p[1] * M) + p[2] * c * M
    x0 = np.r_[0, 0, 0, R.SH["lam"], np.zeros(k_l - 1)]
    xf, llf = R.fit(full, 3 + k_l, x0)
    xn, lln = R.fit(lambda p: full(np.r_[p[0], 0, 0, p[1:]]), 1 + k_l, np.r_[0, x0[3:]])           # 去掉偏离对习惯的两种作用
    xm, llm = R.fit(lambda p: full(np.r_[p[0], p[1], 0, p[2:]]), 2 + k_l, np.r_[0, 0, x0[3:]])      # 只留乘法
    xa, lla = R.fit(lambda p: full(np.r_[p[0], 0, p[1], p[2:]]), 2 + k_l, np.r_[0, 0, x0[3:]])      # 只留加法
    res[lab] = dict(δ信念=round(xf[0], 4), δ惯性=round(xf[1], 4), ψ_dev=round(xf[2], 4),
                    偏离对习惯合并_LR_df2=round(2 * (llf - lln), 2), p=float(chi2.sf(max(2 * (llf - lln), 0), 2)),
                    只留乘法_LR_df1=round(2 * (llm - lln), 2), 只留加法_LR_df1=round(2 * (lla - lln), 2))
    print(lab, res[lab], flush=True)
L.save_json(res, L.OUT / "稳健性_偏离合并.json")
