# -*- coding: utf-8 -*-
"""可靠性信号的"人内"与"人间"成分分开：M = 人内波动（减去本人均值）+ 人间差异（本人均值）。
两步估计会把人间差异也算成"仲裁"，联合估计时这部分会被个体参数（κ、b 等）吸收。真正的动态仲裁只看人内成分。"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rel_variants as RV
import arb_lib as L

rows = []
for a in (0.2, 0.4):
    defs = {"A 状态预测 − 重复成绩": RV.ewma_prev(RV.accB, a) - RV.ewma_prev(RV.wR, a),
            "C 只看信念建议成绩": RV.ewma_prev(RV.wB, a), "D 只看习惯建议成绩": RV.ewma_prev(RV.wH, a)}
    for name, Mraw in defs.items():
        Mraw = Mraw[:, 5:]                                         # 去掉起点附近
        within = Mraw - Mraw.mean(1, keepdims=True); between = np.broadcast_to(Mraw.mean(1, keepdims=True), Mraw.shape)
        for lab, M in (("人内", within / within.std()), ("人间", (between - between.mean()) / between.std())):
            Mf = np.c_[np.zeros((RV.n, 5)), M]
            r = minimize(RV.nll, [0, 0], args=(Mf,), method="L-BFGS-B")
            zz = RV.rest + RV.bp * np.exp(r.x[0] * Mf) + RV.hp * np.exp(r.x[1] * Mf)
            _, ll = L.marginal_sigma(zz, RV.A); LR = 2 * (ll - RV.ll0)
            rows.append(dict(定义=name, 速率=a, 成分=lab, θ信念=round(float(r.x[0]), 4), θ惯性=round(float(r.x[1]), 4),
                             LR含σ_df2=round(float(LR), 2), p=float(chi2.sf(max(LR, 0), 2))))
            print(rows[-1], flush=True)
L.save_json(rows, L.OUT / "可靠性_人内人间.json")
