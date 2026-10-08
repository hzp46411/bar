# -*- coding: utf-8 -*-
"""“轮次”效应的形状：把线性的轮次项换成 8 个 50 轮区块（第 1 块为参照），分别估计信念权重与惯性权重的倍数（两步，HRG 个体参数固定）。"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import per_person as PP

A = L.A_REAL
rest, bp, hp, _ = PP.parts(A, "time")                    # 去掉线性轮次的全部作用
K = 8; blk = np.arange(L.T) // (L.T // K)
def nll_curve(eB_t, eH_t):
    z = rest + bp * np.exp(eB_t)[None] + hp * np.exp(eH_t)[None]
    return float((np.logaddexp(0, z) - A * z).sum())
nll = lambda p: nll_curve(p[:K][blk], p[K:][blk])        # 8 个区块各自的倍数（个体参数固定，故全部可识别）
r = minimize(nll, np.zeros(2 * K), method="L-BFGS-B")
eB = r.x[:K] - r.x[0]; eH = r.x[K:] - r.x[K]              # 报告时相对第 1 块
# 线性版本对照（同一基线）
fit = PP.FIT; sh = fit["shared"]; std = L.load_std()
tB = sh["θG_time"] + sh["θR_time"] / 2; tH = sh["θG_time"] - sh["θR_time"] / 2
mid = (np.arange(K) * (L.T // K) + L.T // K / 2) / L.T
zt = (mid - std["time"][0]) / std["time"][1]
zfull = (np.arange(L.T) / L.T - std["time"][0]) / std["time"][1]
rows = [dict(区块=f"第 {k * 50 + 1}–{(k + 1) * 50} 轮", 信念倍数=round(float(np.exp(eB[k])), 3), 惯性倍数=round(float(np.exp(eH[k])), 3),
             线性模型_信念倍数=round(float(np.exp(tB * (zt[k] - zt[0]))), 3), 线性模型_惯性倍数=round(float(np.exp(tH * (zt[k] - zt[0]))), 3)) for k in range(K)]
out = dict(说明="倍数相对第 1–50 轮（本分析的第 1 轮是实验的第 36 轮）", 区块=rows, 区块模型NLL=round(r.fun, 2),
           线性模型NLL=round(nll_curve(tB * zfull, tH * zfull), 2), 标准化=dict(均值=std["time"][0], SD=std["time"][1]))
L.save_json(out, L.OUT / "轮次形状.json")
print(json.dumps(out, ensure_ascii=False, indent=1))
