# -*- coding: utf-8 -*-
"""大偏离削弱的是"所有人"的惯性，还是"上一轮输了的人"的惯性？（两步：HRG 去掉偏离的作用后，只估计偏离相关的共用参数）
  全员：惯性部分 × e^{δ·M}            只对输者：× e^{δ·M·输}           只对赢者：× e^{δ·M·赢}         分开：× e^{(δ输·输 + δ赢·赢)·M}
  M = 标准化的上一轮偏离幅度；输 = 上一轮按自己的选择没有得分（去了却挤 / 没去却不挤）。信念部分都保留 e^{δB·M}。"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import per_person as PP

A = L.A_REAL
rest, bp, hp, M = PP.parts(A, "dev")
win = A * L.G_REAL + (1 - A) * L.S_REAL
lost_prev = np.c_[np.zeros((len(A), 1)), 1 - win[:, :-1]]; won_prev = np.c_[np.zeros((len(A), 1)), win[:, :-1]]
lost_prev[:, 0] = 0
def nll(z): return float((np.logaddexp(0, z) - A * z).sum())
specs = {"无偏离作用": lambda d: rest + bp + hp,
         "全员": lambda d: rest + bp * np.exp(d[0] * M) + hp * np.exp(d[1] * M),
         "只对输者": lambda d: rest + bp * np.exp(d[0] * M) + hp * np.exp(d[1] * M * lost_prev),
         "只对赢者": lambda d: rest + bp * np.exp(d[0] * M) + hp * np.exp(d[1] * M * won_prev),
         "输赢分开": lambda d: rest + bp * np.exp(d[0] * M) + hp * np.exp((d[1] * lost_prev + d[2] * won_prev) * M)}
k = {"无偏离作用": 0, "全员": 2, "只对输者": 2, "只对赢者": 2, "输赢分开": 3}
res = {}
for name, f in specs.items():
    if k[name] == 0:
        res[name] = dict(NLL=nll(f(None)), k=0); continue
    r = minimize(lambda d: nll(f(d)), np.zeros(k[name]), method="L-BFGS-B", bounds=[(-3, 3)] * k[name])
    s, lls = L.marginal_sigma(f(r.x), A)
    res[name] = dict(NLL=round(r.fun, 3), k=k[name], 参数=np.round(r.x, 4).tolist(), 含σ对数似然=round(lls, 3))
s0, ll0 = L.marginal_sigma(specs["无偏离作用"](None), A); res["无偏离作用"]["含σ对数似然"] = round(ll0, 3)
for name in ("全员", "只对输者", "只对赢者", "输赢分开"):
    res[name]["vs无偏离_LR含σ"] = round(2 * (res[name]["含σ对数似然"] - ll0), 2)
res["输赢分开_vs_全员_LR含σ"] = round(2 * (res["输赢分开"]["含σ对数似然"] - res["全员"]["含σ对数似然"]), 2)
res["输赢分开_vs_只对输者_LR含σ"] = round(2 * (res["输赢分开"]["含σ对数似然"] - res["只对输者"]["含σ对数似然"]), 2)
res["说明"] = "参数顺序：[δ信念, δ惯性(或 δ惯性_输), (δ惯性_赢)]；每 1 SD 偏离"
L.save_json(res, L.OUT / "偏离_输赢.json")
print(json.dumps(res, ensure_ascii=False, indent=1))
