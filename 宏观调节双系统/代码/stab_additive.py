# -*- coding: utf-8 -*-
"""稳定性：乘法"惯性权重"写法 vs 加法"重复推力"写法（两步，HRG 去掉稳定性作用为基线）。
  乘法：惯性部分 × e^{δ·M}（κ<0 的交替型，δ<0 表示交替减弱 = 更多重复；与重复型方向相反，群体平均可能抵消）
  加法：+ ψ·c·M（c = 习惯痕迹方向；ψ>0 = 稳定时人人都更倾向重复自己的习惯，不论 κ 正负）"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import per_person as PP
from scipy.special import expit

A = L.A_REAL
fit = PP.FIT; aH = float(expit(fit["shared"]["logit_aH"]))
H = np.full(len(A), .5); c = np.zeros(A.shape)
for t in range(L.T):
    c[:, t] = 2 * H - 1; H += aH * (A[:, t] - H)
out = {}
for m in ("stab", "dev", "rel"):
    rest, bp, hp, M = PP.parts(A, m)
    nll = lambda z: float((np.logaddexp(0, z) - A * z).sum())
    f0 = nll(rest + bp + hp); s0, ll0 = L.marginal_sigma(rest + bp + hp, A)
    mult = minimize(lambda d: nll(rest + bp * np.exp(d[0] * M) + hp * np.exp(d[1] * M)), [0, 0], method="L-BFGS-B")
    add = minimize(lambda d: nll(rest + bp * np.exp(d[0] * M) + hp + d[1] * c * M), [0, 0], method="L-BFGS-B")
    both = minimize(lambda d: nll(rest + bp * np.exp(d[0] * M) + hp * np.exp(d[1] * M) + d[2] * c * M), [0, 0, 0], method="L-BFGS-B")
    zz = lambda f, d: f(d)
    _, llm = L.marginal_sigma(rest + bp * np.exp(mult.x[0] * M) + hp * np.exp(mult.x[1] * M), A)
    _, lla = L.marginal_sigma(rest + bp * np.exp(add.x[0] * M) + hp + add.x[1] * c * M, A)
    _, llb = L.marginal_sigma(rest + bp * np.exp(both.x[0] * M) + hp * np.exp(both.x[1] * M) + both.x[2] * c * M, A)
    out[L.LABEL[m]] = dict(乘法_δ惯性=round(float(mult.x[1]), 4), 乘法_LR含σ_df2=round(2 * (llm - ll0), 2),
                         加法_ψ=round(float(add.x[1]), 4), 加法_LR含σ_df2=round(2 * (lla - ll0), 2),
                         两者_δ惯性=round(float(both.x[1]), 4), 两者_ψ=round(float(both.x[2]), 4),
                         加法相对乘法再改善_LR=round(2 * (llb - llm), 2), 乘法相对加法再改善_LR=round(2 * (llb - lla), 2))
L.save_json(out, L.OUT / "稳定性_乘法vs加法.json")
print(json.dumps(out, ensure_ascii=False, indent=1))
