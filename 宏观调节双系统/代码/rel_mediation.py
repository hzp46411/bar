# -*- coding: utf-8 -*-
"""可靠性是不是宏观调节的内部中介？（两步：HRGP 个体参数固定）
(1) 可靠性信号与宏观调节变量的相关
(2) 三个模型比较（都含轮次）：只用宏观变量（稳定、偏离 + 稳定推力、偏离推力）/ 只用可靠性（信念建议成绩 → 两系统、习惯建议成绩 → 两系统）/ 两者都用"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import rel_variants as RV

A = L.A_REAL; fit = RV.fit; sp = RV.sp; X = RV.X
sh0 = {k: (v if k in ("lam", "logit_aH") else 0.0) for k, v in fit["shared"].items()}
phi0 = np.array([sh0[k] for k in sp.names()])
run = lambda Xv: L.run(Xv, sp, phi0, A, L.G_REAL, L.S_REAL, L.ATT, out="z")
z = run(X); Xb = X.copy(); Xb[:, 1] = 0; Xk = X.copy(); Xk[:, 2] = 0
bp = z - run(Xb); hp = z - run(Xk); rest = z - bp - hp
aH = float(expit(fit["shared"]["logit_aH"])); H = np.full(A.shape[0], .5); c = np.zeros(A.shape)
for t in range(L.T):
    c[:, t] = 2 * H - 1; H += aH * (A[:, t] - H)
std = L.load_std(); pm, _ = L.public_mods(L.ATT)
Z = lambda v: (v - v.mean()) / v.std()
macro = {m: np.broadcast_to(Z(pm[m]), A.shape) for m in ("stab", "dev", "time")}
relB = Z(RV.ewma_prev(RV.wB, 0.2)); relH = Z(RV.ewma_prev(RV.wH, 0.2))
cor = {f"{a}~{b}": round(float(np.corrcoef(x.ravel(), y.ravel())[0, 1]), 3)
       for a, x in (("信念建议成绩", relB), ("习惯建议成绩", relH)) for b, y in (("稳定", macro["stab"]), ("偏离", macro["dev"]), ("轮次", macro["time"]))}
cor["信念建议成绩~习惯建议成绩"] = round(float(np.corrcoef(relB.ravel(), relH.ravel())[0, 1]), 3)

def make(terms_w, terms_push):
    """terms_w：进入两个权重的变量列表（各有 θB、θH）；terms_push：加法推力变量列表"""
    k = 2 * len(terms_w) + len(terms_push)
    def zfun(p):
        eB = sum(p[2 * j] * M for j, M in enumerate(terms_w)) if terms_w else 0
        eH = sum(p[2 * j + 1] * M for j, M in enumerate(terms_w)) if terms_w else 0
        ps = sum(p[2 * len(terms_w) + j] * M for j, M in enumerate(terms_push)) if terms_push else 0
        return rest + bp * np.exp(eB) + hp * np.exp(eH) + ps * c
    return k, zfun
models = {"只有轮次": ([macro["time"]], []),
          "宏观变量（稳定、偏离、轮次 + 推力）": ([macro["stab"], macro["dev"], macro["time"]], [macro["stab"], macro["dev"]]),
          "可靠性（信念成绩、习惯成绩、轮次）": ([relB, relH, macro["time"]], []),
          "可靠性 + 推力": ([relB, relH, macro["time"]], [macro["stab"], macro["dev"]]),
          "两者都用": ([macro["stab"], macro["dev"], relB, relH, macro["time"]], [macro["stab"], macro["dev"]])}
res = {}
for name, (tw, tp) in models.items():
    k, zf = make(tw, tp)
    r = minimize(lambda p: float((np.logaddexp(0, zf(p)) - A * zf(p)).sum()), np.zeros(k), method="L-BFGS-B")
    _, ll = L.marginal_sigma(zf(r.x), A)
    res[name] = dict(k=k, 含σ对数似然=round(ll, 2), 参数=np.round(r.x, 4).tolist())
    print(name, res[name], flush=True)
base = res["只有轮次"]["含σ对数似然"]
for name in res:
    res[name]["相对只有轮次_LR"] = round(2 * (res[name]["含σ对数似然"] - base), 2)
out = dict(相关=cor, 模型=res,
           宏观之上加可靠性_LR=round(2 * (res["两者都用"]["含σ对数似然"] - res["宏观变量（稳定、偏离、轮次 + 推力）"]["含σ对数似然"]), 2),
           可靠性加推力之上再加宏观权重_LR=round(2 * (res["两者都用"]["含σ对数似然"] - res["可靠性 + 推力"]["含σ对数似然"]), 2))
L.save_json(out, L.OUT / "可靠性_中介.json")
print(json.dumps({k: v for k, v in out.items() if k != "模型"}, ensure_ascii=False, indent=1))
