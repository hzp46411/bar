# -*- coding: utf-8 -*-
"""
第三环引出的微观检验：稳定调节的形状——"局面刚翻转"（意外 / 变点）还是"稳定持续了多久"（线性累积）？
  真实数据中换人率：翻转后那一轮 46%，之后 32%、30%、30%、28%——像一个台阶而不是一条斜线。
  若调节信号是"翻转"，线性写法会在长期稳定时把人推得过度重复 → 闭环里长期稳定后的回调（φs）被削弱。
两步（HRGPR 个体参数固定；含 σ 的边际似然），把稳定的三条作用（信念权重、惯性权重、习惯方向推力）换成不同编码：
  线性（原写法）· 翻转（台阶）· 两者 · 分类（稳定 0/1/2/3/4 各一个水平，非参数）
输出：结果/翻转检验.json
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

A, G, S, N = L.A_REAL, L.G_REAL, L.S_REAL, L.ATT
MODEL = sys.argv[1] if len(sys.argv) > 1 else "HRGPR"
FIT = json.loads((L.OUT / "拟合" / f"{MODEL}.json").read_text(encoding="utf-8"))
SP = L.Spec(**FIT["spec"]); X = np.array(FIT["X"]); SH = dict(FIT["shared"]); NAMES = SP.names()
STD = L.load_std(); PM, LAG = L.public_mods(N)
n, T = A.shape


def parts(zero=()):
    sh = dict(SH)
    for k in zero:
        sh[k] = 0.0
    phi = np.array([sh[k] for k in NAMES])
    run = lambda Xv: L.run(Xv, SP, phi, A, G, S, N, out="z")
    z = run(X); Xb = X.copy(); Xb[:, 1] = 0; Xk = X.copy(); Xk[:, 2] = 0
    bp = z - run(Xb); hp = z - run(Xk)
    return z - bp - hp, bp, hp


def fit(zfun, k, x0=None):
    nll = lambda p: float((np.logaddexp(0, zfun(p)) - A * zfun(p)).sum())
    r = minimize(nll, np.zeros(k) if x0 is None else x0, method="L-BFGS-B")
    _, ll = L.marginal_sigma(zfun(r.x), A)
    return r.x, ll


aH = float(expit(SH["logit_aH"])); H = np.full(n, .5); c = np.zeros((n, T))
for t in range(T):
    c[:, t] = 2 * H - 1; H += aH * (A[:, t] - H)
# 注意：push 项在 run 内部乘的是 c；parts 把 ψ_stab 置 0 后，这里重新加回各种编码
rest, bp, hp = parts(["θR_stab", "θG_stab", "ψ_stab"])
st = PM["stab"]
codes = {"线性": [(st - STD["stab"][0]) / STD["stab"][1]],
         "翻转": [(st == 0).astype(float) - (st == 0).mean()],
         "两者": [(st - STD["stab"][0]) / STD["stab"][1], (st == 0).astype(float) - (st == 0).mean()],
         "分类": [(st == k).astype(float) - (st == k).mean() for k in (1, 2, 3, 4)]}


def make(Ms):
    m = len(Ms)
    def f(p):
        eB = sum(p[j] * Ms[j] for j in range(m)); eH = sum(p[m + j] * Ms[j] for j in range(m)); ps = sum(p[2 * m + j] * Ms[j] for j in range(m))
        return rest + bp * np.exp(eB)[None] + hp * np.exp(eH)[None] + ps[None] * c
    return f, 3 * m


out = {"模型": MODEL, "换人率_按稳定": {}}
sw = (A[:, 1:] != A[:, :-1]).mean(0)
for k in range(5):
    out["换人率_按稳定"][str(k)] = round(float(sw[st[1:] == k].mean()), 4)
ll0 = L.marginal_sigma(rest + bp + hp, A)[1]
res = {}
for lab, Ms in codes.items():
    f, k = make(Ms)
    x, ll = fit(f, k)
    res[lab] = (x, ll, k)
    out[lab] = dict(参数=np.round(x, 4).tolist(), 对数似然=round(ll, 2), 相对无稳定_LR=round(2 * (ll - ll0), 2), df=k)
    print(lab, out[lab], flush=True)
lr = lambda a, b: round(2 * (res[a][1] - res[b][1]), 2)
out["比较"] = {"翻转相对线性_对数似然差(参数数相同)": round(res["翻转"][1] - res["线性"][1], 2),
              "线性之上加翻转_LR_df3": lr("两者", "线性"), "p_加翻转": float(chi2.sf(max(lr("两者", "线性"), 0), 3)),
              "翻转之上加线性_LR_df3": lr("两者", "翻转"), "p_加线性": float(chi2.sf(max(lr("两者", "翻转"), 0), 3)),
              "分类相对翻转_LR_df9": lr("分类", "翻转"), "p_分类相对翻转": float(chi2.sf(max(lr("分类", "翻转"), 0), 9))}
x = res["分类"][0]
out["分类剖面(相对稳定=0)"] = {"信念权重": [0] + np.round(x[0:4], 3).tolist(), "惯性权重": [0] + np.round(x[4:8], 3).tolist(),
                           "习惯推力": [0] + np.round(x[8:12], 3).tolist()}
print(out["比较"]); print(out["分类剖面(相对稳定=0)"])
L.save_json(out, L.OUT / f"翻转检验_{MODEL}.json")
