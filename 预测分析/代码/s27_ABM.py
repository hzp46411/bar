# -*- coding: utf-8 -*-
"""
第 27 步：主模型的 ABM（闭环）——个体的内部模型怎样生成宏观协调
  agent：100 个，每人取主模型层级估计的后验众数（真实被试的参数）；人数 = 100 个 agent 的选择之和（闭环）
  每轮：习惯痕迹 H、慢漂移 D、近期信念 L（EWA δ = 1，或 MF δ = 0）、世界模型（B：以上轮状态为条件的预期；或 W）、
        对前几轮人数大小的分级反应（选择中的 λ1、λ2、λ4；世界模型中的 κ·lag1）进入 logit，
        另加同一轮所有人共享的冲击 ε_t ~ N(0, σ_c²)；之后用模拟的人数更新所有状态
  σ_c：开环（真实历史）下估计——人数残差 N_t − Σp_it 的方差超出二项方差 Σp(1−p) 的部分，换算到 logit 尺度
        Var_excess ≈ (Σ p(1−p))² σ_c²
  A 拟合检验：完整模型闭环模拟 B 次，宏观指标与真实数据比较（真实值在模拟分布中的百分位）
  B 成分敲除（保持水平：修改后把每人在真实历史上的平均 logit 变化加回 b，使平均倾向不变，只改变动态）：
     去习惯（H 与 D）、去慢漂移、去近期信念、去状态预测（B）、去世界模型中的分级（κ）、去世界模型（B 与 L）、
     去选择端的滞后反应（λ1、λ2、λ4；以及只去 λ2、只去 λ4）、去全部公共信息、状态预测无方向（δ = 0）、
     全体反转（δ = +|δ|）、全体外推（δ = −|δ|）、同质人群（各参数取中位数）、打乱人群（各参数列独立置换）、去共同冲击
  C 扰动自稳：所有人的 b 加同一个 Δb，平均人数的平移 / 无反馈时的平移（Σp(1−p)·Δb）；自稳增益 = 1 − 两者之比
     对 完整、去近期信念、去状态预测、去选择端的滞后反应、去全部公共信息、去习惯 分别计算
  指标：平均人数、|平均 − 60|、SD、ACF1–4、挤的轮次比例、效率（每人每轮得分：去且不挤 1，不去且挤 0.7）、换选择率、
        个人去的比例的 SD、常客（去的比例 > .8）、几乎不去（< .2）
用法：python3 s27_ABM.py <模型> [B]      模型 = HDLB（s25）、HLB / HW / HDW / HWL / HDWL（s26）、HB（s22）、HDB（s23）
输出：结果/s27_ABM_<模型>.json
"""
import sys, json
import numpy as np
from scipy.special import expit, ndtr
import pred_lib as PL

T = PL.T
A_REAL, G_REAL, S_REAL = PL.A, PL.G, PL.S
N_REAL = PL.N.astype(float)
N_FULL = PL._raw.iloc[0, 1:].astype(float).values
PRE = N_FULL[-T - 4:-T][::-1].copy()                                     # 第 1 轮之前的 4 轮人数：PRE[k−1] = N_{−k}
SRC = {"HDLB": "s25_真实_HDLB.json", "HDEB": "s25_真实_HDEB.json", "HB": "s22_真实_HB.json", "HDB": "s23_真实_HDB.json",
       "HLB": "s26_真实_HLB.json", "HW": "s26_真实_HW.json", "HDW": "s26_真实_HDW.json", "HWL": "s26_真实_HWL.json",
       "HDWL": "s26_真实_HDWL.json", **{m: f"s29_真实_{m}.json" for m in ("HDWR", "HDWG", "HDWLR", "HDWLG")},
       **{m: f"s30_真实_{m}.json" for m in ("HDWLG1", "HDWLG4", "HDWLG24", "HDWLG124")}}
FIXED = {"HDLB": (50.0, 0.0), "HDEB": (None, 0.0), "HB": (-50.0, 0.0), "HDB": (-50.0, 0.0),
         "HLB": (50.0, 0.0), "HW": (-50.0, 1.0), "HDW": (-50.0, 1.0), "HWL": (50.0, 1.0), "HDWL": (50.0, 1.0),
         "HDWR": (-50.0, 1.0), "HDWG": (-50.0, 1.0), "HDWLR": (50.0, 1.0), "HDWLG": (50.0, 1.0),
         **{m: (50.0, 1.0) for m in ("HDWLG1", "HDWLG4", "HDWLG24", "HDWLG124")}}   # (logit δ_EWA, 世界模型类型)


def load(m):
    """读取模型的个人参数，补成 17 列（与 s30 相同：b, wI, wF, aF, wB, η, δ, lnσ, aH, wD, aD, dE, W, λ1, κ, λ2, λ4）。"""
    X = np.array(json.loads((PL.OUT / SRC[m]).read_text(encoding="utf-8"))["X"])
    Y = np.zeros((X.shape[0], 17)); Y[:, :X.shape[1]] = X
    dE, W = FIXED[m]
    if dE is not None:
        Y[:, 11] = dE
    Y[:, 12] = W
    return Y


def step_terms(X, st, hist):
    """当前状态下各项对 logit 的贡献。st：状态字典；hist：每行前 1–4 轮的人数 (行, 4)。"""
    wI, wF, wB, delta, sig, wD, kap = X[:, 1], X[:, 2], X[:, 4], X[:, 6], np.exp(X[:, 7]), X[:, 9], X[:, 14]
    lag = (hist - 60) / 10; sp = (hist[:, 0] >= 61).astype(float)
    Mp = np.where(X[:, 12] > 0.5, st["mu"] + delta / 2 * (1 - 2 * sp) + kap * lag[:, 0], np.where(sp > 0.5, st["Mc"], st["Mn"]))
    p = ndtr((60.5 - Mp) / sig)
    R = X[:, 13] * lag[:, 0] + X[:, 15] * lag[:, 1] + X[:, 16] * lag[:, 3]
    return dict(H=wI * st["H"], D=wD * st["D"], L=wF * (st["QG"] - st["QS"]), B=wB * (1.7 * p - 0.7), R=R), Mp, sp


def init_state(X):
    r = X.shape[0]; delta = X[:, 6]
    return dict(Mn=60 + delta / 2, Mc=60 - delta / 2, mu=np.full(r, 60.0), QG=np.full(r, 0.5), QS=np.full(r, 0.35),
                H=np.zeros(r), D=np.zeros(r))


def update(X, st, a, Nt, sp, G, S):
    aF, eta, aH = expit(X[:, 3]), expit(X[:, 5]), expit(X[:, 8])
    aD, dE = aH * expit(X[:, 10]), expit(X[:, 11])
    st["Mc"] = np.where(sp > 0.5, st["Mc"] + eta * (Nt - st["Mc"]), st["Mc"])
    st["Mn"] = np.where(sp > 0.5, st["Mn"], st["Mn"] + eta * (Nt - st["Mn"]))
    st["mu"] = st["mu"] + eta * (Nt - st["mu"])
    st["QG"] = st["QG"] + aF * (a + (1 - a) * dE) * (G - st["QG"])
    st["QS"] = st["QS"] + aF * ((1 - a) + a * dE) * (0.7 * S - st["QS"])
    st["H"] = st["H"] + aH * (2 * a - 1 - st["H"])
    st["D"] = st["D"] + aD * (2 * a - 1 - st["D"])


def open_loop(X):
    """真实历史下的各项贡献（n, T）与选择概率。"""
    n = X.shape[0]; st = init_state(X); terms = {k: np.zeros((n, T)) for k in "HDLBR"}; P = np.zeros((n, T))
    hist = np.tile(PRE, (n, 1))
    for t in range(T):
        tm, _, sp = step_terms(X, st, hist)
        for k in tm:
            terms[k][:, t] = tm[k]
        P[:, t] = expit(X[:, 0] + sum(tm.values()))
        update(X, st, A_REAL[:, t], N_REAL[t], sp, G_REAL[:, t], S_REAL[:, t])
        hist = np.column_stack([np.full(n, N_REAL[t]), hist[:, :3]])
    return terms, P


def common_shock(X):
    _, P = open_loop(X)
    resid = N_REAL - P.sum(0); binom = (P * (1 - P)).sum(0)
    excess = max(resid.var() - binom.mean(), 0.0)
    return float(np.sqrt(excess) / binom.mean()), dict(残差方差=float(resid.var()), 二项方差=float(binom.mean()), 残差均值=float(resid.mean()))


def simulate(X, R, seed, sig_c, db=0.0):
    """R 次闭环模拟（一起向量化）。返回 人数 (R, T) 与 选择 (R, n, T)（int8）。"""
    n = X.shape[0]; rng = np.random.default_rng(seed)
    Xr = np.tile(X, (R, 1)); Xr[:, 0] = Xr[:, 0] + db
    st = init_state(Xr); hist_run = np.tile(PRE, (R, 1))
    Ns = np.zeros((R, T)); A = np.zeros((R, n, T), dtype=np.int8)
    for t in range(T):
        tm, _, sp = step_terms(Xr, st, np.repeat(hist_run, n, axis=0))
        eps = np.repeat(sig_c * rng.standard_normal(R), n)
        a = (rng.random(R * n) < expit(Xr[:, 0] + sum(tm.values()) + eps)).astype(float)
        Nt = a.reshape(R, n).sum(1)
        G = (np.repeat(Nt, n) - a <= 59).astype(float); S = (np.repeat(Nt, n) - a >= 61).astype(float)
        update(Xr, st, a, np.repeat(Nt, n), sp, G, S)
        Ns[:, t] = Nt; A[:, :, t] = a.reshape(R, n).astype(np.int8); hist_run = np.column_stack([Nt, hist_run[:, :3]])
    return Ns, A


def acf(x, k):
    x = x - x.mean(-1, keepdims=True)
    return (x[..., :-k] * x[..., k:]).sum(-1) / (x * x).sum(-1)


def metrics(Ns, A):
    """每次模拟一组宏观与个体指标。Ns：(R, T)；A：(R, n, T)。"""
    A = A.astype(float); others = Ns[:, None, :] - A
    pay = A * (others <= 59) + (1 - A) * 0.7 * (others >= 61)
    g = A.mean(2)
    return {"平均人数": Ns.mean(1), "|平均−60|": np.abs(Ns.mean(1) - 60), "SD": Ns.std(1),
            **{f"ACF{k}": acf(Ns, k) for k in (1, 2, 3, 4)}, "挤的比例": (Ns >= 61).mean(1), "效率": pay.mean((1, 2)),
            "换选择率": (A[:, :, 1:] != A[:, :, :-1]).mean((1, 2)), "个人去的比例SD": g.std(1),
            "常客(>.8)": (g > .8).sum(1).astype(float), "几乎不去(<.2)": (g < .2).sum(1).astype(float)}


def summarize(M, real=None):
    out = {}
    for k, v in M.items():
        d = dict(均值=round(float(v.mean()), 3), 模拟间SD=round(float(v.std()), 3))
        if real is not None:
            d["真实"] = round(float(real[k]), 3); d["真实的百分位"] = round(float((v < real[k]).mean() + 0.5 * (v == real[k]).mean()), 3)
        out[k] = d
    return out


def recenter(X, Y):
    """保持水平：把 Y 相对 X 在真实历史上的个人平均 logit 变化加回 b，使每人的平均倾向不变，只改变动态。"""
    tx, _ = open_loop(X); ty, _ = open_loop(Y)
    Y = Y.copy(); Y[:, 0] = Y[:, 0] + sum(tx[k].mean(1) for k in tx) - sum(ty[k].mean(1) for k in ty)
    return Y


def knockouts(X, terms):
    """成分敲除（保持水平）与人群构成的改变（不调整水平）。"""
    def drop(cols):
        Y = X.copy()
        for c in cols:
            Y[:, c] = 0.0
        return recenter(X, Y)
    cols = [c for c in range(17) if c not in (11, 12)]
    med = X.copy(); med[:, cols] = np.median(X[:, cols], 0)
    shuf = X.copy(); rng = np.random.default_rng(11)
    for c in cols:
        shuf[:, c] = rng.permutation(X[:, c])
    def set_delta(f):
        Y = X.copy(); Y[:, 6] = f(X[:, 6]); return Y
    return {"去习惯（H 与 D）": drop([1, 9]), "去慢漂移（D）": drop([9]), "去近期信念（L）": drop([2]),
            "去状态预测（B）": drop([4]), "去世界模型中的分级（κ = 0）": drop([14]), "去世界模型（B 与 L）": drop([2, 4]),
            "去选择端的滞后反应（λ1、λ2、λ4）": drop([13, 15, 16]), "只去 λ4": drop([16]), "只去 λ2": drop([15]),
            "去全部公共信息（B、L、λ）": drop([2, 4, 13, 15, 16]),
            "状态预测无方向（δ = 0）": set_delta(lambda d: 0 * d), "全体反转（δ = +|δ|）": set_delta(np.abs),
            "全体外推（δ = −|δ|）": set_delta(lambda d: -np.abs(d)), "同质人群（参数取中位数）": med, "打乱人群（参数列独立置换）": shuf}


if __name__ == "__main__":
    m = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    X = load(m)
    terms, _ = open_loop(X)
    sig_c, sc_info = common_shock(X)
    real = {k: float(v[0]) for k, v in metrics(N_REAL[None], A_REAL[None].astype(np.int8)).items()}
    out = dict(模型=m, 共同冲击σ_c=round(sig_c, 4), 共同冲击估计=sc_info,
               预测类型=dict(反转_δ大于1=int((X[:, 6] > 1).sum()), 外推_δ小于负1=int((X[:, 6] < -1).sum()), 接近零=int((np.abs(X[:, 6]) <= 1).sum())),
               各项在真实历史上的平均绝对贡献={k: round(float(np.abs(v).mean()), 3) for k, v in terms.items()})
    print(json.dumps(out, ensure_ascii=False), flush=True)
    out["A 拟合检验"] = summarize(metrics(*simulate(X, B, 1, sig_c)), real)
    out["A 拟合检验（无共同冲击）"] = summarize(metrics(*simulate(X, B, 2, 0.0)), real)
    print("A 完成", flush=True)
    out["B 成分敲除"] = {}
    for k, Y in knockouts(X, terms).items():
        out["B 成分敲除"][k] = summarize(metrics(*simulate(Y, B, 3, sig_c)))
        print("  ", k, flush=True)
    out["B 成分敲除"]["去共同冲击"] = summarize(metrics(*simulate(X, B, 3, 0.0)))
    ko = knockouts(X, terms)
    gains = {}
    for k, Y in [("完整", X), ("去近期信念（L）", ko["去近期信念（L）"]), ("去状态预测（B）", ko["去状态预测（B）"]),
                 ("去选择端的滞后反应（λ1、λ2、λ4）", ko["去选择端的滞后反应（λ1、λ2、λ4）"]), ("去全部公共信息（B、L、λ）", ko["去全部公共信息（B、L、λ）"]),
                 ("去习惯（H 与 D）", ko["去习惯（H 与 D）"])]:
        Ns0, A0 = simulate(Y, B // 2, 4, sig_c)
        base = Ns0.mean()
        p_bar = A0.astype(float).mean((0, 2)); slope = float((p_bar * (1 - p_bar)).sum())
        row = {}
        for db in (-1.0, -0.5, 0.5, 1.0):
            Ns1, _ = simulate(Y, B // 2, 4, sig_c, db=db)
            shift = float(Ns1.mean() - base); nofb = slope * db
            row[f"Δb={db:+.1f}"] = dict(平移=round(shift, 2), 无反馈平移=round(nofb, 2), 自稳增益=round(1 - shift / nofb, 3))
        gains[k] = dict(基线平均人数=round(float(base), 2), **row)
        print("  自稳", k, flush=True)
    out["C 扰动自稳"] = gains
    PL.save(out, f"s27_ABM_{m}.json")
    print(json.dumps({k: out[k] for k in ("共同冲击σ_c", "A 拟合检验")}, ensure_ascii=False, indent=1))
