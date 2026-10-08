# -*- coding: utf-8 -*-
"""
闭环第三环："被调节的微观"如何生成宏观
  宏观不止"多少人去"（水平：人数序列），还有"谁去"（结构：组成、换人、分工、收益分配）。
  调节作用在 c（自己的习惯方向）上：对人数，重复者与交替者、去者与不去者相互抵消；
  对"换不换人"，每一次换选都同号相加——所以预期调节的宏观足迹主要在结构上，而不在水平上。
本脚本：
  structure(N, A)  计算水平 + 结构 + 福利三类宏观量（真实数据与模拟用同一函数）
  python third_arrow.py 敲除 [模型名] [B]      逐条关掉调节通路，看各宏观量变了多少（结果/第三环_敲除_<模型>.json）
  python third_arrow.py 剂量 [模型名] [B]      把每条调节通路的强度乘以 0–3 倍（结果/第三环_剂量_<模型>.json）
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import abm_arb as Ab
from state_ar import coefs

STD = L.load_std()


def acf(x, k):
    x = x - x.mean(); d = x @ x
    return float(x[k:] @ x[:-k] / d) if d > 0 else 0.0


def payoff(N, A):
    crowd = N > L.CAP
    return np.where(A == 1, (~crowd).astype(float)[None, :], 0.7 * crowd.astype(float)[None, :])


def gini(x):
    x = np.sort(x); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def structure(N, A):
    n, T = A.shape
    d = N - L.CAP
    pm, _ = L.public_mods(N)
    st = (pm["stab"] - STD["stab"][0]) / STD["stab"][1]
    out = {}
    # ---- 水平（人数序列）----
    out["sd"] = float(N.std()); out["acf1"] = acf(N, 1); out["sq_acf1"] = acf((N - N.mean()) ** 2, 1)
    out["φ0"], out["φs"], out["φn"] = map(float, coefs(N))
    # ---- 结构（谁去）----
    p = A.mean(1); pb = p.mean()
    out["分工度"] = float(p.var() / (pb * (1 - pb)))                        # 0 = 人人去的比例相同；1 = 完全固定角色
    h1, h2 = A[:, :T // 2].mean(1), A[:, T // 2:].mean(1)
    out["角色稳定"] = float(np.corrcoef(h1, h2)[0, 1])                        # 前后半程个人去的比例的相关
    blk = [A[:, i * 100:(i + 1) * 100].mean(1) for i in range(T // 100)]
    sp = [b.var() / (b.mean() * (1 - b.mean())) for b in blk]
    out["分工度_增长"] = float(np.polyfit(np.arange(len(sp)), sp, 1)[0])      # 每 100 轮分工度的变化
    sw = (A[:, 1:] != A[:, :-1]).mean(0)                                    # 每轮换选的比例（换人率）
    out["换人率"] = float(sw.mean()); out["换人率_sd"] = float(sw.std()); out["换人率_acf1"] = acf(sw, 1)
    # 换人对上一轮宏观状态的依赖（宏观 → 换人）：sw_t ~ |d_{t-1}| + 稳定_{t-1}
    ad = np.abs(d[:-1]) / 10; s1 = st[:-1]
    Xr = np.column_stack([np.ones(T - 2), ad[1:], s1[1:]])
    b, *_ = np.linalg.lstsq(Xr, sw[1:], rcond=None)
    out["换人_偏离斜率"], out["换人_稳定斜率"] = float(b[1]), float(b[2])
    # 换人是否"配对"：净变化 |ΔN| 相对于换选总数（越小 = 换进与换出越平衡）
    nsw = sw * n; dN = np.abs(np.diff(N))
    out["换人配对度"] = float(1 - dN.sum() / nsw.sum())
    out["换人→下轮偏离"] = float(np.corrcoef(sw[:-1], np.abs(d[2:]))[0, 1])   # 本轮大换人是否预示下一轮更大偏离
    # 个人选择的去均值自相关（组成记忆）
    Ac = A - p[:, None]
    for k in (1, 2, 5):
        num = (Ac[:, k:] * Ac[:, :-k]).sum(1); den = (Ac * Ac).sum(1)
        out[f"个人记忆{k}"] = float(np.nanmean(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)))
    # ---- 福利 ----
    P = payoff(N, A)
    out["效率"] = float(P.sum(0).mean() / L.CAP)                              # 每轮总收益 / 60（理论最大值）
    out["不平等"] = gini(P.sum(1))
    out["拥挤率"] = float((N > L.CAP).mean())
    return out


def sim_struct(args):
    fname, mod, seed = args
    N, A, _ = Ab.simulate(fname, mod, seed)
    return structure(N, A)


def summarize(res, obs):
    keys = list(obs)
    arr = {k: np.array([r[k] for r in res]) for k in keys}
    return {k: dict(均值=float(np.nanmean(arr[k])), sd=float(np.nanstd(arr[k])),
                    真实分位=float(np.nanmean(arr[k] <= obs[k]))) for k in keys}, arr


def run_set(variants, B, base_seed):
    jobs = [(fn, mod, base_seed + 7919 * i + 101 * j) for j, (lab, (fn, mod)) in enumerate(variants.items()) for i in range(B)]
    with Pool(4) as pool:
        res = pool.map(sim_struct, jobs, chunksize=20)
    out = {}
    for j, lab in enumerate(variants):
        out[lab] = res[j * B:(j + 1) * B]
    return out


def channel_off(which, sh):
    """只关掉一个通道上的宏观调节（稳定、偏离、可靠性）。信念效应 eb = θG + θR/2，惯性效应 eh = θG − θR/2。"""
    new = {}
    for m in ("stab", "dev"):
        eb = sh[f"θG_{m}"] + sh[f"θR_{m}"] / 2; eh = sh[f"θG_{m}"] - sh[f"θR_{m}"] / 2
        if which == "信念":
            new[f"θG_{m}"], new[f"θR_{m}"] = eh / 2, -eh
        else:
            new[f"θG_{m}"], new[f"θR_{m}"] = eb / 2, eb; new[f"ψ_{m}"] = 0.0
    new["θB_relB" if which == "信念" else "θH_relH"] = 0.0
    return new


def knockouts(model):
    reg = ["stab", "dev"]
    sh = Ab.FIT(model)["shared"]
    return {"完整": (model, {}),
            "只关信念通道": (model, {"phi_set": channel_off("信念", sh)}),
            "只关惯性通道": (model, {"phi_set": channel_off("惯性", sh)}),
            "关稳定调节": (model, {"zero_mods": ["stab"]}),
            "关偏离调节": (model, {"zero_mods": ["dev"]}),
            "关可靠性仲裁": (model, {"no_rel": True}),
            "关稳定推力": (model, {"phi_scale": {"ψ_stab": 0.0}}),
            "关习惯累积": (model, {"no_accum": True}),
            "关全部宏观调节": (model, {"zero_mods": reg, "no_rel": True}),
            "只有共同因素(M0)": ("M0", {}), "习惯无调节(H0)": ("H0", {})}


def doses(model):
    sh = Ab.FIT(model)["shared"]
    v = {}
    for f in (0.0, 0.5, 1.0, 2.0, 3.0):
        v[f"ψ_stab×{f}"] = (model, {"phi_scale": {"ψ_stab": f}})
        v[f"偏离调节×{f}"] = (model, {"phi_scale": {"θR_dev": f, "θG_dev": f, "ψ_dev": f}})
        v[f"可靠性×{f}"] = (model, {"phi_scale": {"θB_relB": f, "θH_relH": f}})
    return v


if __name__ == "__main__":
    mode = sys.argv[1]; model = sys.argv[2] if len(sys.argv) > 2 else "HRGPR"; B = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    obs = structure(L.ATT, L.A_REAL)
    variants = knockouts(model) if mode == "敲除" else doses(model)
    raw = run_set(variants, B, 500000 if mode == "敲除" else 700000)
    out = {"模型": model, "B": B, "真实": obs, "变体": {}}
    full = {k: np.array([r[k] for r in raw[next(iter(raw))]]) for k in obs}
    for lab, res in raw.items():
        s, arr = summarize(res, obs)
        if mode == "敲除" and lab != "完整":                                  # 与完整模型的差，以完整模型的模拟 SD 为单位
            for k in obs:
                s[k]["差(SD)"] = float((np.nanmean(arr[k]) - np.nanmean(full[k])) / np.nanstd(full[k]))
        out["变体"][lab] = s
    fn = L.OUT / f"第三环_{mode}_{model}.json"
    L.save_json(out, fn)
    keys = list(obs)
    print("真实", {k: round(obs[k], 3) for k in keys})
    for lab, s in out["变体"].items():
        print(lab, {k: (round(s[k]["均值"], 3), round(s[k]["真实分位"], 2), round(s[k].get("差(SD)", 0), 2)) for k in keys})
