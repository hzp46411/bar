# -*- coding: utf-8 -*-
"""
第 41 步：用联合估计的最终模型做完整的 ABM——HB（习惯 + 世界模型）的个体参数与共同因素（γ、σ）一起估计（s38）
  agent：100 个，参数取 s38 的联合估计；每轮所有人共享 −γ·(N_{t−1} − 60)/10 + σ·ε_t（在 ABM 中 γ 作为全体共同的 λ1 = −γ）
  评价：事先固定的联合检验（s36：指纹 = 均值、SD、ACF1–3；马氏距离² + Hotelling；合成似然；D、换选择比例），每个人群 1000 次；
        另报告效率、挤的比例、常客等（s27 的指标）
  A 拟合检验（完整模型）
  B 成分敲除（保持水平：修改后把每人在真实历史上的平均 logit 变化加回 b，只改变动态）：
     去习惯（w_I = 0）、世界模型不进选择（w_B = 0）、去共同分级反应（γ = 0）、去共同冲击（σ = 0）、只留共同因素（w_I = w_B = 0）
  C 人群构成（不调整水平）：δ = 0、全体反转（+|δ|）、全体外推（−|δ|）、同质人群（各参数取中位数）、打乱人群（各参数列独立置换）
  D 扰动自稳：所有人的 b 加同一个 Δb（±0.5、±1），自稳增益 = 1 − 平移 / 无反馈平移（Σp(1−p)·Δb）；完整、去世界模型、去 γ、去习惯
  E 预测者生态：反转者比例 f = 0…1（|δ| 不变，方向随机分配，每个 f 20 种分配 × 25 次）；真实分配；同比例下随机置换方向（30 次 × 25）
用法：python3 s41_联合估计ABM.py
输出：结果/s41_联合估计ABM.json
"""
import json, importlib.util, pathlib
import numpy as np
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s36", str(pathlib.Path(__file__).with_name("s36_共同因素层.py")))
S36 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S36)
S27 = S36.S27
B_SIM = 1000
KEYS = ("平均人数", "SD", "ACF1", "ACF2", "ACF3", "ACF4", "挤的比例", "效率", "换选择率", "个人去的比例SD", "常客(>.8)")


def evaluate(Y, sc, seed, OBS, OBS_D, OBS_SW, real):
    fps, Ds, sws, css, acf4, mets = [], [], [], [], [], []
    for b in range(4):
        Ns, As = S27.simulate(Y, B_SIM // 4, seed + 10 * b, sc)
        fps.append(S36.fingerprint(Ns)); d_, sw_ = S36.micro(Ns, As); Ds.append(d_); sws.append(sw_)
        css.append((Ns >= 61).mean(1)); acf4.append(S27.acf(Ns, 4)); mets.append(S27.metrics(Ns, As))
    SIM = tuple(np.concatenate(x) for x in (fps, Ds, sws, css, acf4))
    M = {k: np.concatenate([m[k] for m in mets]) for k in mets[0]}
    summ = S27.summarize(M, real)
    return dict(联合检验=S36.joint(OBS, OBS_D, OBS_SW, *SIM), 指标={k: summ[k] for k in KEYS})


def gain(Y, sc):
    Ns0, A0 = S27.simulate(Y, 200, 4, sc); base = Ns0.mean()
    p_bar = A0.astype(float).mean((0, 2)); sl = float((p_bar * (1 - p_bar)).sum())
    row = {}
    for db in (-1.0, -0.5, 0.5, 1.0):
        sh = float(S27.simulate(Y, 200, 4, sc, db=db)[0].mean() - base)
        row[f"Δb={db:+.1f}"] = dict(平移=round(sh, 2), 无反馈平移=round(sl * db, 2), 自稳增益=round(1 - sh / (sl * db), 3))
    return dict(基线平均人数=round(float(base), 2), **row)


if __name__ == "__main__":
    r = json.loads((PL.OUT / "s38_联合估计_HB.json").read_text(encoding="utf-8"))
    g, s = r["γ"], r["σ"]
    X0 = S27.load("s38:HB")
    X = X0.copy(); X[:, 13] = -g
    OBS = S36.fingerprint(S27.N_REAL[None])[0]
    OBS_D, OBS_SW = (v[0] for v in S36.micro(S27.N_REAL[None], S36.A[None].astype(np.int8)))
    real = {k: float(v[0]) for k, v in S27.metrics(S27.N_REAL[None], S36.A[None].astype(np.int8)).items()}
    out = dict(模型="HB + γ + σ（联合估计，s38）", γ=g, σ=s,
               预测类型_模型δ=dict(反转=int((X[:, 6] > 1).sum()), 外推=int((X[:, 6] < -1).sum()), 接近零=int((np.abs(X[:, 6]) <= 1).sum())))
    out["A 拟合检验"] = evaluate(X, s, 100, OBS, OBS_D, OBS_SW, real)
    print("A", out["A 拟合检验"]["联合检验"]["联合p"], flush=True)

    def drop(cols):
        Y = X.copy()
        for c in cols:
            Y[:, c] = 0.0
        return S27.recenter(X, Y)
    ko = {"去习惯（w_I = 0）": (drop([1]), s), "世界模型不进选择（w_B = 0）": (drop([4]), s), "去共同分级反应（γ = 0）": (drop([13]), s),
          "去共同冲击（σ = 0）": (X, 0.0), "只留共同因素（w_I = w_B = 0）": (drop([1, 4]), s)}
    out["B 成分敲除"] = {}
    for k, (Y, sc) in ko.items():
        out["B 成分敲除"][k] = evaluate(Y, sc, 200, OBS, OBS_D, OBS_SW, real); print("B", k, flush=True)
    cols = [c for c in range(18) if c not in (11, 12)]
    med = X.copy(); med[:, cols] = np.median(X[:, cols], 0)
    shuf = X.copy(); rng = np.random.default_rng(11)
    for c in cols:
        shuf[:, c] = rng.permutation(X[:, c])
    def set_delta(f):
        Y = X.copy(); Y[:, 6] = f(X[:, 6]); return Y
    comp = {"δ = 0": set_delta(lambda d: 0 * d), "全体反转（+|δ|）": set_delta(np.abs), "全体外推（−|δ|）": set_delta(lambda d: -np.abs(d)),
            "同质人群（参数取中位数）": med, "打乱人群（参数列独立置换）": shuf}
    out["C 人群构成"] = {}
    for k, Y in comp.items():
        out["C 人群构成"][k] = evaluate(Y, s, 300, OBS, OBS_D, OBS_SW, real); print("C", k, flush=True)
    out["D 扰动自稳"] = {"完整": gain(X, s), "世界模型不进选择": gain(ko["世界模型不进选择（w_B = 0）"][0], s),
                     "去共同分级反应": gain(ko["去共同分级反应（γ = 0）"][0], s), "去习惯": gain(ko["去习惯（w_I = 0）"][0], s)}
    print("D 完成", flush=True)
    mag = np.abs(X[:, 6]); eco = {}
    for f in np.round(np.arange(0, 1.01, 0.1), 1):
        acc = {k: [] for k in ("SD", "ACF1", "效率")}
        for k in range(20):
            Y = X.copy(); Y[:, 6] = np.where(np.random.default_rng(100 + k).random(len(mag)) < f, mag, -mag)
            Mx = S27.metrics(*S27.simulate(Y, 25, 1000 + k, s))
            for kk in acc:
                acc[kk].append(Mx[kk])
        eco[f"反转比例={f:.1f}"] = {kk: round(float(np.concatenate(v).mean()), 4) for kk, v in acc.items()}
    Mr = S27.metrics(*S27.simulate(X, 500, 7, s))
    perm = {k: [] for k in ("SD", "ACF1", "效率")}
    for k in range(30):
        Y = X.copy(); Y[:, 6] = mag * np.where(np.random.default_rng(500 + k).permutation(np.sign(X[:, 6])) > 0, 1, -1)
        Mx = S27.metrics(*S27.simulate(Y, 25, 1700 + k, s))
        for kk in perm:
            perm[kk].append(float(Mx[kk].mean()))
    out["E 预测者生态"] = dict(扫描=eco, 真实分配=dict(反转比例=round(float((X[:, 6] > 0).mean()), 2),
                                                 **{k: round(float(Mr[k].mean()), 4) for k in ("SD", "ACF1", "效率")}),
                              同比例随机置换方向={k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)] for k, v in perm.items()})
    PL.save(out, "s41_联合估计ABM.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))
