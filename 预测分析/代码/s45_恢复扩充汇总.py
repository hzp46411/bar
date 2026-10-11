# -*- coding: utf-8 -*-
"""
第 45 步（汇总部分）：标准模型空间（2×2×2）的恢复扩充——混淆矩阵与因子层面的概率、参数恢复、MF 校准分布
  数据：s22 的假数据拟合（结果/s22_<生成模型>_<编号>_<候选>.json；运行见 s45_恢复扩充.sh）
    编号 1–10：8 个生成模型各 10 套，每套拟合全部 8 个候选 → 混淆矩阵、因子层面判断、参数恢复
    编号 1–30：HB、HFB 为生成模型，每套至少拟合 HB 与 HFB → MF 校准：Δ = iBIC(HFB) − iBIC(HB) 的分布
  模型恢复：每套按 iBIC 选模型；P(选中 | 生成模型)
  因子恢复：由选中的模型读出 H、F、B 各自"有 / 无"，与生成模型比较；分别报告因子存在时与不存在时判对的概率
  参数恢复：生成模型拟合自己的数据，真值与估计值的 Spearman 相关（各套分别计算，报告中位数与范围；另报告合并后的相关）；
           α_F、η、α_H 用概率尺度
  MF 校准：真实数据的 Δ（s22 的真实拟合）落在两个分布中的位置
只汇总已经完成的拟合，可随时运行。
用法：python3 s45_恢复扩充汇总.py
输出：结果/s45_恢复扩充.json
"""
import json
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
import pred_lib as PL

MODELS = ["0", "H", "F", "B", "HF", "HB", "FB", "HFB"]
NAMES = {0: "b", 1: "w_I", 2: "w_F", 3: "α_F", 4: "w_B", 5: "η", 6: "δ", 7: "lnσ", 8: "α_H"}
FREE = {m: [0, 5, 6, 7] + ([1, 8] if "H" in m else []) + ([2, 3] if "F" in m else []) + ([4] if "B" in m else []) for m in MODELS}
PROB = (3, 5, 8)
f_ = lambda g, r, m: PL.OUT / f"s22_{g}_{r}_{m}.json"
J = lambda p: json.loads(p.read_text(encoding="utf-8"))
scale = lambda X, c: expit(X[:, c]) if c in PROB else X[:, c]

if __name__ == "__main__":
    out = {}
    conf = {g: {m: 0 for m in MODELS} for g in MODELS}; lead = {g: [] for g in MODELS}
    fac = {f: dict(存在_判对=0, 存在_套数=0, 不存在_判对=0, 不存在_套数=0) for f in "HFB"}
    par = {g: {} for g in MODELS}; pooled = {g: ([], []) for g in MODELS}
    for g in MODELS:
        for r in range(1, 11):
            if not all(f_(g, r, m).exists() for m in MODELS):
                continue
            ib = {m: J(f_(g, r, m))["iBIC"] for m in MODELS}
            best = min(ib, key=ib.get); order = sorted(ib, key=ib.get)
            conf[g][best] += 1
            lead[g].append((ib[order[1]] if best == g else ib[best]) - ib[g])
            for f in "HFB":
                k = "存在" if f in g else "不存在"
                fac[f][f"{k}_套数"] += 1; fac[f][f"{k}_判对"] += int((f in best) == (f in g))
            d = J(f_(g, r, g)); Xt, Xe = np.array(d["X真值"]), np.array(d["X"])
            for c in FREE[g]:
                par[g].setdefault(NAMES[c], []).append(float(spearmanr(scale(Xt, c), scale(Xe, c))[0]))
            pooled[g][0].append(Xt); pooled[g][1].append(Xe)
    nset = {g: sum(conf[g].values()) for g in MODELS}
    out["模型恢复"] = dict(套数=nset,
                       选中概率={g: {m: round(conf[g][m] / nset[g], 3) for m in MODELS} if nset[g] else None for g in MODELS},
                       判对概率={g: round(conf[g][g] / nset[g], 3) if nset[g] else None for g in MODELS},
                       生成模型的领先量={g: dict(中位数=round(float(np.median(v)), 1), 最小=round(float(np.min(v)), 1)) if v else None
                                 for g, v in lead.items()},
                       说明="领先量 = 次优（或选中）模型的 iBIC − 生成模型的 iBIC；正 = 选对")
    out["因子恢复"] = {f: dict(因子存在时判对概率=round(v["存在_判对"] / max(v["存在_套数"], 1), 3),
                            因子不存在时判对概率=round(v["不存在_判对"] / max(v["不存在_套数"], 1), 3),
                            总判对概率=round((v["存在_判对"] + v["不存在_判对"]) / max(v["存在_套数"] + v["不存在_套数"], 1), 3), **v)
                     for f, v in fac.items()}
    pr = {}
    for g in MODELS:
        if not par[g]:
            continue
        Xt, Xe = np.vstack(pooled[g][0]), np.vstack(pooled[g][1])
        pr[g] = {nm: dict(各套中位数=round(float(np.median(v)), 3), 范围=[round(float(min(v)), 3), round(float(max(v)), 3)],
                          合并=round(float(spearmanr(scale(Xt, c), scale(Xe, c))[0]), 3))
                 for c in FREE[g] for nm, v in [(NAMES[c], par[g][NAMES[c]])]}
    out["参数恢复"] = pr
    real = J(PL.OUT / "s22_真实_HFB.json")["iBIC"] - J(PL.OUT / "s22_真实_HB.json")["iBIC"]
    cal = {}
    for g in ("HB", "HFB"):
        v = np.array([J(f_(g, r, "HFB"))["iBIC"] - J(f_(g, r, "HB"))["iBIC"] for r in range(1, 31)
                      if f_(g, r, "HFB").exists() and f_(g, r, "HB").exists()])
        if len(v):
            cal[g + " 为真"] = dict(套数=len(v), 中位数=round(float(np.median(v)), 1), 范围=[round(float(v.min()), 1), round(float(v.max()), 1)],
                                  选_HFB_的比例=round(float((v < 0).mean()), 3), 小于等于真实值的比例=round(float((v <= real).mean()), 3))
    out["MF 校准：Δ = iBIC(HFB) − iBIC(HB)"] = dict(真实数据=round(float(real), 1), **cal,
                                                 说明="负 = HFB 更好；HFB 为真时应全部为负，HB 为真时应全部为正")
    PL.save(out, "s45_恢复扩充.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))
