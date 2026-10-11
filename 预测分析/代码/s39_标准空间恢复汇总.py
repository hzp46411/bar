# -*- coding: utf-8 -*-
"""
第 39 步：标准模型空间（2×2×2：习惯 H、MF F、世界模型 B 各有或无）的模型恢复与参数恢复汇总
  数据：s22 的恢复拟合（每个生成模型用其真实拟合值生成 1 套假数据，8 个候选都拟合；层级 EM + iBIC）；
        另加 s22 的校准（HB、HFB 各 4 套，只拟合 HB 与 HFB）
  模型恢复：群体层面按 iBIC 选模型 → 8 × 8 选中矩阵；生成模型的领先量（次优 − 生成模型；正 = 选对）
           因子层面：每个因子"有 / 无"的判断是否正确（由选中的模型读出）
  参数恢复：生成模型拟合自己的数据，真值与估计值的 Spearman 相关（α、η 用概率尺度）
用法：python3 s39_标准空间恢复汇总.py
输出：结果/s39_标准空间恢复.json
"""
import json
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
import pred_lib as PL

MODELS = ["0", "H", "F", "B", "HF", "HB", "FB", "HFB"]
NAMES = {0: "b", 1: "w_I", 2: "w_F", 3: "α_F", 4: "w_B", 5: "η", 6: "δ", 7: "lnσ", 8: "α_H"}
FREE = {m: [0, 5, 6, 7] + ([1, 8] if "H" in m else []) + ([2, 3] if "F" in m else []) + ([4] if "B" in m else []) for m in MODELS}
J = lambda f: json.loads((PL.OUT / f).read_text(encoding="utf-8"))

if __name__ == "__main__":
    out = {"选中": {}, "生成模型的领先量": {}, "相对最优": {}, "参数恢复": {}}
    for g in MODELS:
        ib = {m: J(f"s22_{g}_1_{m}.json")["iBIC"] for m in MODELS}
        best = min(ib, key=ib.get); order = sorted(ib, key=ib.get)
        out["选中"][g] = best
        out["生成模型的领先量"][g] = round((ib[order[1]] if best == g else ib[best]) - ib[g], 1)
        out["相对最优"][g] = {m: round(v - ib[best], 1) for m, v in ib.items()}
        d = J(f"s22_{g}_1_{g}.json"); Xt, Xe = np.array(d["X真值"]), np.array(d["X"])
        pr = {}
        for c in FREE[g]:
            tv, ev = (expit(Xt[:, c]), expit(Xe[:, c])) if c in (3, 5, 8) else (Xt[:, c], Xe[:, c])
            pr[NAMES[c]] = round(float(spearmanr(tv, ev)[0]), 3)
        out["参数恢复"][g] = pr
    fac = {}
    for f in "HFB":
        right = [(f in out["选中"][g]) == (f in g) for g in MODELS]
        fac[f] = dict(判断正确的生成模型数=int(sum(right)), 共=len(MODELS))
    out["因子层面"] = fac
    cal = {}
    for g in ("HB", "HFB"):
        cal[g] = [round(J(f"s22_{g}_{r}_HFB.json")["iBIC"] - J(f"s22_{g}_{r}_HB.json")["iBIC"], 1) for r in (1, 2, 3, 4)]
    out["校准：iBIC(HFB) − iBIC(HB)，各 4 套"] = cal
    pr_hb = {}
    for r in (1, 2, 3, 4):
        d = J(f"s22_HB_{r}_HB.json"); Xt, Xe = np.array(d["X真值"]), np.array(d["X"])
        for c in FREE["HB"]:
            tv, ev = (expit(Xt[:, c]), expit(Xe[:, c])) if c in (5, 8) else (Xt[:, c], Xe[:, c])
            pr_hb.setdefault(NAMES[c], []).append(round(float(spearmanr(tv, ev)[0]), 3))
    out["HB 参数恢复（4 套）"] = pr_hb
    PL.save(out, "s39_标准空间恢复.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))
