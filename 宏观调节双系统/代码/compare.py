# -*- coding: utf-8 -*-
"""
第 4 层 · 步骤 1b：模型比较与系数表（读取 结果/拟合/*.json）
  似然比两种：条件似然（不含 σ，同轮选择相关会让它偏乐观）与含 σ 的边际似然（主判据）
  信息准则：共用参数的 BIC 惩罚用总观测数 ln(100 × 400)（与原项目 graded.py 相同）
输出：结果/模型比较.json；并打印表格
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

DIR = L.OUT / "拟合"
F = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in DIR.glob("*.json")}
NOBS = L.A_REAL.size


def lr(big, small):
    a, b = F[big], F[small]
    df = a["k_shared"] - b["k_shared"]
    LRc = 2 * (b["nll"] - a["nll"]); LRs = 2 * (a["loglik_sigma"] - b["loglik_sigma"])
    return dict(比较=f"{big} vs {small}", df=df, LR条件=round(LRc, 2), LR含σ=round(LRs, 2),
                p含σ=float(chi2.sf(max(LRs, 0), df)), ΔBIC=round(-LRc + df * np.log(NOBS), 1), ΔAIC=round(-LRc + 2 * df, 1))


def ratio_table(name):
    fit = F[name]; sh = fit["shared"]
    rows = []
    for m in L.MODS:
        tR, tG = sh.get(f"θR_{m}"), sh.get(f"θG_{m}")
        rows.append(dict(变量=L.LABEL[m], θR比例=None if tR is None else round(tR, 4), 比例倍数每SD=None if tR is None else round(float(np.exp(tR)), 3),
                         θG增益=None if tG is None else round(tG, 4),
                         θ信念=None if tR is None or tG is None else round(tG + tR / 2, 4), θ惯性=None if tR is None or tG is None else round(tG - tR / 2, 4)))
    return rows


if __name__ == "__main__":
    out = {"模型": {n: dict(k_shared=f["k_shared"], NLL=round(f["nll"], 2), σ=round(f["sigma"], 4), 边际对数似然=round(f["loglik_sigma"], 2),
                          共用参数={k: round(v, 4) for k, v in f["shared"].items()}) for n, f in sorted(F.items())}}
    tests = []
    pairs = [("RG", "M0"), ("R", "M0"), ("G", "M0"), ("RG", "G"), ("RG", "R"), ("H0", "M0"), ("HRG", "H0")] + \
            [("RG", f"RG-{m}") for m in L.MODS]
    for big, small in pairs:
        if big in F and small in F:
            tests.append(lr(big, small))
    out["检验"] = tests
    for n in ("RG", "HRG", "R"):
        if n in F:
            out[f"{n}_系数"] = ratio_table(n)
    if "HRG" in F:
        out["HRG_习惯痕迹速率"] = float(1 / (1 + np.exp(-F["HRG"]["shared"]["logit_aH"])))
    if "H0" in F:
        out["H0_习惯痕迹速率"] = float(1 / (1 + np.exp(-F["H0"]["shared"]["logit_aH"])))
    L.save_json(out, L.OUT / "模型比较.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    (L.CKPT / "compare.done").write_text("ok")
