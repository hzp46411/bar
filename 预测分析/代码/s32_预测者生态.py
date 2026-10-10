# -*- coding: utf-8 -*-
"""
第 32 步：预测者生态的剂量—反应（Arthur 的第三问：异质的内部模型怎样共同产生协调）
  用主模型（s28 的最终拟合）的 ABM（s27 的 simulate；共同冲击 σ_c 同 s27 的估计）：
  A 反转 / 外推的比例：每人保留方向强度 |δ|，随机让比例 f 的人取 +|δ|（反转），其余取 −|δ|（外推）；
     f = 0, .1, …, 1，每个 f 取 20 种随机分配 × 每种 20 次模拟；另报告真实的方向分配
  B 内部模型的强度：所有人的 δ 乘以 c（0、.5、1、1.5、2）；κ 同样缩放
  C 世界模型在选择中的权重 w_B 乘以 c；D 习惯权重（w_I、w_D）乘以 c——C、D 保持水平（s27 的 recenter）
  指标：平均人数、|平均 − 60|、SD、ACF1、效率（每人每轮得分）、挤的比例
用法：python3 s32_预测者生态.py <模型>       如 s28:HDWLG1
输出：结果/s32_预测者生态_<模型>.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s27", str(pathlib.Path(__file__).with_name("s27_ABM.py")))
S27 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S27)
KEYS = ("平均人数", "|平均−60|", "SD", "ACF1", "效率", "挤的比例")


def summ(Y, R, seed, sig_c):
    M = S27.metrics(*S27.simulate(Y, R, seed, sig_c))
    return {k: [round(float(M[k].mean()), 4), round(float(M[k].std()), 4)] for k in KEYS}


if __name__ == "__main__":
    m = sys.argv[1]
    X = S27.load(m); sig_c, _ = S27.common_shock(X)
    mag = np.abs(X[:, 6]); out = dict(模型=m, 共同冲击σ_c=round(sig_c, 4), 说明="每格为 [模拟均值, 模拟间 SD]")
    out["真实的方向分配"] = summ(X, 200, 1, sig_c)
    rows = {}
    for f in np.round(np.arange(0, 1.01, 0.1), 1):
        acc = {k: [] for k in KEYS}
        for s in range(20):
            rng = np.random.default_rng(100 + s)
            Y = X.copy(); Y[:, 6] = np.where(rng.random(len(mag)) < f, mag, -mag)
            Mx = S27.metrics(*S27.simulate(Y, 20, 1000 + s, sig_c))
            for k in KEYS:
                acc[k].append(Mx[k])
        rows[f"反转比例={f:.1f}"] = {k: [round(float(np.concatenate(v).mean()), 4), round(float(np.concatenate(v).std()), 4)] for k, v in acc.items()}
        print("A", f, rows[f"反转比例={f:.1f}"]["SD"], rows[f"反转比例={f:.1f}"]["效率"], flush=True)
    out["A 反转 / 外推的比例"] = rows
    out["B 内部模型的强度（δ、κ × c）"] = {}
    out["C 世界模型权重（w_B × c，保持水平）"] = {}
    out["D 习惯权重（w_I、w_D × c，保持水平）"] = {}
    for c in (0.0, 0.5, 1.0, 1.5, 2.0):
        Y = X.copy(); Y[:, 6] *= c; Y[:, 14] *= c
        out["B 内部模型的强度（δ、κ × c）"][f"c={c}"] = summ(Y, 200, 2, sig_c)
        Y = X.copy(); Y[:, 4] *= c
        out["C 世界模型权重（w_B × c，保持水平）"][f"c={c}"] = summ(S27.recenter(X, Y), 200, 3, sig_c)
        Y = X.copy(); Y[:, 1] *= c; Y[:, 9] *= c
        out["D 习惯权重（w_I、w_D × c，保持水平）"][f"c={c}"] = summ(S27.recenter(X, Y), 200, 4, sig_c)
        print("BCD", c, flush=True)
    PL.save(out, f"s32_预测者生态_{m.replace(':', '_')}.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))
