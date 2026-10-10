# -*- coding: utf-8 -*-
"""第 17 步（续）：群体层面的"双重分离"——信念系统与习惯系统各自维持群体秩序的哪一部分？
  在同一个闭环模型（HRGPR）中分别"关掉"信念系统（β = 0）与习惯系统（κ = 0、ψ = 0），比较两类群体特征：
    人数（多少人去）：平均人数偏离容量、扰动自稳（来自 s17）、人数 SD
    人选（谁在去）：换人对稳定的斜率（稳定越久换人越少）、换人对偏离的斜率、分工度、前后半程角色稳定、个人记忆（去均值的自相关）
  口径与第 4 层 third_arrow.structure 相同；每种 300 次。
输出：结果/s17b_双重分离.json
"""
import json, sys, importlib.util, pathlib
import numpy as np
import pred_lib as PL
spec = importlib.util.spec_from_file_location("s17", str(pathlib.Path(__file__).with_name("s17_宏观秩序的生成.py")))
S17 = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]; spec.loader.exec_module(S17)
L = PL.L; STD = L.load_std()
X0, PHI0, SIG0, n, T = S17.X0, S17.PHI0, S17.SIG0, S17.n, S17.T
B = 300


def struct(N, A):
    d = N - 60; pm, _ = L.public_mods(N)
    st = (pm["stab"] - STD["stab"][0]) / STD["stab"][1]
    sw = (A[:, 1:] != A[:, :-1]).mean(0)
    ad = np.abs(d[:-1]) / 10
    Xr = np.column_stack([np.ones(T - 2), ad[1:], st[2:]]); b, *_ = np.linalg.lstsq(Xr, sw[1:], rcond=None)
    p = A.mean(1); pb = p.mean()
    h1, h2 = A[:, :T // 2].mean(1), A[:, T // 2:].mean(1)
    Ac = A - p[:, None]; num = (Ac[:, 1:] * Ac[:, :-1]).sum(1); den = (Ac * Ac).sum(1)
    return dict(平均人数=float(N.mean()), 偏离容量=float(abs(N.mean() - 60)), 人数SD=float(N.std()),
                换人率=float(sw.mean()), 换人_稳定斜率=float(b[2]), 换人_偏离斜率=float(b[1]),
                分工度=float(p.var() / (pb * (1 - pb))), 角色稳定=float(np.corrcoef(h1, h2)[0, 1]),
                个人记忆1=float(np.nanmean(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan))))


X_nb = X0.copy(); X_nb[:, 1] = 0.0
conds = {"完整": (X0, {}), "关信念系统（β = 0）": (X_nb, {}), "关习惯系统（κ = 0，ψ = 0）": (X0, dict(no_habit=True)), "关 λ": (X0, dict(no_lam=True))}
out = {"真实": struct(PL.N.astype(float), PL.A)}
print("真实", {k: round(v, 4) for k, v in out["真实"].items()}, flush=True)
for name, (X, kw) in conds.items():
    ms = [struct(*S17.simulate(X, PHI0, SIG0, 400000 + k, **kw)) for k in range(B)]
    out[name] = {k: dict(均值=float(np.mean([m[k] for m in ms])), SD=float(np.std([m[k] for m in ms]))) for k in ms[0]}
    print(name, {k: round(v["均值"], 4) for k, v in out[name].items()}, flush=True)
# 相对完整模型的变化（以模拟间 SD 计）
for name in list(conds)[1:]:
    out[name + "：相对完整的变化（SD）"] = {k: (out[name][k]["均值"] - out["完整"][k]["均值"]) / max(out["完整"][k]["SD"], 1e-9) for k in out["完整"]}
d17 = json.loads((PL.OUT / "s17_宏观秩序的生成.json").read_text(encoding="utf-8"))
out["扰动自稳（来自 s17）"] = {k: v["自稳增益"] for k, v in d17["B 扰动自稳"]["格"].items()}
PL.save(out, "s17b_双重分离.json")
print(json.dumps({k: v for k, v in out.items() if "变化" in k}, ensure_ascii=False)[:1500])
