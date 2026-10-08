# -*- coding: utf-8 -*-
"""HRGPRS（HRGPR + 饱和分级反应）的联合估计 → 精修 → 补充优化 → 模型比较。起点用 HRGPR 的个体参数。"""
import sys, json, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import fit_models as FM
import polish as P
import refine as RF

if __name__ == "__main__":
    name = "HRGPRS"; spec = FM.MODELS[name]
    if not (L.CKPT / f"fit_{name}.done").exists():
        src = json.loads((L.OUT / "拟合" / "HRGPR.json").read_text(encoding="utf-8"))
        sh = dict(src["shared"]); sh.update({"λ_+5": 0.0, "λ_+10": 0.0, "λ_-5": 0.0, "λ_-10": 0.0})
        phi0 = np.array([sh[k] for k in spec.names()])
        logf = open(L.W / "日志" / f"fit_{name}.log", "a", encoding="utf-8"); log = lambda s: (logf.write(s + "\n"), logf.flush())
        X, phi, f, hist = L.joint_fit(spec, L.A_REAL, L.G_REAL, L.S_REAL, L.ATT, np.array(src["X"]), phi0, log=log, ckpt=L.CKPT / f"轮_{name}.npz")
        z = L.run(X, spec, phi, L.A_REAL, L.G_REAL, L.S_REAL, L.ATT, out="z"); sig, lls = L.marginal_sigma(z, L.A_REAL)
        L.save_json(dict(name=name, spec=spec.to_dict(), shared=dict(zip(spec.names(), phi.tolist())), k_shared=spec.k, nll=float(f.sum()),
                         nll_i=f, X=X, sigma=sig, loglik_sigma=lls, history=hist), L.OUT / "拟合" / f"{name}.json")
        (L.CKPT / f"fit_{name}.done").write_text("ok")
    path = L.OUT / "拟合" / f"{name}.json"
    print(P.job(path), flush=True); print(RF.job(path), flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("compare.py"))], check=True, stdout=subprocess.DEVNULL)
    (L.CKPT / "fit_models6.done").write_text("ok")
