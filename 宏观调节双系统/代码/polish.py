# -*- coding: utf-8 -*-
"""
第 4 层 · 步骤 1a：全参数联合精修（交替估计在 8 轮内未必完全收敛）
  对 结果/拟合/*.json 中尚未精修的模型，从交替估计的终点出发，个体参数与共用参数一起做 L-BFGS-B，
  再重估 σ。原交替结果保留在字段 "交替" 中；精修后的值覆盖 nll、X、shared、sigma、loglik_sigma。
"""
import sys, json, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

DIR = L.OUT / "拟合"


def job(path):
    fit = json.loads(path.read_text(encoding="utf-8"))
    if fit.get("polished"):
        return path.stem, "已精修（跳过）"
    sp = L.Spec(**fit["spec"])
    X = np.array(fit["X"]); phi = np.array([fit["shared"][k] for k in sp.names()])
    t0 = time.time()
    X2, phi2, f2, info = L.joint_lbfgs(sp, L.A_REAL, L.G_REAL, L.S_REAL, L.ATT, X, phi)
    if f2.sum() > fit["nll"]:                                  # 精修不应变差；万一变差则保留原解
        X2, phi2, f2 = X, phi, np.array(fit["nll_i"])
    z = L.run(X2, sp, phi2, L.A_REAL, L.G_REAL, L.S_REAL, L.ATT, out="z")
    sig, lls = L.marginal_sigma(z, L.A_REAL)
    fit["交替"] = dict(nll=fit["nll"], shared=fit["shared"], sigma=fit["sigma"], loglik_sigma=fit["loglik_sigma"])
    fit.update(nll=float(f2.sum()), nll_i=f2, X=X2, shared=dict(zip(sp.names(), phi2.tolist())), sigma=sig, loglik_sigma=lls,
               polished=True, polish_info=info, polish_seconds=time.time() - t0)
    L.save_json(fit, path)
    return path.stem, f"{fit['交替']['nll']:.3f} → {f2.sum():.3f}"


if __name__ == "__main__":
    paths = sorted(DIR.glob("*.json"))
    with Pool(4) as pool:
        for name, msg in pool.imap_unordered(job, paths):
            print(name, msg, flush=True)
    (L.CKPT / "polish.done").write_text("ok")
