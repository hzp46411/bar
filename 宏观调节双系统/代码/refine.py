# -*- coding: utf-8 -*-
"""
第 4 层 · 步骤 1c：补充优化 —— 在每个模型精修后的共用参数下，再做一次个体多起点（含 4 个盆地起点），然后全参数 L-BFGS。
（测试发现 H0 由此从 19740.08 降到 19734.51：部分被试的个体参数停在局部最优。）分阶段存检查点，重启后续跑。
全部完成后重算模型比较（compare.py）。
"""
import sys, json, subprocess
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import recovery as R

DIR = L.OUT / "拟合"


def job(path):
    fit = json.loads(path.read_text(encoding="utf-8"))
    if fit.get("refined"):
        return path.stem, "已补充优化（跳过）"
    sp = L.Spec(**fit["spec"]); X = np.array(fit["X"]); phi = np.array([fit["shared"][k] for k in sp.names()])
    ck = L.CKPT / f"refine_{path.stem}.npz"
    if not ck.exists():
        np.savez(ck, stage=2, X=X, phi=phi)                    # 从"新共用参数下的个体多起点"阶段开始
    logf = open(L.W / "日志" / f"refine_{path.stem}.log", "a", encoding="utf-8")
    log = lambda s: (logf.write(s + "\n"), logf.flush())
    X2, phi2, f2 = R.staged_fit(sp, L.A_REAL, X, phi, 4242, ck, log)
    if f2.sum() >= fit["nll"] - 1e-6:
        fit["refined"] = True; L.save_json(fit, path)
        return path.stem, f"无改善（{fit['nll']:.3f}）"
    z = L.run(X2, sp, phi2, L.A_REAL, L.G_REAL, L.S_REAL, L.ATT, out="z")
    sig, lls = L.marginal_sigma(z, L.A_REAL)
    fit["补充优化前"] = dict(nll=fit["nll"], loglik_sigma=fit["loglik_sigma"], shared=fit["shared"])
    fit.update(nll=float(f2.sum()), nll_i=f2, X=X2, shared=dict(zip(sp.names(), phi2.tolist())), sigma=sig, loglik_sigma=lls, refined=True)
    L.save_json(fit, path)
    return path.stem, f"{fit['补充优化前']['nll']:.3f} → {f2.sum():.3f}"


if __name__ == "__main__":
    paths = sorted(DIR.glob("*.json"))
    with Pool(4) as pool:
        for name, msg in pool.imap_unordered(job, paths):
            print(name, msg, flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("compare.py"))], check=True, stdout=subprocess.DEVNULL)
    (L.CKPT / "refine.done").write_text("ok")
