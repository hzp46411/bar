# -*- coding: utf-8 -*-
"""
第 4 层 · 步骤 1：联合估计宏观调节双系统比例的模型族（个体参数与共用参数一起估计）
用法: python3 fit_models.py        （4 进程；每个模型完成即写 结果/拟合/<名>.json 与 检查点/fit_<名>.done）

模型（都含共用分级反应 λ；ratio = 进入"比例"r 的调节变量，gain = 进入"增益"g 的调节变量）
  M0        无调节（= 原项目联合估计的 BBLI + λ）
  R         只调比例：稳定、可靠性、偏离、轮次
  G         只调增益：同上四个
  RG        主模型：比例与增益都调
  RG-<m>    主模型去掉某一个变量对比例的作用（检验该变量的比例效应）
  H0 / HRG  习惯痕迹替代上一轮选择（Miller et al., 2019）：无调节 / 主模型调节——检验"稳定 → 惯性"是否只是习惯累积
"""
import sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

MODELS = {"M0": L.Spec(), "R": L.Spec(ratio=L.MODS), "G": L.Spec(gain=L.MODS), "RG": L.Spec(ratio=L.MODS, gain=L.MODS)}
for m in L.MODS:
    MODELS[f"RG-{m}"] = L.Spec(ratio=[x for x in L.MODS if x != m], gain=L.MODS)
MODELS["H0"] = L.Spec(habit=True)
MODELS["HRG"] = L.Spec(ratio=L.MODS, gain=L.MODS, habit=True)
for m in L.MODS:                                         # 习惯痕迹主模型的逐项检验（HRG 收敛后加入）
    MODELS[f"HRG-{m}"] = L.Spec(ratio=[x for x in L.MODS if x != m], gain=L.MODS, habit=True)
# 偏离对两个系统分开检验：偏离只作用于惯性（信念不随偏离变）/ 只作用于信念（惯性不随偏离变）
_nodev = [x for x in L.MODS if x != "dev"]
MODELS["HRG_dev仅惯性"] = L.Spec(ratio=_nodev, gain=_nodev, habit=True, honly=["dev"])
MODELS["HRG_dev仅信念"] = L.Spec(ratio=_nodev, gain=_nodev, habit=True, bonly=["dev"])
# 加法"重复推力"（稳定 / 偏离时人人被推向或推离自己的习惯方向），在 HRG 之上
MODELS["HRGP"] = L.Spec(ratio=L.MODS, gain=L.MODS, habit=True, push=["stab", "dev"])
MODELS["HRGP-stab"] = L.Spec(ratio=L.MODS, gain=L.MODS, habit=True, push=["dev"])
MODELS["HRGP-dev"] = L.Spec(ratio=L.MODS, gain=L.MODS, habit=True, push=["stab"])
DIR = L.OUT / "拟合"


def job(name):
    done = L.CKPT / f"fit_{name}.done"
    if done.exists():
        return name, "已完成（跳过）"
    spec = MODELS[name]
    logf = open(L.W / "日志" / f"fit_{name}.log", "a", encoding="utf-8")
    log = lambda s: (logf.write(s + "\n"), logf.flush())
    t0 = time.time()
    phi0 = np.r_[L.base_lambda(), np.zeros(spec.k - 1)]
    if spec.habit:
        phi0[-1] = 3.0                                   # α_H 起点 ≈ 0.95（接近原模型）
    log(f"开始 {name}: {spec.to_dict()}")
    X, phi, f, hist = L.joint_fit(spec, L.A_REAL, L.G_REAL, L.S_REAL, L.ATT, L.base_X(), phi0, log=log,
                                  ckpt=L.CKPT / f"轮_{name}.npz")
    z = L.run(X, spec, phi, L.A_REAL, L.G_REAL, L.S_REAL, L.ATT, out="z")
    sig, ll_sig = L.marginal_sigma(z, L.A_REAL)
    res = dict(name=name, spec=spec.to_dict(), shared=dict(zip(spec.names(), phi.tolist())), k_shared=spec.k,
               nll=float(f.sum()), nll_i=f, X=X, sigma=sig, loglik_sigma=ll_sig, history=hist, seconds=time.time() - t0)
    L.save_json(res, DIR / f"{name}.json")
    done.parent.mkdir(parents=True, exist_ok=True); done.write_text("ok")
    log(f"完成 {name}: NLL={f.sum():.3f} σ={sig:.4f} 边际 ll={ll_sig:.3f} 用时 {time.time() - t0:.0f}s")
    return name, f"NLL={f.sum():.2f}"


if __name__ == "__main__":
    L.load_std()
    (L.W / "日志").mkdir(exist_ok=True)
    order = sys.argv[1:] or (["M0", "RG", "R", "G", "H0", "HRG"] + [f"RG-{m}" for m in L.MODS])
    with Pool(4) as pool:
        for name, msg in pool.imap_unordered(job, order):
            print(name, msg, flush=True)
    if not sys.argv[1:]:
        (L.CKPT / "fit_models.done").write_text("ok")
