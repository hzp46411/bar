"""第 4 步：参数恢复与模型恢复的模拟—再拟合（带检查点，可断点续跑）。

两种参数取值方案（regime）：
  empirical : 每个生成模型用 100 名真实被试在该模型下的拟合参数（以及各自的有效试次）
  uniform   : 在合理范围内均匀抽样 alpha~U(0,1), w~U(0,1), b~U(-3,3), sigma~U(2,8)
对每个 (regime, 生成模型, 合成被试) 模拟一次数据，然后用全部 4 个模型拟合。
  · 生成模型 = 拟合模型 的行 → 参数恢复
  · 4 个模型的 BIC/AIC 比较 → 模型恢复（混淆矩阵）

检查点：检查点/recovery_<regime>_<model>.csv，已存在的块直接跳过。
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import (CKPT_DIR, MODEL_NAMES, MODELS, RES_DIR, GridCache, fit_subject, load_data,
                    simulate, theta_from_row)

d = load_data()
cache = GridCache(d.A, d.A_prev)
fits = pd.read_csv(os.path.join(RES_DIR, "拟合结果.csv"))
N = d.Y.shape[0]
REGIMES = ("empirical", "uniform")
UNIFORM = {"alpha": (0, 1), "w": (0, 1), "b": (-3, 3), "sigma": (2, 8)}


def true_theta(regime, gen, i, rng):
    if regime == "empirical":
        row = fits[(fits.subject == i + 1) & (fits.model == gen)].iloc[0]
        return theta_from_row(gen, row)
    return np.array([rng.uniform(*UNIFORM[p]) for p in MODELS[gen].params])


def job(args):
    regime, gen, i = args
    seed = 1_000_000 * REGIMES.index(regime) + 10_000 * MODEL_NAMES.index(gen) + i
    rng = np.random.default_rng(seed)
    theta = true_theta(regime, gen, i, rng)
    valid = d.valid[i]
    y = simulate(gen, theta, d.A, d.A_prev, valid, rng)
    rows = []
    for fm in MODEL_NAMES:
        r = fit_subject(fm, y, valid, d.A, d.A_prev, cache, n_random=2, seed=seed)
        r.update(regime=regime, gen_model=gen, sim_id=i + 1, fit_model=r.pop("model"))
        for p, v in zip(MODELS[gen].params, theta):
            r[f"true_{p}"] = v
        rows.append(r)
    return rows


if __name__ == "__main__":
    for regime in REGIMES:
        for gen in MODEL_NAMES:
            path = os.path.join(CKPT_DIR, f"recovery_{regime}_{gen}.csv")
            if os.path.exists(path):
                print("跳过（已完成）", path, flush=True)
                continue
            with Pool(os.cpu_count()) as pool:
                out = pool.map(job, [(regime, gen, i) for i in range(N)], chunksize=2)
            df = pd.DataFrame([r for rows in out for r in rows])
            df.to_csv(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)
            print("完成", regime, gen, flush=True)
    allr = pd.concat([pd.read_csv(os.path.join(CKPT_DIR, f"recovery_{r}_{g}.csv"))
                      for r in REGIMES for g in MODEL_NAMES], ignore_index=True)
    allr.to_csv(os.path.join(RES_DIR, "恢复检验_模拟拟合.csv"), index=False)
    open(os.path.join(CKPT_DIR, "04_recovery_sim.done"), "w").close()
    sys.exit(0)
