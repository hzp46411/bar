"""第 2 步：参数估计（逐被试最大似然，网格起点 + 随机起点 + L-BFGS-B）。

输出：结果/拟合结果.csv（每名被试 × 每个模型一行：参数、NLL、AIC、BIC）
"""
import os
from multiprocessing import Pool

import pandas as pd

from common import CKPT_DIR, MODEL_NAMES, RES_DIR, GridCache, fit_subject, load_data

d = load_data()
cache = GridCache(d.A, d.A_prev)


def job(args):
    i, model = args
    r = fit_subject(model, d.Y[i], d.valid[i], d.A, d.A_prev, cache, n_random=3, seed=i)
    r["subject"] = i + 1
    return r


if __name__ == "__main__":
    jobs = [(i, m) for m in MODEL_NAMES for i in range(d.Y.shape[0])]
    with Pool(os.cpu_count()) as pool:
        out = pool.map(job, jobs, chunksize=4)
    df = pd.DataFrame(out)
    cols = ["subject", "model", "alpha", "w", "b", "sigma", "nll", "n", "k", "AIC", "BIC", "converged"]
    df = df.reindex(columns=cols).sort_values(["model", "subject"])
    df.to_csv(os.path.join(RES_DIR, "拟合结果.csv"), index=False)
    print(df.groupby("model")[["alpha", "w", "b", "sigma", "nll", "BIC"]].median().round(3))
    print("未收敛：", int((~df.converged).sum()))
    open(os.path.join(CKPT_DIR, "02_fit.done"), "w").close()
