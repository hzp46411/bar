"""第 1 步：数据整理与模型无关的描述统计。

输出：
  结果/描述统计.json        环境结构（出席人数的均值、自相关、状态条件均值）与数据质量
  结果/行为特征_真实.csv    每名被试的行为特征（对上一试次出席人数的斜率等）
"""
import json
import os

import numpy as np
import pandas as pd

from common import CAPACITY, CKPT_DIR, RES_DIR, load_data, signatures

d = load_data()
A, Ap = d.A, d.A_prev
crowded_prev = Ap > CAPACITY

env = {
    "n_subjects": int(d.Y.shape[0]),
    "window_trials": [int(d.trial_idx[0]), int(d.trial_idx[-1])],
    "A_mean": float(A.mean()),
    "A_sd": float(A.std(ddof=1)),
    "A_frac_crowded(>60)": float((A > CAPACITY).mean()),
    "A_lag1_autocorr": float(np.corrcoef(A[1:], A[:-1])[0, 1]),
    "A_lag2_autocorr": float(np.corrcoef(A[2:], A[:-2])[0, 1]),
    "E[A_t | A_{t-1}>60]": float(A[crowded_prev].mean()),
    "E[A_t | A_{t-1}<=60]": float(A[~crowded_prev].mean()),
    "P(crowded_t | crowded_{t-1})": float((A[crowded_prev] > CAPACITY).mean()),
    "P(crowded_t | not crowded_{t-1})": float((A[~crowded_prev] > CAPACITY).mean()),
    "excluded_trials_timeout_or_missing": int((~d.valid).sum()),
    "included_trials": int(d.valid.sum()),
}

rows = []
for i in range(d.Y.shape[0]):
    s = signatures(d.Y[i], d.valid[i], Ap)
    yv = d.Y[i][d.valid[i]]
    s.update(subject=i + 1, sd=yv.std(ddof=1), n_valid=int(d.valid[i].sum()),
             mae_to_actual=np.abs(yv - A[d.valid[i]]).mean())
    rows.append(s)
sig = pd.DataFrame(rows)[["subject", "n_valid", "mean", "sd", "slope", "cond_diff", "mae_to_actual"]]
sig.to_csv(os.path.join(RES_DIR, "行为特征_真实.csv"), index=False)

# 斜率是否显著偏离 0、个体差异是否超过抽样误差（由置换 A_{t-1} 的零分布估计）
rng = np.random.default_rng(0)
null_sd = []
for _ in range(200):
    perm = rng.permutation(Ap)
    null_sd.append(np.std([signatures(d.Y[i], d.valid[i], perm)["slope"] for i in range(d.Y.shape[0])]))
t_slope = sig.slope.mean() / (sig.slope.std(ddof=1) / np.sqrt(len(sig)))
env.update({
    "subject_slope_mean": float(sig.slope.mean()),
    "subject_slope_sd": float(sig.slope.std(ddof=1)),
    "subject_slope_t_vs_0": float(t_slope),
    "subject_slope_frac_negative": float((sig.slope < 0).mean()),
    "subject_slope_sd_null_perm_mean": float(np.mean(null_sd)),
    "subject_slope_sd_null_perm_95pct": float(np.percentile(null_sd, 95)),
    "subject_mean_pred_mean": float(sig["mean"].mean()),
    "subject_pred_sd_median": float(sig.sd.median()),
    "subject_mae_mean": float(sig.mae_to_actual.mean()),
    "mae_constant60": float(np.abs(A - CAPACITY).mean()),
})
with open(os.path.join(RES_DIR, "描述统计.json"), "w", encoding="utf-8") as f:
    json.dump(env, f, ensure_ascii=False, indent=2)
for k, v in env.items():
    print(f"{k}: {v}")
open(os.path.join(CKPT_DIR, "01_describe.done"), "w").close()
