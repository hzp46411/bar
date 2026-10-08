"""第 7 步：补充汇总。
  (a) 各模型参数估计值的描述统计（中位数、四分位距）
  (b) 探索性补充（不在原方案内）：按 BIC 最优模型分组，比较各组的预测准确性

输出：结果/参数汇总.csv、结果/分组行为特征.csv、结果/分组预测准确性.csv、结果/分组预测准确性_逐人.csv
"""
import os

import numpy as np
import pandas as pd

from common import CAPACITY, CKPT_DIR, MODEL_NAMES, MODELS, RES_DIR, load_data

d = load_data()
fits = pd.read_csv(os.path.join(RES_DIR, "拟合结果.csv"))
best = fits.pivot(index="subject", columns="model", values="BIC").idxmin(axis=1)

# (a) 参数汇总
prow = []
for m in MODEL_NAMES:
    f = fits[fits.model == m]
    for p in MODELS[m].params:
        q = f[p].quantile([0.25, 0.5, 0.75])
        prow.append({"model": m, "param": p, "mean": f[p].mean(), "Q1": q[0.25], "median": q[0.5],
                     "Q3": q[0.75], "min": f[p].min(), "max": f[p].max(),
                     "n_at_lower_bound": int((f[p] <= MODELS[m].bounds[MODELS[m].params.index(p)][0] + 1e-4).sum()),
                     "n_at_upper_bound": int((f[p] >= MODELS[m].bounds[MODELS[m].params.index(p)][1] - 1e-4).sum())})
psum = pd.DataFrame(prow)
psum.to_csv(os.path.join(RES_DIR, "参数汇总.csv"), index=False)
print(psum.round(3).to_string(index=False))
# 混合模型中 BIC 归类为 MF / MB 的被试的 w
h = fits[fits.model == "Hybrid"].set_index("subject")
print("混合模型 w 的中位数（按 BIC 最优模型分组）:", h.w.groupby(best).median().round(3).to_dict())

# 各组的行为特征（条件差、斜率），以及超出 MB 可表达范围（斜率 < -0.2）的人数
sig = pd.read_csv(os.path.join(RES_DIR, "行为特征_真实.csv")).set_index("subject")
sig_by = sig.groupby(best)[["slope", "cond_diff"]].mean()
sig_by["n"] = best.value_counts()
sig_by.loc["全部", ["slope", "cond_diff"]] = sig[["slope", "cond_diff"]].mean().values
sig_by.loc["斜率<-0.2 的人数", "n"] = int((sig.slope < -0.2).sum())
sig_by.loc["斜率>0.2 的人数", "n"] = int((sig.slope > 0.2).sum())
sig_by.to_csv(os.path.join(RES_DIR, "分组行为特征.csv"))
print(sig_by.round(3))

# (b) 分组预测准确性

rows = []
for i in range(d.Y.shape[0]):
    v = d.valid[i]
    y, a = d.Y[i][v], d.A[v]
    rows.append({"subject": i + 1, "best_model": best.loc[i + 1],
                 "MAE": np.abs(y - a).mean(),
                 "crowded_hit_rate": np.mean((y > CAPACITY) == (a > CAPACITY))})
acc = pd.DataFrame(rows)
summary = acc.groupby("best_model").agg(n=("subject", "size"), MAE=("MAE", "mean"),
                                        MAE_sd=("MAE", "std"), hit=("crowded_hit_rate", "mean"))
# 参照：始终按“上次拥挤 → 这次不拥挤”反转预测的命中率
summary.loc["参照_始终反转", "hit"] = np.mean((d.A_prev <= CAPACITY) == (d.A > CAPACITY))
acc.to_csv(os.path.join(RES_DIR, "分组预测准确性_逐人.csv"), index=False)
summary.to_csv(os.path.join(RES_DIR, "分组预测准确性.csv"))
print(summary.round(3))

# MF 型 vs MB 型的 Welch t 检验（探索性）
from scipy.stats import ttest_ind

for col in ("MAE", "crowded_hit_rate"):
    t = ttest_ind(acc[acc.best_model == "MB"][col], acc[acc.best_model == "MF"][col], equal_var=False)
    print(f"MB 型 vs MF 型 {col}: t = {t.statistic:.2f}, p = {t.pvalue:.4f}")
open(os.path.join(CKPT_DIR, "07_supplement.done"), "w").close()
