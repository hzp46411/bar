"""第 5 步：参数恢复与模型恢复的统计汇总（读取第 4 步的模拟—再拟合结果）。

参数恢复：生成模型 = 拟合模型 的行；真值 vs 估计值的 Pearson / Spearman r、偏差、RMSE，
          以及“真值 i × 估计值 j”的交叉相关（检查参数间的权衡 / 可辨识性）。
模型恢复：混淆矩阵 p(BIC 最优 = 拟合模型 | 生成模型) 与反演矩阵 p(生成模型 | BIC 最优)，AIC 同理。

输出：结果/参数恢复.csv、结果/参数恢复_交叉相关.csv、结果/模型恢复_混淆矩阵.csv、结果/模型恢复.json
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import CKPT_DIR, MODEL_NAMES, MODELS, RES_DIR

R = pd.read_csv(os.path.join(RES_DIR, "恢复检验_模拟拟合.csv"))

# ---------------------------------------------------------------- 参数恢复
rows, cross = [], []
for (regime, gen), g in R[R.gen_model == R.fit_model].groupby(["regime", "gen_model"]):
    params = MODELS[gen].params
    for p in params:
        t, e = g[f"true_{p}"].to_numpy(), g[p].to_numpy()
        rows.append({"regime": regime, "model": gen, "param": p,
                     "pearson_r": pearsonr(t, e)[0], "spearman_r": spearmanr(t, e)[0],
                     "bias": float(np.mean(e - t)), "RMSE": float(np.sqrt(np.mean((e - t) ** 2))),
                     "true_sd": float(np.std(t)), "n": len(t)})
        for q in params:
            cross.append({"regime": regime, "model": gen, "true": p, "estimated": q,
                          "r": pearsonr(g[f"true_{p}"], g[q])[0]})
prec = pd.DataFrame(rows)
prec.to_csv(os.path.join(RES_DIR, "参数恢复.csv"), index=False)
pd.DataFrame(cross).to_csv(os.path.join(RES_DIR, "参数恢复_交叉相关.csv"), index=False)

# ---------------------------------------------------------------- 模型恢复
out, conf_rows = {}, []
for regime, g in R.groupby("regime"):
    out[regime] = {}
    for crit in ("BIC", "AIC"):
        piv = g.pivot_table(index=["gen_model", "sim_id"], columns="fit_model", values=crit)[MODEL_NAMES]
        best = piv.idxmin(axis=1).rename("best").reset_index()
        conf = pd.crosstab(best.gen_model, best.best, normalize="index").reindex(
            index=MODEL_NAMES, columns=MODEL_NAMES, fill_value=0)
        counts = pd.crosstab(best.gen_model, best.best).reindex(index=MODEL_NAMES, columns=MODEL_NAMES, fill_value=0)
        inv = (counts / counts.sum(0).replace(0, np.nan)).fillna(0)  # p(生成 | 最优)
        out[regime][crit] = {"confusion p(best|gen)": conf.round(4).to_dict(orient="index"),
                             "inversion p(gen|best)": inv.round(4).to_dict(orient="index"),
                             "diag_mean": float(np.mean(np.diag(conf)))}
        for gm in MODEL_NAMES:
            for fm in MODEL_NAMES:
                conf_rows.append({"regime": regime, "criterion": crit, "gen_model": gm, "best_model": fm,
                                  "p_best_given_gen": conf.loc[gm, fm], "p_gen_given_best": inv.loc[gm, fm]})
    # MF vs MB 二选一（只在两个关键模型之间判别）
    piv = g.pivot_table(index=["gen_model", "sim_id"], columns="fit_model", values="BIC")[["MF", "MB"]]
    sub = piv.reset_index()
    sub = sub[sub.gen_model.isin(["MF", "MB"])]
    sub["best"] = np.where(sub.MF < sub.MB, "MF", "MB")
    out[regime]["MF_vs_MB_BIC"] = pd.crosstab(sub.gen_model, sub.best, normalize="index").round(4).to_dict(orient="index")

pd.DataFrame(conf_rows).to_csv(os.path.join(RES_DIR, "模型恢复_混淆矩阵.csv"), index=False)
with open(os.path.join(RES_DIR, "模型恢复.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

pd.set_option("display.width", 200)
print(prec.round(3).to_string(index=False))
for regime in out:
    print(f"\n== {regime}  BIC 混淆矩阵 p(最优 | 生成)")
    print(pd.DataFrame(out[regime]["BIC"]["confusion p(best|gen)"]).T.reindex(index=MODEL_NAMES)[MODEL_NAMES].round(2))
    print("MF vs MB:", out[regime]["MF_vs_MB_BIC"])
open(os.path.join(CKPT_DIR, "05_recovery_analysis.done"), "w").close()
