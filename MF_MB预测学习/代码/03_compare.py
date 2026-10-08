"""第 3 步：模型比较。

  (a) 固定效应：总 AIC / BIC；逐被试最优模型计数
  (b) 随机效应贝叶斯模型选择（RFX-BMS；Stephan et al., 2009; Rigoux et al., 2014）：
      以 -BIC/2 近似对数模型证据，给出期望模型频率、超越概率 XP、受保护超越概率 PXP
  (c) 时间前向交叉验证：用窗口前 200 试次估参，在后 200 试次计算样本外对数似然
  (d) 后验预测检验：用每人的拟合参数模拟 50 次，看模型能否复现行为特征
      （对上一试次出席人数的斜率、拥挤后 vs 不拥挤后的预测差）

输出：结果/模型比较_汇总.csv、结果/RFX_BMS.json、结果/交叉验证.csv、结果/后验预测.csv
"""
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln, logsumexp

from common import (CKPT_DIR, MODEL_NAMES, MODELS, RES_DIR, GridCache, fit_subject, load_data,
                    nll, signatures, simulate, theta_from_row)

d = load_data()
cache = GridCache(d.A, d.A_prev)
fits = pd.read_csv(os.path.join(RES_DIR, "拟合结果.csv"))
N = d.Y.shape[0]
HALF = 200


def rfx_bms(lme, n_samples=1_000_000, seed=0):
    """VB 随机效应 BMS。lme: (N, K) 对数模型证据。"""
    n, k = lme.shape
    a0 = np.ones(k)
    a = a0.copy()
    for _ in range(10_000):
        log_u = lme + digamma(a) - digamma(a.sum())
        g = np.exp(log_u - logsumexp(log_u, axis=1, keepdims=True))
        a_new = a0 + g.sum(0)
        if np.max(np.abs(a_new - a)) < 1e-10:
            a = a_new
            break
        a = a_new
    exp_r = a / a.sum()
    rng = np.random.default_rng(seed)
    r = rng.dirichlet(a, n_samples)
    xp = np.bincount(r.argmax(1), minlength=k) / n_samples
    # 贝叶斯遗漏风险 BOR：RFX 模型 vs “所有模型频率相等”零模型的自由能
    e_log_r = digamma(a) - digamma(a.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        glogg = np.where(g > 0, g * np.log(g), 0.0)
    kl = (gammaln(a.sum()) - gammaln(a).sum() - gammaln(a0.sum()) + gammaln(a0).sum()
          + ((a - a0) * e_log_r).sum())
    f1 = (g * (lme + e_log_r)).sum() - glogg.sum() - kl
    f0 = (logsumexp(lme, axis=1) - np.log(k)).sum()
    bor = 1 / (1 + np.exp(np.clip(f1 - f0, -700, 700)))
    pxp = xp * (1 - bor) + bor / k
    return {"alpha": a, "exp_r": exp_r, "xp": xp, "bor": bor, "pxp": pxp, "posterior_g": g}


def cv_job(args):
    i, model = args
    train = d.valid[i] & (np.arange(d.A.size) < HALF)
    test = d.valid[i] & (np.arange(d.A.size) >= HALF)
    r = fit_subject(model, d.Y[i], train, d.A, d.A_prev, cache, n_random=2, seed=1000 + i)
    theta = theta_from_row(model, r)
    return {"subject": i + 1, "model": model,
            "test_loglik": -nll(theta, model, d.Y[i], test, d.A, d.A_prev),
            "n_test": int(test.sum())}


def ppc_job(args):
    i, model, n_sim = args
    row = fits[(fits.subject == i + 1) & (fits.model == model)].iloc[0]
    theta = theta_from_row(model, row)
    rng = np.random.default_rng(10_000 * (MODEL_NAMES.index(model) + 1) + i)
    sims = [signatures(simulate(model, theta, d.A, d.A_prev, d.valid[i], rng), d.valid[i], d.A_prev)
            for _ in range(n_sim)]
    return {"subject": i + 1, "model": model,
            "slope_sim": np.mean([s["slope"] for s in sims]),
            "cond_diff_sim": np.mean([s["cond_diff"] for s in sims])}


if __name__ == "__main__":
    out = {}
    # (a) 固定效应
    bic = fits.pivot(index="subject", columns="model", values="BIC")[MODEL_NAMES]
    aic = fits.pivot(index="subject", columns="model", values="AIC")[MODEL_NAMES]
    summ = pd.DataFrame({
        "参数个数": [MODELS[m].k for m in MODEL_NAMES],
        "总NLL": fits.groupby("model").nll.sum()[MODEL_NAMES].values,
        "总AIC": aic.sum().values,
        "总BIC": bic.sum().values,
        "ΔBIC(相对最优)": (bic.sum() - bic.sum().min()).values,
        "BIC最优人数": bic.idxmin(axis=1).value_counts().reindex(MODEL_NAMES, fill_value=0).values,
        "AIC最优人数": aic.idxmin(axis=1).value_counts().reindex(MODEL_NAMES, fill_value=0).values,
    }, index=MODEL_NAMES)

    # (b) RFX-BMS：四模型，以及只比较 MF vs MB
    bms4 = rfx_bms(-bic.to_numpy() / 2)
    bms2 = rfx_bms(-bic[["MF", "MB"]].to_numpy() / 2)
    summ["RFX期望频率"] = bms4["exp_r"]
    summ["XP"] = bms4["xp"]
    summ["PXP"] = bms4["pxp"]
    out["四模型"] = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
                  for k, v in bms4.items() if k != "posterior_g"}
    out["四模型"]["models"] = MODEL_NAMES
    out["MF_vs_MB"] = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
                       for k, v in bms2.items() if k != "posterior_g"}
    out["MF_vs_MB"]["models"] = ["MF", "MB"]
    pd.DataFrame(bms4["posterior_g"], columns=MODEL_NAMES,
                 index=bic.index).to_csv(os.path.join(RES_DIR, "RFX_逐人后验.csv"))

    # (c) 交叉验证
    with Pool(os.cpu_count()) as pool:
        cv = pd.DataFrame(pool.map(cv_job, [(i, m) for m in MODEL_NAMES for i in range(N)], chunksize=4))
    cv.to_csv(os.path.join(RES_DIR, "交叉验证.csv"), index=False)
    cvw = cv.pivot(index="subject", columns="model", values="test_loglik")[MODEL_NAMES]
    summ["CV样本外总对数似然"] = cvw.sum().values
    summ["CV最优人数"] = cvw.idxmax(axis=1).value_counts().reindex(MODEL_NAMES, fill_value=0).values

    # (d) 后验预测检验
    with Pool(os.cpu_count()) as pool:
        ppc = pd.DataFrame(pool.map(ppc_job, [(i, m, 50) for m in MODEL_NAMES for i in range(N)], chunksize=4))
    sig = pd.read_csv(os.path.join(RES_DIR, "行为特征_真实.csv"))
    ppc = ppc.merge(sig[["subject", "slope", "cond_diff"]], on="subject")
    ppc.to_csv(os.path.join(RES_DIR, "后验预测.csv"), index=False)
    ppc_summary = {}
    for m in MODEL_NAMES:
        p = ppc[ppc.model == m]
        ppc_summary[m] = {
            "r(slope_obs, slope_sim)": float(np.corrcoef(p.slope, p.slope_sim)[0, 1]) if p.slope_sim.std() > 0 else None,
            "r(cond_obs, cond_sim)": float(np.corrcoef(p.cond_diff, p.cond_diff_sim)[0, 1]) if p.cond_diff_sim.std() > 0 else None,
            "RMSE_slope": float(np.sqrt(((p.slope - p.slope_sim) ** 2).mean())),
            "RMSE_cond_diff": float(np.sqrt(((p.cond_diff - p.cond_diff_sim) ** 2).mean())),
        }
    out["后验预测"] = ppc_summary
    summ["PPC斜率r"] = [ppc_summary[m]["r(slope_obs, slope_sim)"] for m in MODEL_NAMES]
    summ["PPC斜率RMSE"] = [ppc_summary[m]["RMSE_slope"] for m in MODEL_NAMES]

    summ.index.name = "model"
    summ.to_csv(os.path.join(RES_DIR, "模型比较_汇总.csv"))
    with open(os.path.join(RES_DIR, "RFX_BMS.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    pd.set_option("display.width", 200)
    print(summ.round(3).T)
    print("MF vs MB:", {k: out["MF_vs_MB"][k] for k in ("exp_r", "xp", "pxp", "bor")})
    open(os.path.join(CKPT_DIR, "03_compare.done"), "w").close()
