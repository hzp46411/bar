"""酒吧问题（El Farol）出席人数预测的 MF / MB 学习模型：公共函数。

数据：allpredict.csv 第 1 行为每个试次的真实出席人数 A_t，其后 100 行为 100 名被试
在每个试次（看到 A_t 之前）报告的出席人数预测 y_t（0–100 的整数）。
allpredict_rt.csv 中 RT 缺失的试次是超时未作答，不计入似然。

分析窗口：最后 400 个试次（第 36–435 试次）。容量 C = 60。

模型（潜在预期 mu_t 只依赖参数和全组共用的 A 序列，与被试自己的作答无关）：
  M0 Baseline : mu_t = C + b                                    参数 b, sigma
  M1 MF       : mu_t = V_t + b,  V <- V + alpha (A_t - V)         参数 alpha, b, sigma
  M2 MB       : s_{t-1} = 1[A_{t-1} > C]
                mu_t = M_t(s_{t-1}) + b,
                M(s_{t-1}) <- M(s_{t-1}) + alpha (A_t - M(s_{t-1}))  参数 alpha, b, sigma
  M3 Hybrid   : mu_t = w M_t(s_{t-1}) + (1 - w) V_t + b          参数 alpha, w, b, sigma
作答模型（全部模型相同）：
  p(y_t) = (1 - EPS) * DiscreteNormal(y_t; mu_t, sigma) + EPS / 101
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "数据")
RES_DIR = os.path.join(ROOT, "结果")
FIG_DIR = os.path.join(ROOT, "图")
CKPT_DIR = os.path.join(ROOT, "检查点")
for _d in (RES_DIR, FIG_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)

CAPACITY = 60.0          # 酒吧容量 C
N_LAST = 400             # 只取最后 400 个试次
EPS = 0.01               # 固定失误率（与 0–100 均匀分布混合，抵抗离群作答）
N_OPTIONS = 101          # 作答取值 0..100
V0 = CAPACITY            # 所有潜变量在窗口起点初始化为容量（无偏先验）


# ----------------------------------------------------------------------------- 数据
@dataclass
class Data:
    A_all: np.ndarray      # (435,) 全部试次真实出席人数
    A: np.ndarray          # (400,) 窗口内 A_t
    A_prev: np.ndarray     # (400,) 窗口内 A_{t-1}（第一个窗口试次取第 35 试次）
    Y: np.ndarray          # (100, 400) 预测，缺失为 nan
    valid: np.ndarray      # (100, 400) bool，计入似然的试次
    trial_idx: np.ndarray  # (400,) 窗口试次的 1-based 编号


def load_data(data_dir: str = DATA_DIR) -> Data:
    pred = pd.read_csv(os.path.join(data_dir, "allpredict.csv"), header=None)
    rt = pd.read_csv(os.path.join(data_dir, "allpredict_rt.csv"), header=None)
    A_all = pred.iloc[0, 1:].to_numpy(float)
    Y_all = pred.iloc[1:, 1:].to_numpy(float)
    RT_all = rt.iloc[1:, 1:1 + A_all.size].to_numpy(float)
    T = A_all.size
    w = np.arange(T - N_LAST, T)                     # 0-based 列号 35..434
    Y = Y_all[:, w]
    valid = ~np.isnan(Y) & ~np.isnan(RT_all[:, w])   # 超时（RT 缺失）不计入似然
    return Data(A_all=A_all, A=A_all[w], A_prev=A_all[w - 1], Y=Y, valid=valid,
                trial_idx=w + 1)


# ----------------------------------------------------------------------------- 学习规则
def latent_mf(alpha, A):
    """MF：单一缓存值 V 的 delta 规则。alpha 可为标量或数组（向量化网格）。返回 (T, ...)"""
    alpha = np.asarray(alpha, float)
    V = np.full(alpha.shape, V0)
    out = np.empty((A.size,) + alpha.shape)
    for t in range(A.size):
        out[t] = V
        V = V + alpha * (A[t] - V)
    return out


def latent_mb(alpha, A, A_prev):
    """MB：以容量定义状态（上一试次是否拥挤），学习状态条件的转移预期 M(s)=E[A_t|s_{t-1}]。"""
    alpha = np.asarray(alpha, float)
    M = np.full((2,) + alpha.shape, V0)
    out = np.empty((A.size,) + alpha.shape)
    state = (A_prev > CAPACITY).astype(int)
    for t in range(A.size):
        s = state[t]
        out[t] = M[s]
        M[s] = M[s] + alpha * (A[t] - M[s])
    return out


# ----------------------------------------------------------------------------- 模型表
@dataclass
class Model:
    name: str
    label: str
    params: tuple
    bounds: tuple

    @property
    def k(self):
        return len(self.params)


MODELS = {
    "Baseline": Model("Baseline", "M0 无学习基线", ("b", "sigma"), ((-20, 20), (0.5, 30))),
    "MF": Model("MF", "M1 无模型 MF", ("alpha", "b", "sigma"), ((0, 1), (-20, 20), (0.5, 30))),
    "MB": Model("MB", "M2 基于模型 MB", ("alpha", "b", "sigma"), ((0, 1), (-20, 20), (0.5, 30))),
    "Hybrid": Model("Hybrid", "M3 混合 MF+MB", ("alpha", "w", "b", "sigma"),
                    ((0, 1), (0, 1), (-20, 20), (0.5, 30))),
}
MODEL_NAMES = list(MODELS)


def predicted_mean(model: str, theta, A, A_prev):
    """给定参数返回每个试次的预测均值 mu_t（不含作答噪声）。"""
    p = dict(zip(MODELS[model].params, theta))
    if model == "Baseline":
        return np.full(A.size, CAPACITY + p["b"])
    if model == "MF":
        return latent_mf(p["alpha"], A) + p["b"]
    if model == "MB":
        return latent_mb(p["alpha"], A, A_prev) + p["b"]
    if model == "Hybrid":
        mf = latent_mf(p["alpha"], A)
        mb = latent_mb(p["alpha"], A, A_prev)
        return p["w"] * mb + (1 - p["w"]) * mf + p["b"]
    raise ValueError(model)


# ----------------------------------------------------------------------------- 似然
def trial_loglik(y, mu, sigma):
    """离散正态（整数作答）与均匀失误的混合，返回逐试次对数似然。"""
    z_hi = (y + 0.5 - mu) / sigma
    z_lo = (y - 0.5 - mu) / sigma
    p = ndtr(z_hi) - ndtr(z_lo)
    return np.log((1 - EPS) * p + EPS / N_OPTIONS)


def nll(theta, model, y, valid, A, A_prev):
    mu = predicted_mean(model, theta, A, A_prev)
    sigma = theta[-1]
    return -trial_loglik(y[valid], mu[valid], sigma).sum()


# ----------------------------------------------------------------------------- 拟合
ALPHA_GRID = np.linspace(0, 1, 201)


class GridCache:
    """缓存 alpha 网格上的潜变量轨迹（只依赖 A 序列），用于给局部优化找好起点。"""

    def __init__(self, A, A_prev):
        self.mf = latent_mf(ALPHA_GRID, A)          # (T, G)
        self.mb = latent_mb(ALPHA_GRID, A, A_prev)  # (T, G)


def _grid_starts(model, y, valid, cache: GridCache, n_top=3):
    """高斯近似下的网格搜索：b、w 用最小二乘闭式解，sigma 用残差标准差。返回若干起点。"""
    yv = y[valid]
    n = yv.size
    if model == "Baseline":
        b = yv.mean() - CAPACITY
        return [np.array([b, max(yv.std(), 1.0)])]
    mf, mb = cache.mf[valid], cache.mb[valid]
    cands = []
    if model in ("MF", "MB"):
        lat = mf if model == "MF" else mb
        r = yv[:, None] - lat
        b = r.mean(0)
        s = np.sqrt(((r - b) ** 2).mean(0))
        score = n * np.log(s)
        for g in np.argsort(score)[:n_top]:
            cands.append(np.array([ALPHA_GRID[g], b[g], max(s[g], 1.0)]))
        return cands
    # Hybrid：y - mf = b + w (mb - mf)
    d = mb - mf
    r0 = yv[:, None] - mf
    dm, rm = d.mean(0), r0.mean(0)
    var_d = ((d - dm) ** 2).mean(0)
    cov = ((d - dm) * (r0 - rm)).mean(0)
    w = np.where(var_d > 1e-12, cov / np.maximum(var_d, 1e-12), 0.5)
    w = np.clip(w, 0, 1)
    b = rm - w * dm
    s = np.sqrt(((r0 - b - w * d) ** 2).mean(0))
    score = n * np.log(s)
    for g in np.argsort(score)[:n_top]:
        cands.append(np.array([ALPHA_GRID[g], w[g], b[g], max(s[g], 1.0)]))
    return cands


def fit_subject(model, y, valid, A, A_prev, cache: GridCache, n_random=2, seed=0):
    """最大似然估计：网格起点 + 随机起点，L-BFGS-B 局部优化，取最优。"""
    m = MODELS[model]
    rng = np.random.default_rng(seed)
    starts = _grid_starts(model, y, valid, cache)
    start_range = {"alpha": (0, 1), "w": (0, 1), "b": (-5, 5), "sigma": (2, 10)}
    for _ in range(n_random):
        starts.append(np.array([rng.uniform(*start_range[p]) for p in m.params]))
    best = None
    for x0 in starts:
        x0 = np.clip(x0, [lo for lo, _ in m.bounds], [hi for _, hi in m.bounds])
        res = minimize(nll, x0, args=(model, y, valid, A, A_prev), method="L-BFGS-B",
                       bounds=m.bounds)
        if best is None or res.fun < best.fun:
            best = res
    n = int(valid.sum())
    k = m.k
    return {
        "model": model,
        **{p: float(v) for p, v in zip(m.params, best.x)},
        "nll": float(best.fun),
        "n": n,
        "k": k,
        "AIC": float(2 * best.fun + 2 * k),
        "BIC": float(2 * best.fun + k * np.log(n)),
        "converged": bool(best.success),
    }


# ----------------------------------------------------------------------------- 模拟
def simulate(model, theta, A, A_prev, valid=None, rng=None):
    """按模型生成一名被试的预测序列（与真实实验相同的 A 序列；A 为外生）。"""
    rng = np.random.default_rng() if rng is None else rng
    mu = predicted_mean(model, theta, A, A_prev)
    sigma = theta[-1]
    y = np.clip(np.rint(rng.normal(mu, sigma)), 0, 100)
    lapse = rng.random(A.size) < EPS
    y[lapse] = rng.integers(0, 101, lapse.sum())
    if valid is not None:
        y = np.where(valid, y, np.nan)
    return y


def theta_from_row(model, row):
    return np.array([row[p] for p in MODELS[model].params], float)


# ----------------------------------------------------------------------------- 行为特征
def signatures(y, valid, A_prev):
    """模型无关的行为特征：
    slope   : y_t 对 (A_{t-1} - C) 的回归斜率（>0 追随趋势，<0 预期反转）
    cond_diff: 上一试次拥挤后 vs 不拥挤后的平均预测之差
    mean    : 平均预测
    """
    yv, av = y[valid], A_prev[valid]
    x = av - CAPACITY
    slope = np.polyfit(x, yv, 1)[0]
    crowded = av > CAPACITY
    return {"slope": slope,
            "cond_diff": yv[crowded].mean() - yv[~crowded].mean(),
            "mean": yv.mean()}
