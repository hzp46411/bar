"""第 6 步：作图（读取前面各步保存的结果，不重新计算）。"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from common import CAPACITY, FIG_DIR, MODEL_NAMES, MODELS, RES_DIR, load_data

# 中文字体
for fam in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei"):
    if any(fam in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [fam]
        break
plt.rcParams.update({
    "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9a9893", "axes.labelcolor": "#2b2a27", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "figure.dpi": 110, "savefig.dpi": 160,
})
# 分类色（已用 dataviz 校验器验证该顺序在相邻对上色盲可分）；颜色跟随模型，固定不变
COLOR = {"Baseline": "#eda100", "MF": "#2a78d6", "MB": "#eb6834", "Hybrid": "#1baf7a"}
LABEL = {m: MODELS[m].label for m in MODEL_NAMES}
INK, MUTED = "#2b2a27", "#7a7974"


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.close(fig)
    print("保存", name)


d = load_data()
sig = pd.read_csv(os.path.join(RES_DIR, "行为特征_真实.csv"))
desc = json.load(open(os.path.join(RES_DIR, "描述统计.json"), encoding="utf-8"))
fits = pd.read_csv(os.path.join(RES_DIR, "拟合结果.csv"))

# ---------------------------------------------------------------- 图 1：环境结构与行为特征
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [2.2, 1, 1]})
ax[0].plot(d.trial_idx, d.A, color="#2a78d6", lw=1.2)
ax[0].axhline(CAPACITY, color=INK, lw=1, ls="--")
ax[0].text(d.trial_idx[0], 77, "虚线：容量 C = 60", ha="left", va="top", color=INK)
ax[0].set(xlabel="试次", ylabel="真实出席人数 A_t",
          title=f"A. 出席人数（第 36–435 试次）  一阶自相关 = {desc['A_lag1_autocorr']:.2f}")
ax[1].hist(sig.slope, bins=np.linspace(-0.8, 0.8, 33), color="#2a78d6", edgecolor="#fcfcfb", linewidth=1)
null_sd = desc["subject_slope_sd_null_perm_mean"]
ax[1].axvspan(-1.96 * null_sd, 1.96 * null_sd, color="#9a9893", alpha=0.25, lw=0)
ax[1].text(1.96 * null_sd + 0.02, ax[1].get_ylim()[1] * 0.92, "置换零分布\n95% 范围", color=INK, fontsize=8, va="top")
ax[1].axvline(0, color=MUTED, lw=0.8)
ax[1].set(xlabel="预测对 (A_{t-1} − 60) 的斜率", ylabel="被试数",
          title="B. 个体斜率远比零分布分散")
ax[2].scatter(sig.slope, sig.cond_diff, s=22, color="#2a78d6", edgecolor="#fcfcfb", linewidth=0.8)
ax[2].axhline(0, color=MUTED, lw=0.8)
ax[2].axvline(0, color=MUTED, lw=0.8)
ax[2].text(0.03, 0.95, "右上：追随趋势（MF 式）", transform=ax[2].transAxes, ha="left", va="top", color=INK, fontsize=9)
ax[2].text(0.97, 0.05, "左下：预期反转（MB 式）", transform=ax[2].transAxes, ha="right", va="bottom", color=INK, fontsize=9)
ax[2].set(xlabel="斜率", ylabel="拥挤后 − 不拥挤后 的平均预测", title="C. 两个行为特征一致")
fig.tight_layout()
save(fig, "图1_环境与行为特征.png")

# ---------------------------------------------------------------- 图 2：参数分布
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
bins = np.linspace(0, 1, 21)
for m in ("MF", "MB", "Hybrid"):
    ax[0].hist(fits[fits.model == m].alpha, bins=bins, histtype="step", lw=2, color=COLOR[m], label=LABEL[m])
ax[0].set(xlabel="学习率 α", ylabel="被试数", title="A. 学习率分布（多数 α 很小）")
ax[0].legend(frameon=False)
h = fits[fits.model == "Hybrid"]
ax[1].hist(h.w, bins=np.linspace(0, 1, 11), color=COLOR["Hybrid"], edgecolor="#fcfcfb", linewidth=1.5)
ax[1].set(xlabel="MB 权重 w（混合模型）", ylabel="被试数", title="B. w 呈两端分布（0 = 纯 MF，1 = 纯 MB）")
for m in MODEL_NAMES:
    ax[2].hist(fits[fits.model == m].sigma, bins=np.linspace(0, 15, 31), histtype="step", lw=2,
               color=COLOR[m], label=LABEL[m])
ax[2].set(xlabel="作答噪声 σ", ylabel="被试数", title="C. 作答噪声")
ax[2].legend(frameon=False)
fig.tight_layout()
save(fig, "图2_参数分布.png")

# ---------------------------------------------------------------- 图 3：模型比较
summ = pd.read_csv(os.path.join(RES_DIR, "模型比较_汇总.csv")).set_index("model").loc[MODEL_NAMES]
bms = json.load(open(os.path.join(RES_DIR, "RFX_BMS.json"), encoding="utf-8"))
fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
x = np.arange(len(MODEL_NAMES))
names = ["M0\n基线", "M1\nMF", "M2\nMB", "M3\n混合"]
cols = [COLOR[m] for m in MODEL_NAMES]


def bars(a, vals, title, ylabel, fmt):
    a.bar(x, vals, color=cols, width=0.62)
    for xi, v in zip(x, vals):
        a.text(xi, v, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=9)
    a.set_xticks(x, names)
    a.set(title=title, ylabel=ylabel)
    a.grid(axis="x", visible=False)


bars(ax[0], summ["ΔBIC(相对最优)"].to_numpy(), "A. 总 BIC 差（越小越好）", "ΔBIC", "{:.0f}")
bars(ax[1], summ["BIC最优人数"].to_numpy(), "B. 逐被试 BIC 最优人数", "被试数", "{:.0f}")
bars(ax[2], summ["RFX期望频率"].to_numpy(), "C. RFX-BMS 人群频率（括号内 PXP）", "期望频率", "{:.2f}")
for t, v in zip(ax[2].texts, summ["PXP"].to_numpy()):
    t.set_text(t.get_text() + f"\n(PXP {v:.2f})")
ax[2].set_ylim(0, 0.5)
cvll = summ["CV样本外总对数似然"].to_numpy()
bars(ax[3], cvll - cvll.min(), "D. 样本外对数似然（相对最差）", "Δ 对数似然（越大越好）", "{:.0f}")
fig.tight_layout()
save(fig, "图3_模型比较.png")

# ---------------------------------------------------------------- 图 4：后验预测检验
ppc = pd.read_csv(os.path.join(RES_DIR, "后验预测.csv"))
fig, ax = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True)
lim = (-0.9, 0.9)
for a, m in zip(ax, MODEL_NAMES):
    p = ppc[ppc.model == m]
    a.plot(lim, lim, color=MUTED, lw=1, ls="--")
    a.scatter(p.slope, p.slope_sim, s=20, color=COLOR[m], edgecolor="#fcfcfb", linewidth=0.8)
    r = bms["后验预测"][m]["r(slope_obs, slope_sim)"]
    a.set(title=f"{LABEL[m]}   r = {r:.2f}", xlabel="真实斜率", xlim=lim, ylim=lim)
ax[0].set_ylabel("模型模拟斜率（50 次平均）")
fig.suptitle("后验预测检验：模型能否复现每个人对上一试次出席人数的依赖", y=1.02)
fig.tight_layout()
save(fig, "图4_后验预测.png")

# ---------------------------------------------------------------- 图 5–6：恢复检验
rec_path = os.path.join(RES_DIR, "恢复检验_模拟拟合.csv")
if os.path.exists(rec_path):
    R = pd.read_csv(rec_path)
    REGIME_TAGS = [(r, t) for r, t in (("empirical", "经验参数_全部被试"), ("uniform", "均匀抽样参数"),
                                       ("group", "经验参数_对应类型被试")) if r in set(R.regime)]
    prec = pd.read_csv(os.path.join(RES_DIR, "参数恢复.csv"))
    for regime, tag in REGIME_TAGS:
        panels = [(m, p) for m in MODEL_NAMES for p in MODELS[m].params]
        fig, ax = plt.subplots(3, 4, figsize=(14, 10))
        ax = ax.ravel()
        for a, (m, p) in zip(ax, panels):
            g = R[(R.regime == regime) & (R.gen_model == m) & (R.fit_model == m)]
            t, e = g[f"true_{p}"], g[p]
            lo, hi = min(t.min(), e.min()), max(t.max(), e.max())
            a.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
            a.scatter(t, e, s=14, color=COLOR[m], edgecolor="#fcfcfb", linewidth=0.6)
            r = prec[(prec.regime == regime) & (prec.model == m) & (prec.param == p)].pearson_r.iloc[0]
            a.set_title(f"{m} · {p}   r = {r:.2f}", fontsize=10)
            a.set_xlabel("真值", fontsize=9)
            a.set_ylabel("估计值", fontsize=9)
        for a in ax[len(panels):]:
            a.axis("off")
        fig.suptitle(f"参数恢复（{tag}，每个模型 100 名合成被试）", y=1.0)
        fig.tight_layout()
        save(fig, f"图5_参数恢复_{tag}.png")

    mr = pd.read_csv(os.path.join(RES_DIR, "模型恢复_混淆矩阵.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for a, (regime, tag) in zip(ax, REGIME_TAGS):
        cm = (mr[(mr.regime == regime) & (mr.criterion == "BIC")]
              .pivot(index="gen_model", columns="best_model", values="p_best_given_gen")
              .reindex(index=MODEL_NAMES, columns=MODEL_NAMES))
        a.imshow(cm.to_numpy(), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"]), vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                v = cm.iloc[i, j]
                a.text(j, i, f"{v:.2f}", ha="center", va="center", color="#ffffff" if v > 0.55 else INK)
        short = [MODELS[m].label.split(" ", 1)[1] for m in MODEL_NAMES]
        a.set_xticks(range(4), short)
        a.set_yticks(range(4), short)
        a.set(xlabel="BIC 判定的最优模型", ylabel="生成数据的模型", title=f"模型恢复：{tag}")
        a.grid(False)
    fig.tight_layout()
    save(fig, "图6_模型恢复.png")
