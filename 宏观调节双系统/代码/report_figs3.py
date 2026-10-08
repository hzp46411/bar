# -*- coding: utf-8 -*-
"""第三环报告用图：图 9 换人率随局面稳定的剖面；图 10 两个通道的宏观足迹（双重分离）；图 11 剂量—反应。"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import abm_arb as Ab
import third_arrow as TA
import stab_shape_loop as SS

plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})
BLUE, ORANGE, INK, MUTED, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984", "#c9c8c3"
OUT = L.OUT / "图"; OUT.mkdir(parents=True, exist_ok=True)
J = lambda n: json.loads((L.OUT / n).read_text(encoding="utf-8"))
M = "HRGPRS"

# ---------- 图 9 ----------
sh = Ab.FIT(M)["shared"]
var = {"完整模型": {}, "关掉惯性通道的调节": {"phi_set": TA.channel_off("惯性", sh)},
       "稳定按分级估计": {"zero_mods": ["stab"], "stab_terms": SS.terms("分类")}}
prof = {lab: np.nanmean([SS.swprof(*Ab.simulate(M, mod, 880000 + s)[:2]) for s in range(200)], 0) for lab, mod in var.items()}
real = SS.swprof(L.ATT, L.A_REAL)
fig, ax = plt.subplots(figsize=(5.0, 3.0))
x = np.arange(5)
ax.bar(x, real, width=0.55, color=GREY, label="真实数据")
for lab, col, ls in (("完整模型", BLUE, "-"), ("稳定按分级估计", BLUE, ":"), ("关掉惯性通道的调节", ORANGE, "-")):
    ax.plot(x, prof[lab], ls, color=col, marker="o", ms=4, lw=1.8, label=lab)
ax.set_xticks(x); ax.set_xticklabels(["刚翻转", "1", "2", "3", "≥4"]); ax.set_xlabel("同一拥挤状态已持续的轮数")
ax.set_ylabel("本轮换选的人的比例"); ax.set_ylim(0.2, 0.5)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "图9_换人节律.png", dpi=200); plt.close(fig)

# ---------- 图 10 ----------
K = J(f"第三环_敲除_{M}.json")["变体"]
rows = [("人数 SD", "sd"), ("人数 ACF1", "acf1"), ("负反馈强度 φ0", "φ0"), ("大偏离饱和 φn", "φn"), ("波动聚集", "sq_acf1"),
        ("长期稳定后的回调 φs", "φs"), ("换人率波动", "换人率_sd"), ("换人对偏离的反应", "换人_偏离斜率"), ("换人率自相关", "换人率_acf1"),
        ("群体效率", "效率"), ("收益不平等", "不平等")]
fig, ax = plt.subplots(figsize=(6.0, 3.8))
y = np.arange(len(rows))[::-1]
b = [K["只关信念通道"][k]["差(SD)"] for _, k in rows]; h = [K["只关惯性通道"][k]["差(SD)"] for _, k in rows]
ax.barh(y + 0.18, b, height=0.34, color=BLUE, label="只关信念通道的调节")
ax.barh(y - 0.18, h, height=0.34, color=ORANGE, label="只关惯性通道的调节")
ax.axvline(0, color=MUTED, lw=0.8)
for v in (-2, 2):
    ax.axvline(v, color=GREY, lw=0.6, ls=":")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("相对完整模型的变化（以完整模型模拟 SD 为单位）")
for yy in (4.5, 1.5):
    ax.axhline(yy, color=GREY, lw=0.6)
for yy, txt in ((10.45, "水平（多少人去）"), (4.42, "结构（谁换、何时换）"), (1.42, "福利")):
    ax.text(-6.5, yy, txt, ha="left", va="top", fontsize=8, color=MUTED)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.9))
fig.tight_layout(); fig.savefig(OUT / "图10_两个通道的宏观足迹.png", dpi=200); plt.close(fig)

# ---------- 图 11 ----------
D = J(f"第三环_剂量_{M}.json"); obs = D["真实"]
fs = [0.0, 0.5, 1.0, 2.0, 3.0]
fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
for ax, key, lab in ((axs[0], "换人率_sd", "换人率波动（SD）"), (axs[1], "φs", "长期稳定后的回调 φs")):
    for nm, col, txt in (("ψ_stab", ORANGE, "稳定推力（惯性通道）"), ("可靠性", BLUE, "可靠性仲裁")):
        ax.plot(fs, [D["变体"][f"{nm}×{f}"][key]["均值"] for f in fs], "-o", color=col, ms=4, lw=1.8, label=txt)
    ax.axhline(obs[key], color=INK, lw=0.9, ls="--"); ax.text(3.0, obs[key], " 真实", va="center", fontsize=8)
    ax.axvline(1.0, color=GREY, lw=0.6, ls=":")
    ax.set_xlabel("通路强度（× 估计值）"); ax.set_title(lab, fontsize=9)
axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "图11_剂量反应.png", dpi=200); plt.close(fig)
L.save_json(dict(换人剖面={k: list(map(float, v)) for k, v in prof.items()}, 真实=real), L.OUT / "图9_数据.json")
print("ok", {k: np.round(v, 3).tolist() for k, v in prof.items()}, np.round(real, 3))
