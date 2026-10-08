# -*- coding: utf-8 -*-
"""第 4 层报告用图：图 6 两个系统的权重如何被各调节通路改变；图 7 稳定期的残差重复；图 8 经验的学习曲线。"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984"
OUT = L.OUT / "图"; OUT.mkdir(parents=True, exist_ok=True)
J = lambda n: json.loads((L.OUT / n).read_text(encoding="utf-8"))

# ---------- 图 6 ----------
sh = json.loads((L.OUT / "拟合" / "HRGPR.json").read_text(encoding="utf-8"))["shared"]
eff = lambda m: (sh[f"θG_{m}"] + sh[f"θR_{m}"] / 2, sh[f"θG_{m}"] - sh[f"θR_{m}"] / 2)
rows = [("经验（轮次）", *eff("time")), ("上一轮大偏离", *eff("dev")), ("局面稳定（权重）", *eff("stab")),
        ("信念建议近期成绩", sh["θB_relB"], 0.0), ("习惯建议近期成绩", 0.0, sh["θH_relH"])]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
y = np.arange(len(rows))[::-1]
ax.axvline(0, color=MUTED, lw=0.8)
for yi, (lab, b, h) in zip(y, rows):
    ax.plot([b, h], [yi, yi], color="#d9d8d3", lw=2, zorder=1)
    ax.scatter(b, yi, s=48, color=BLUE, zorder=3, edgecolor="white", lw=1.5)
    ax.scatter(h, yi, s=48, color=ORANGE, zorder=3, edgecolor="white", lw=1.5)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("每 1 SD 调节信号，权重的对数变化（0.1 ≈ 权重 ×1.105）")
ax.scatter([], [], color=BLUE, label="信念权重"); ax.scatter([], [], color=ORANGE, label="惯性权重")
ax.legend(frameon=False, loc="lower right", fontsize=8)
ax.text(0.99, 1.02, f"另：局面稳定的加法推力（推向习惯）ψ = {sh['ψ_stab']:+.3f}", transform=ax.transAxes, ha="right", fontsize=8, color=INK)
fig.tight_layout(); fig.savefig(OUT / "图6_调节通路.png", dpi=200); plt.close(fig)

# ---------- 图 7 ----------
R = J("残差重复率.json")["状态持续轮数"]
fig, ax = plt.subplots(figsize=(4.6, 2.8))
x = np.arange(len(R)); d = np.array([r["差值"] for r in R]) * 100; se = np.array([r["差值SE"] for r in R]) * 100
ax.axhline(0, color=MUTED, lw=0.8)
ax.bar(x, d, width=0.55, color=[ORANGE if v > 0 else BLUE for v in d])
ax.errorbar(x, d, yerr=1.96 * se, fmt="none", ecolor=INK, lw=1, capsize=3)
for xi, v in zip(x, d):
    ax.text(xi + 0.31, v / 2, f"{v:+.1f}", ha="left", va="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels([r["组"] for r in R]); ax.set_xlabel("同一拥挤状态已持续的轮数")
ax.set_ylabel("实际重复 − 基线模型预测（百分点）")
fig.tight_layout(); fig.savefig(OUT / "图7_稳定期残差重复.png", dpi=200); plt.close(fig)

# ---------- 图 8 ----------
B = J("轮次形状.json")["区块"]
fig, ax = plt.subplots(figsize=(5.2, 2.9))
xb = np.arange(len(B)) * 50 + 25
ax.plot(xb, [b["信念倍数"] for b in B], "-o", color=BLUE, lw=2, ms=5, label="信念权重")
ax.plot(xb, [b["惯性倍数"] for b in B], "-o", color=ORANGE, lw=2, ms=5, label="惯性权重")
ax.axhline(1, color=MUTED, lw=0.8, ls=":")
ax.set_xlabel("分析中的轮次（第 1 轮 = 实验第 36 轮），每点为 50 轮区块"); ax.set_ylabel("相对第 1–50 轮的倍数")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "图8_经验学习曲线.png", dpi=200); plt.close(fig)
print("图已保存：", sorted(p.name for p in OUT.glob("*.png")))
