# -*- coding: utf-8 -*-
"""预测分析的图：P1 选择对自己预测的依赖（台阶）；P2 冲突时跟随习惯的比例随宏观状态变化；P3 信念类型的汇聚效度；P4 共同信念冲击与人数；
   P5 被习惯推去 / 推留时报告分布的差（工具变量）；P6 合理化的个体差异与工具变量的交叉验证"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pred_lib as PL
plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})
BLUE, ORANGE, INK, MUTED, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984", "#d9d8d3"
J = lambda f: json.loads((PL.OUT / f).read_text(encoding="utf-8"))
OUT = PL.OUT / "图"; OUT.mkdir(parents=True, exist_ok=True)
P, A, N, OK = PL.P, PL.A, PL.N, PL.OK

# ---------- P1 ----------
vals = np.arange(50, 72); rate = []; cnt = []
for v in vals:
    m = OK & (P == v); rate.append(A[m].mean() if m.sum() >= 100 else np.nan); cnt.append(m.sum())
fig, ax = plt.subplots(figsize=(5.4, 3.0))
ax.axvline(60.5, color=MUTED, lw=0.8, ls=":")
ax.text(60.7, 0.95, "容量 60", fontsize=8, color=MUTED)
ax.plot(vals, rate, "-o", color=BLUE, ms=4, lw=1.8)
ax.set_xlabel("自己预测的本轮人数（含自己；每个值 ≥ 100 次）"); ax.set_ylabel("实际去的比例"); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(OUT / "图P1_选择依赖预测的台阶.png", dpi=200); plt.close(fig)

# ---------- P2 ----------
d = J("s3_不依赖模型的双系统检验.json")
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
s = d["按稳定分组（冲突情境）"]
axs[0].plot([r["稳定"] for r in s], [r["跟随习惯的比例"] for r in s], "-o", color=ORANGE, ms=4, lw=1.8)
axs[0].set_xticks(range(5)); axs[0].set_xticklabels(["刚翻转", "1", "2", "3", "≥4"]); axs[0].set_xlabel("同一状态已持续的轮数")
s = d["按偏离分组（冲突情境）"]
axs[1].plot(range(4), [r["跟随习惯的比例"] for r in s], "-o", color=ORANGE, ms=4, lw=1.8)
axs[1].set_xticks(range(4)); axs[1].set_xticklabels(["0–2", "3–5", "6–8", "≥9"]); axs[1].set_xlabel("上一轮偏离 60 的人数")
s = d["按轮次分组（冲突情境）"]
axs[2].plot(range(4), [r["跟随习惯的比例"] for r in s], "-o", color=ORANGE, ms=4, lw=1.8)
axs[2].set_xticks(range(4)); axs[2].set_xticklabels(["1–100", "101–200", "201–300", "301–400"]); axs[2].set_xlabel("轮次")
axs[0].set_ylabel("违背自己的预测、\n跟随习惯的比例"); axs[0].set_ylim(0.2, 0.5)
fig.tight_layout(); fig.savefig(OUT / "图P2_冲突时跟随习惯.png", dpi=200); plt.close(fig)

# ---------- P3 ----------
d = J("s4_信念系统的直接测量.json"); sc = np.array(d["逐人符号系数"]); beta = PL.model_states("HRGPR")["beta"]
agree = np.sign(sc) == np.sign(beta)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.axhline(0, color=MUTED, lw=0.7); ax.axvline(0, color=MUTED, lw=0.7)
ax.scatter(np.clip(beta, -16, 16)[agree], sc[agree], s=18, color=BLUE, label=f"类型一致（{agree.sum()} 人）", edgecolor="white", lw=0.5)
ax.scatter(np.clip(beta, -16, 16)[~agree], sc[~agree], s=18, color=ORANGE, label=f"类型不一致（{(~agree).sum()} 人）", edgecolor="white", lw=0.5)
ax.set_xlabel("由选择估计的 BBL β（> 0 外推；截断在 ±16）"); ax.set_ylabel("由预测估计：上一轮挤之后\n预测多报的人数（> 0 外推）")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "图P3_信念类型汇聚.png", dpi=200); plt.close(fig)

# ---------- P4 ----------
d = J("s5_群体层面.json"); eb = np.array(d["ebar"], float)
Pm = np.nanmean(P, 0)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
axs[0].scatter(Pm[1:], N[1:], s=8, color=BLUE, alpha=0.6, edgecolor="none")
axs[0].set_xlabel("本轮平均预测（人）"); axs[0].set_ylabel("本轮实际人数"); axs[0].set_title("原始：r = −0.51（含组成效应）", fontsize=9)
axs[1].scatter(eb[1:], N[1:], s=8, color=ORANGE, alpha=0.6, edgecolor="none")
b = np.polyfit(eb[1:], N[1:], 1); xx = np.linspace(np.nanmin(eb), np.nanmax(eb), 10); axs[1].plot(xx, np.polyval(b, xx), color=INK, lw=1.2)
axs[1].set_xlabel("共同信念残差 ē_t（去掉组成效应，人）"); axs[1].set_title("去掉组成效应：b = −2.1（t = −3.9）", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "图P4_共同信念冲击.png", dpi=200); plt.close(fig)
# ---------- P5 ----------
d = J("s9b_γ稳健性与报告分布.json")["B 报告分布"]; dd = J("s9b_γ稳健性与报告分布.json")
k = np.array(d["k"]); y = np.array(d["顺从者_去减留_累积差"]); se = np.array(d["se"])
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.axvline(60.5, color=MUTED, lw=0.8, ls=":"); ax.axhline(0, color=MUTED, lw=0.7)
ax.fill_between(k, y - 1.96 * se, y + 1.96 * se, color=BLUE, alpha=0.15, lw=0)
ax.plot(k, y, "-o", color=BLUE, ms=3.5, lw=1.8, label="数据：被习惯推去 − 推留（工具变量）")
fl = dd["B 类别翻转的最佳拟合"]; sh = dd["B 恒定平移的最佳拟合"]
ax.plot(k, fl["预测"], color=ORANGE, lw=1.6, label=f"类别式合理化（ρ = {fl['ρ']:.2f}）")
ax.plot(k, sh["预测"], color=INK, lw=1.2, ls="--", label=f"恒定少报（γ = {sh['γ']:.1f}）")
ax.set_xlabel("k（人）"); ax.set_ylabel("报告 ≤ k 的概率之差")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "图P5_合理化的形状.png", dpi=200); plt.close(fig)

# ---------- P6 ----------
from scipy.special import expit
rho = expit(np.array(J("s10_合理化联合模型_ρ逐人.json")["X"])[:, 7])
iv = J("s11_合理化对前面结论的影响.json")["工具变量：选择 → 报告不挤"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.2, 1]))
axs[0].hist(rho, bins=np.linspace(0, 1, 21), color=ORANGE, edgecolor="white", lw=1)
axs[0].set_xlabel("ρ_i：信念与选择不一致时把报告挪到一致一侧的概率"); axs[0].set_ylabel("人数")
labs = ["如实报告者 ρ_i<0.05", "中间 0.05–0.5", "合理化者 ρ_i>0.5"]; xs = np.arange(3)
b = [iv[l]["效应"]["b"] for l in labs]; e = [1.96 * iv[l]["效应"]["se"] for l in labs]
axs[1].axhline(0, color=MUTED, lw=0.7)
axs[1].bar(xs, b, width=0.55, color=[BLUE, GREY, ORANGE], edgecolor="white")
axs[1].errorbar(xs, b, yerr=e, fmt="none", ecolor=INK, lw=1, capsize=3)
cnt = [(rho < 0.05).sum(), ((rho >= 0.05) & (rho <= 0.5)).sum(), (rho > 0.5).sum()]
axs[1].set_xticks(xs); axs[1].set_xticklabels([f"如实报告者\n（{cnt[0]} 人）", f"中间\n（{cnt[1]} 人）", f"合理化者\n（{cnt[2]} 人）"])
axs[1].set_ylabel("被习惯推去后报「不挤」\n的概率增加（工具变量）")
fig.tight_layout(); fig.savefig(OUT / "图P6_合理化的个体差异.png", dpi=200); plt.close(fig)
print("ok")
