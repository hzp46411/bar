# -*- coding: utf-8 -*-
"""预测分析的图：P1 选择对自己预测的依赖（台阶）；P2 冲突时跟随习惯的比例随宏观状态变化；P3 信念类型的汇聚效度；P4 共同信念冲击与人数；
   P5 被习惯推去 / 推留时报告分布的差（工具变量）；P6 联合模型结构；P7 合理化的个体差异与工具变量的交叉验证"""
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

# ---------- P7 合理化的个体差异 ----------
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
fig.tight_layout(); fig.savefig(OUT / "图P7_合理化的个体差异.png", dpi=200); plt.close(fig)
print("ok")

# ---------- P6 联合模型结构 ----------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(7.4, 3.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
def box(x, y, w, h, t, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=INK, lw=0.8))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5, color=INK)
def arr(p, q, t="", c=INK, ls="-", off=(0, 0.12), rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=c, lw=1.2, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if t: ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], t, ha="center", fontsize=7.5, color=c)
LB, LO, LG = "#dbe8f8", "#fbe1d6", "#eeede9"
box(0.1, 3.6, 1.9, 0.9, "上一轮人数\nN(t−1)", LG)
box(3.0, 3.6, 2.0, 0.9, "信念 B ~ N(B̄, v)\n类别：挤 / 不挤", LB)
box(3.0, 0.4, 2.0, 0.9, "习惯痕迹 c\n（过去的选择）", LO)
box(0.1, 0.4, 1.9, 0.9, "宏观状态\n稳定、偏离", LG)
box(6.1, 2.0, 1.5, 0.9, "选择 a\n去 / 不去", "white")
box(8.3, 3.6, 1.6, 0.9, "报告 P\n（预测人数）", "white")
arr((2.0, 4.05), (3.0, 4.05), "μ, ws, wm")
arr((5.0, 3.85), (6.25, 2.9), "s：信念 → 选择", BLUE, off=(1.05, 0.05))
arr((5.0, 0.85), (6.25, 2.0), "κ：习惯 → 选择", ORANGE, off=(1.1, -0.25))
arr((2.0, 0.85), (3.0, 0.85), "ψ、δ", MUTED)
arr((5.0, 4.25), (8.3, 4.25), "如实报告（一致时，或 1 − ρ）", BLUE, off=(0, 0.1))
arr((7.6, 2.6), (8.6, 3.6), "合理化 ρ_i：\n不一致时挪到\n与选择一致的一侧", ORANGE, off=(0.75, -0.55))
arr((1.05, 3.6), (6.3, 2.35), "λ（推力）", MUTED, ls="--", off=(-1.2, -0.05), rad=0.0)
fig.tight_layout(); fig.savefig(OUT / "图P6_联合模型结构.png", dpi=200); plt.close(fig)
print("ok P6")

# ---------- P8 ACF4：4 轮前人数的影响在选择里、不在预测里；加上它，闭环就能复现 ACF4 ----------
d13 = J("s13_四轮前与ACF4.json")["个体回归"]; d13b = J("s13b_四轮项与闭环ACF4.json")
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0))
ks = np.arange(1, 7)
for lab, key, col, sc, off in (("选择残差（a − HRGPR，× 100 人）", "选择残差（a − HRGPR 概率）", ORANGE, 100, -0.12),
                               ("预测（如实报告者）", "预测：如实报告者", BLUE, 1, 0.12)):
    b = np.array([d13[key][f"N(t−{k})"]["b"] for k in ks]) * sc; se = np.array([d13[key][f"N(t−{k})"]["se"] for k in ks]) * sc
    axs[0].errorbar(ks + off, b, yerr=1.96 * se, fmt="o", color=col, ms=4, capsize=2, lw=1, label=lab)
axs[0].axhline(0, color=MUTED, lw=0.7); axs[0].set_xticks(ks); axs[0].set_xticklabels([f"t−{k}" for k in ks])
axs[0].set_xlabel("哪一轮的人数（每 10 人）"); axs[0].set_ylabel("系数（人）"); axs[0].legend(frameon=False, fontsize=7.5, loc="lower left")
lags = ["acf1", "acf2", "acf3", "acf4"]; x = np.arange(4)
for j, (lab, col) in enumerate((("HRGPR", GREY), ("HRGPR + δ4", ORANGE))):
    mu = [d13b[lab][k]["均值"] for k in lags]; sd = [d13b[lab][k]["SD"] for k in lags]
    axs[1].bar(x + (j - 0.5) * 0.36, mu, width=0.34, color=col, edgecolor="white", label=f"闭环模拟：{lab}")
    axs[1].errorbar(x + (j - 0.5) * 0.36, mu, yerr=sd, fmt="none", ecolor=INK, lw=0.8, capsize=2)
axs[1].scatter(x, [d13b["真实"][k] for k in lags], marker="D", color=INK, s=22, zorder=3, label="真实人数")
axs[1].axhline(0, color=MUTED, lw=0.7); axs[1].set_xticks(x); axs[1].set_xticklabels(["ACF1", "ACF2", "ACF3", "ACF4"])
axs[1].set_ylabel("自相关"); axs[1].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "图P8_四轮前与ACF4.png", dpi=200); plt.close(fig)

# ---------- P9 从预测估出的学习率 vs 从选择估出的学习率 ----------
d12 = J("s12_BBL信念与预测_BBL.json"); rp = np.array(d12["ρᴾ"]); rc = np.array(d12["ρ_选择（HRGPR）"])
from scipy.stats import spearmanr
fig, ax = plt.subplots(figsize=(4.4, 3.6))
jit = np.random.default_rng(1).uniform(-0.012, 0.012, (2, len(rp)))
ax.scatter(np.clip(rc, 0, 1) + jit[0], np.clip(rp, 0, 1) + jit[1], s=16, color=BLUE, edgecolor="white", lw=0.5)
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
ax.set_xlabel("从选择估出的学习率 ρ（HRGPR）"); ax.set_ylabel("从预测估出的学习率 ρᴾ")
ax.set_title(f"Spearman = {spearmanr(rp, rc)[0]:.2f}", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "图P9_两种学习率.png", dpi=200); plt.close(fig)
print("ok P8 P9")

# ---------- P10 参数恢复：只用选择 vs 选择 + 预测 ----------
import glob
def load_sets(pattern):
    return [J(pathlib_name) for pathlib_name in sorted(p.split("/")[-1] for p in glob.glob(str(PL.OUT / pattern)))]
MET = [("ρ_Spearman", "ρ\n（Spearman）"), ("ρ_类别判对比例", "ρ 类别\n判对比例"), ("β_相关", "β"), ("κ_相关", "κ"), ("b_相关", "b")]
sb = load_sets("s14b_联合模型恢复_套*.json"); sc = load_sets("s14c_两种学习率下的恢复_套*.json")
panels = [("真值：预测与选择共用学习率（J1）", sb, [("只用选择（HRGPR）", GREY, "只用选择"), ("选择 + 预测（J1）", ORANGE, "选择 + 预测")])]
if sc:
    panels.append(("真值：两个学习率不同（J2）", sc, [("只用选择（HRGPR）", GREY, "只用选择"), ("选择 + 预测，两个学习率（J2，设定正确）", ORANGE, "选择 + 预测（J2）"),
                                                ("选择 + 预测，共用学习率（J1，设定错误）", BLUE, "选择 + 预测（J1，设定错）")]))
fig, axs = plt.subplots(1, len(panels), figsize=(3.9 * len(panels) + 0.4, 3.2), sharey=True, squeeze=False)
for ax, (title, sets, arms) in zip(axs[0], panels):
    x = np.arange(len(MET)); w = 0.8 / len(arms)
    for j, (key, col, lab) in enumerate(arms):
        vals = np.array([[row[key][m] for m, _ in MET] for d in sets for row in d["套"]])
        xx = x + (j - (len(arms) - 1) / 2) * w
        ax.bar(xx, vals.mean(0), width=w * 0.92, color=col, edgecolor="white", label=lab)
        for v in vals: ax.scatter(xx, v, s=8, color=INK, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in MET], fontsize=8); ax.set_ylim(0.5, 1.0)
    ax.set_title(title, fontsize=9); ax.legend(frameon=False, fontsize=7.5, loc="lower left")
axs[0][0].set_ylabel("真值与估计值的相关 / 判对比例")
fig.tight_layout(); fig.savefig(OUT / "图P10_参数恢复.png", dpi=200); plt.close(fig)
print("ok P10")

# ---------- P11 宏观秩序的来源：成分拆分 + 扰动自稳 ----------
d17 = J("s17_宏观秩序的生成.json")
order = ["完整人群", "去习惯（κ = 0，ψ = 0）", "去 λ", "去共同冲击（σ = 0）", "偏好统一（b = 中位数）", "去信念（β = 0）",
         "全体反向（β = −|β|）", "全体外推（β = +|β|）", "打乱人群（参数独立置换）", "同质人群（中位数参数）"]
short = ["完整", "去习惯", "去 λ", "去共同冲击", "偏好统一", "去信念", "全体反向", "全体外推", "参数打乱", "同质人群"]
A_ = d17["A 成分拆分"]
fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
y = np.arange(len(order))[::-1]
mu = [A_[k]["均值"]["均值"] for k in order]; sdv = [A_[k]["SD"]["均值"] for k in order]
cols = [ORANGE if k == "完整人群" else GREY for k in order]
axs[0].barh(y, mu, color=cols, edgecolor="white"); axs[0].axvline(60, color=INK, lw=0.8, ls=":")
axs[0].set_yticks(y); axs[0].set_yticklabels(short, fontsize=8); axs[0].set_xlabel("平均人数（虚线 = 容量 60）"); axs[0].set_xlim(25, 66)
axs[1].barh(y, sdv, color=cols, edgecolor="white"); axs[1].axvline(d17["真实"]["SD"], color=INK, lw=0.8, ls=":")
axs[1].set_yticks(y); axs[1].set_yticklabels([]); axs[1].set_xlabel("人数的 SD（虚线 = 真实 5.7）")
B_ = d17["B 扰动自稳"]["格"]; dbs = ["-1.0", "-0.5", "0.0", "0.5", "1.0"]; xs = [-1, -.5, 0, .5, 1]
for k, col, lab in (("完整", ORANGE, "完整"), ("去 λ", BLUE, "去 λ"), ("去信念", INK, "去信念"), ("去信念且去 λ", MUTED, "去信念且去 λ")):
    axs[2].plot(xs, [B_[k]["各Δb"][d]["均值"] for d in dbs], "-o", color=col, ms=3.5, lw=1.6, label=f"{lab}（吸收 {max(B_[k]['自稳增益'], 0):.0%}）")
axs[2].axhline(60, color=INK, lw=0.8, ls=":"); axs[2].set_xlabel("给所有人去的倾向加的推动 Δb"); axs[2].set_ylabel("平均人数")
axs[2].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "图P11_宏观秩序的来源.png", dpi=200); plt.close(fig)

# ---------- P12 频率依赖：外推者越多越吃亏，得分相等处人数恰在容量 ----------
d18 = J("s18_内部模型为什么会产生.json"); fb = d18["A 频率依赖：选择中的信念方向（β > 0 外推）"]
rows = fb["各比例"]; f_ = np.array([r["f"] for r in rows])
pp = np.array([np.nan if r["正向者得分"] is None else r["正向者得分"] for r in rows]); pm = np.array([np.nan if r["负向者得分"] is None else r["负向者得分"] for r in rows])
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.2))
axs[0].plot(f_, pp, "-o", color=ORANGE, ms=4, lw=1.8, label="外推者（β > 0）的得分")
axs[0].plot(f_, pm, "-o", color=BLUE, ms=4, lw=1.8, label="反向者（β < 0）的得分")
axs[0].axvline(fb["得分相等的比例"], color=INK, lw=0.8, ls="--"); axs[0].axvline(fb["真实比例"], color=MUTED, lw=4, alpha=0.35)
axs[0].text(fb["得分相等的比例"] + 0.02, 0.60, f"得分相等\nf* = {fb['得分相等的比例']:.2f}", fontsize=7.5)
axs[0].text(fb["真实比例"] - 0.30, 0.62, f"真实人群\nf = {fb['真实比例']:.2f}", fontsize=7.5, color=MUTED)
axs[0].set_xlabel("人群中外推者的比例 f"); axs[0].set_ylabel("每人每轮平均得分"); axs[0].legend(frameon=False, fontsize=7.5, loc="upper right")
axs[1].plot(f_, [r["平均人数"] for r in rows], "-o", color=INK, ms=4, lw=1.6)
axs[1].axhline(60, color=INK, lw=0.8, ls=":"); axs[1].axvline(fb["得分相等的比例"], color=INK, lw=0.8, ls="--"); axs[1].axvline(fb["真实比例"], color=MUTED, lw=4, alpha=0.35)
axs[1].set_xlabel("人群中外推者的比例 f"); axs[1].set_ylabel("平均人数（虚线 = 容量）")
fig.tight_layout(); fig.savefig(OUT / "图P12_频率依赖.png", dpi=200); plt.close(fig)
print("ok P11 P12")

# ---------- P13 Arthur 机制：人用哪些预测规则；是否按近期准确度换规则 ----------
d16 = J("s16_Arthur机制检验.json"); eco = d16["规则生态"]
labs = ["锚定60", "其他", "重复自己", "镜像", "延续", "四轮均值", "八轮趋势", "两轮周期"]
show = ["锚定 60", "个人其他", "重复自己", "镜像", "延续", "四轮均值", "八轮趋势", "两轮周期"]
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
x = np.arange(len(labs)); w = 0.4
axs[0].bar(x - w / 2, [eco["各规则的平均后验份额"][k] for k in labs], width=w, color=ORANGE, edgecolor="white", label="全体")
axs[0].bar(x + w / 2, [eco["如实报告者_各规则份额"][k] for k in labs], width=w, color=BLUE, edgecolor="white", label="如实报告者")
axs[0].set_xticks(x); axs[0].set_xticklabels(show, fontsize=7.5, rotation=30, ha="right"); axs[0].set_ylabel("预测归于该规则的平均后验份额")
axs[0].legend(frameon=False, fontsize=7.5)
prof = d16["θ 的轮廓似然与检验"]
for r, col in (("0.1", ORANGE), ("0.3", INK), ("0.6", BLUE)):
    th = sorted(prof[r]["轮廓"], key=float); base = prof[r]["轮廓"]["0.0"]["全体"]
    axs[1].plot([float(t) for t in th], [prof[r]["轮廓"][t]["全体"] - base for t in th], "-o", color=col, ms=3.5, lw=1.6, label=f"准确度记忆速率 r = {r}")
axs[1].axhline(0, color=MUTED, lw=0.7); axs[1].set_xlabel("对近期准确度的敏感度 θ"); axs[1].set_ylabel("负对数似然 − θ = 0 时的值")
axs[1].set_ylim(-40, 200); axs[1].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "图P13_Arthur机制检验.png", dpi=200); plt.close(fig)
print("ok P13")
