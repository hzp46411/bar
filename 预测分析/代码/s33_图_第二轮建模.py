# -*- coding: utf-8 -*-
"""第二轮建模（s20–s32）的图
  Q1 内部模型同时驱动预测与选择：不依赖模型的预测侧 δ̂ 与选择侧状态效应（逐人），按主模型的类型着色
  M1 成分敲除：主模型 HDWLG1 去掉各成分（或把 L 换成 MF）后的 iBIC 变化（s28 的同一设置）
  A1 ABM 的人数自相关 ACF1–4：真实值与各版本模型的闭环模拟（均值 ± 模拟间 SD）
  A2 预测者生态：反转者比例与人数 SD、效率（s32），并标出真实分配
  A3 扰动自稳：各敲除下的自稳增益（s27，主模型）
  M6 样本外预测：交错区组与前后半程两种交叉验证下，各模型相对主模型的检验负对数似然（s28、s35）
输出：结果/图/图M*_*.png
"""
import json, importlib.util, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pred_lib as PL

plt.rcParams.update({"font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})
BLUE, ORANGE, INK, MUTED, GREY, GREEN = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984", "#d9d8d3", "#2f9e6e"
J = lambda f: json.loads((PL.OUT / f).read_text(encoding="utf-8"))
OUT = PL.OUT / "图"; OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("s31", str(pathlib.Path(__file__).with_name("s31_内部模型的内容.py")))
S31 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S31)
MAIN = "HDWLG1"

# ---------- Q1 ----------
X = S31.S27.load(f"s28:{MAIN}"); delta = X[:, 6]
fc, ch = S31.state_contrast(np.ones(S31.T, bool))
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for lab, msk, col in (("反转（δ > 1）", delta > 1, BLUE), ("外推（δ < −1）", delta < -1, ORANGE), ("接近零", np.abs(delta) <= 1, MUTED)):
    ax.scatter(fc[msk], ch[msk], s=18, color=col, edgecolor="white", linewidth=0.6, label=f"{lab}，{msk.sum()} 人", zorder=3)
k = ~np.isnan(fc); r = np.corrcoef(fc[k], ch[k])[0, 1]
b1, b0 = np.polyfit(fc[k], ch[k], 1); xs = np.linspace(np.nanmin(fc), np.nanmax(fc), 50)
ax.plot(xs, b0 + b1 * xs, color=INK, lw=1, zorder=2)
ax.axhline(0, color=GREY, lw=0.8, zorder=1); ax.axvline(0, color=GREY, lw=0.8, zorder=1)
ax.set_xlabel("预测侧：上轮不挤之后 − 上轮挤之后的平均预测（人）")
ax.set_ylabel("选择侧：上轮挤之后 − 上轮不挤之后去的比例")
ax.set_title(f"同一个内部模型驱动预测与选择（r = {r:.2f}）", fontsize=9.5)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "图M1_内部模型的交叉检验.png", dpi=200); plt.close(fig)

# ---------- M1 ----------
cand = {"去快习惯 H": "DWLG1", "去慢漂移 D": "HWLG1", "去近期信念 L": "HDWG1", "世界模型不进选择 W": "HDLG1",
        "去分级反应 κ": "HDWL1", "去 λ1": "HDWLG", "L 换成 MF（参数数相同）": "HDWFG1", "加 λ2、λ4": "HDWLG124", "最简两系统 HW": "HW"}
ib0 = J(f"s28_真实_{MAIN}.json")["iBIC"]
items = [(k, J(f"s28_真实_{v}.json")["iBIC"] - ib0) for k, v in cand.items()]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ys = np.arange(len(items))[::-1]
vals = np.array([v for _, v in items])
ax.barh(ys, vals, color=[ORANGE if v > 10 else (BLUE if v < -10 else MUTED) for v in vals], height=0.6)
for y, v in zip(ys, vals):
    ax.text(v * 1.08 + 0.3, y, f"{v:+.0f}", va="center", ha="left", fontsize=8, color=INK)
ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, 4000)
ax.set_xticks([0, 10, 100, 1000]); ax.set_xticklabels(["0", "10", "100", "1000"])
ax.set_yticks(ys); ax.set_yticklabels([k for k, _ in items]); ax.axvline(0, color=INK, lw=0.8)
ax.set_xlabel("相对主模型的 iBIC 变化（10 以上为对数刻度；正 = 变差）")
ax.set_title("主模型（习惯 + 世界模型）的成分敲除", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "图M2_成分敲除.png", dpi=200); plt.close(fig)

# ---------- A1 ----------
vers = [("HDLB", "HDLB（s25）"), ("HDWLG", "HDWLG（s29）"), (f"s28_{MAIN}", "主模型 HDWLG1"), ("s28_HDWLG124", "加 λ2、λ4")]
cols = [GREY, MUTED, BLUE, GREEN]
fig, ax = plt.subplots(figsize=(5.4, 3.2))
lags = np.arange(1, 5); w = 0.18
real = [J(f"s27_ABM_{vers[2][0]}.json")["A 拟合检验"][f"ACF{k}"]["真实"] for k in lags]
for j, ((f, lab), c) in enumerate(zip(vers, cols)):
    d = J(f"s27_ABM_{f}.json")["A 拟合检验"]
    mu = [d[f"ACF{k}"]["均值"] for k in lags]; sd = [d[f"ACF{k}"]["模拟间SD"] for k in lags]
    ax.bar(lags + (j - 1.5) * w, mu, width=w, yerr=sd, color=c, error_kw=dict(lw=0.8, ecolor=INK), label=lab)
ax.scatter(lags, real, marker="D", s=30, color=ORANGE, zorder=5, label="真实")
ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(lags); ax.set_xticklabels([f"ACF{k}" for k in lags])
ax.set_ylabel("人数的自相关"); ax.set_title("ABM：闭环模拟的人数自相关", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "图M3_ABM自相关.png", dpi=200); plt.close(fig)

# ---------- A2 ----------
d = J(f"s32_预测者生态_s28_{MAIN}.json"); A = d["A 反转 / 外推的比例"]
f = np.array([float(k.split("=")[1]) for k in A]); sd = np.array([v["SD"][0] for v in A.values()]); ef = np.array([v["效率"][0] for v in A.values()])
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8))
for ax, y, lab, rv in ((axs[0], sd, "人数的 SD", d["真实的方向分配"]["SD"][0]), (axs[1], ef, "效率（每人每轮得分）", d["真实的方向分配"]["效率"][0])):
    ax.plot(f, y, color=BLUE, lw=2, marker="o", ms=4)
    ax.scatter([0.53], [rv], marker="D", s=36, color=ORANGE, zorder=5, label="真实的方向分配（53% 反转）")
    ax.set_xlabel("反转者的比例（方向随机分配，强度 |δ| 不变）"); ax.set_ylabel(lab)
axs[0].legend(frameon=False, fontsize=7.5)
fig.suptitle("预测者生态：反转与外推的混合决定协调的好坏", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "图M4_预测者生态.png", dpi=200); plt.close(fig)

# ---------- A3 ----------
g = J(f"s27_ABM_s28_{MAIN}.json")["C 扰动自稳"]
labs = list(g); vals = [np.mean([v["自稳增益"] for k, v in g[l].items() if k.startswith("Δ")]) for l in labs]
fig, ax = plt.subplots(figsize=(5.4, 2.8))
ys = np.arange(len(labs))[::-1]
ax.barh(ys, vals, color=[BLUE if v > 0 else ORANGE for v in vals], height=0.6)
for y, v in zip(ys, vals):
    ax.text(v + 0.03 if v >= 0 else 0.03, y, f"{v:.2f}", va="center", ha="left", fontsize=8)
ax.set_yticks(ys); ax.set_yticklabels(labs); ax.axvline(0, color=INK, lw=0.8); ax.set_xlim(min(vals) - 0.1, 1.15)
ax.set_xlabel("自稳增益 = 1 − 实际平移 / 无反馈平移（Δb 四档的平均）")
ax.set_title("扰动自稳来自世界层（公共信息）", fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "图M5_扰动自稳.png", dpi=200); plt.close(fig)
# ---------- M6 ----------
cvs = J("s35_汇总.json")
mods = [("HDWG1", "去近期信念 L"), ("HDWL1", "去分级反应 κ"), ("HDWLG", "去 λ1"), ("HDWFG1", "L 换成 MF"), ("HDWLT1", "κ 随时间变化")]
fig, ax = plt.subplots(figsize=(5.8, 3.0))
ys = np.arange(len(mods))[::-1]; h = 0.36
for j, (scheme, col, lab) in enumerate((("交错区组", BLUE, "交错区组（同一时期内泛化）"), ("前后半程", ORANGE, "前后半程（跨时间泛化）"))):
    v = np.array([cvs[scheme][m]["合计"] for m, _ in mods])
    shown = np.clip(v, -400, 400)
    ax.barh(ys + (0.5 - j) * h, shown, height=h, color=col, label=lab)
    for y, a, b in zip(ys + (0.5 - j) * h, v, shown):
        ax.text(b + (8 if b >= 0 else -8), y, f"{a:+.0f}", va="center", ha="left" if b >= 0 else "right", fontsize=7.5)
ax.set_yticks(ys); ax.set_yticklabels([l for _, l in mods]); ax.axvline(0, color=INK, lw=0.8); ax.set_xlim(-520, 520)
ax.set_xlabel("检验负对数似然相对主模型（正 = 预测更差；超出 ±400 截断显示）")
ax.set_title("样本外预测：κ 在同一时期内泛化，但随时间变化", fontsize=9.5)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "图M6_交叉验证.png", dpi=200); plt.close(fig)
print("完成")
