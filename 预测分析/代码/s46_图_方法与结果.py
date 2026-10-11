# -*- coding: utf-8 -*-
"""
第 46 步：方法与结果报告的插图（只读已有结果文件，不重新拟合、不重新模拟）
  图 1 观测指纹（3.1）：每轮人数、自相关、选择的滞后回归、预测方向 × 选择侧状态效应
  图 2 可识别性（3.2）：模型恢复的选中矩阵、HB 的参数恢复（需要 s45 / s22 的假数据拟合）
  图 3 成分检验（3.3）：ΔiBIC、无模型强化的校准分布、样本外损失
  图 4 主模型机制（3.4）：模型 δ 与陈述信念、习惯记忆半衰期、习惯权重、开环后验预测
  图 5 共同成分（3.5）：联合估计的收敛轨迹、两步法与联合估计的恢复（需要 s44）
  图 6 闭环检验（3.6）：完整模型与敲除的 ACF1、SD，人群自稳增益，完整模型的自相关形状
配色：dataviz 参考调色板（蓝 #2a78d6、橙 #eb6834、水绿 #1baf7a 已通过 CVD 校验）；文字用中性墨色
用法：python3 s46_图_方法与结果.py [图号…]      （不给图号则画全部能画的）
输出：交付/方法与结果报告/图/图<号>_*.png
"""
import sys, json, importlib.util, pathlib
import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import pearsonr
import pred_lib as PL

FIG = PL.W / "交付" / "方法与结果报告" / "图"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e0"
TYPE_COL = {"反转": BLUE, "外推": ORANGE, "无方向": GRAY}
plt.rcParams.update({"font.family": ["sans-serif"], "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
                     "axes.edgecolor": GRAY, "axes.linewidth": 0.8, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 200,
                     "savefig.bbox": "tight", "figure.facecolor": "white",
                     "mathtext.fontset": "custom", "mathtext.rm": "WenQuanYi Zen Hei", "mathtext.sf": "WenQuanYi Zen Hei", "mathtext.default": "regular",
                     })
J = lambda f: json.loads((PL.OUT / f).read_text(encoding="utf-8"))
PLAIN = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")


def _load(fname, mod):
    spec = importlib.util.spec_from_file_location(mod, str(pathlib.Path(__file__).with_name(fname)))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    return M


def fig1():
    S43 = _load("s43_框架补充.py", "s43")
    N = PL.N.astype(float); T = PL.T
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.2))
    a = ax[0, 0]
    a.plot(np.arange(1, T + 1), N, color=BLUE, lw=0.8)
    a.axhline(60, color=INK, lw=1, ls="--"); a.text(T * 0.99, 77, "虚线 = 容量 60", ha="right", va="top", color=INK, fontsize=8)
    a.set(title="A  每轮出席人数", xlabel="轮次", ylabel="人数", xlim=(1, T))
    a = ax[0, 1]
    ks = np.arange(1, 7); x = N - N.mean()
    ac = np.array([(x[:-k] * x[k:]).sum() / (x * x).sum() for k in ks])
    a.bar(ks, ac, color=BLUE, width=0.6)
    a.axhspan(-1.96 / np.sqrt(T), 1.96 / np.sqrt(T), color=GRID, zorder=0)
    a.axhline(0, color=INK2, lw=0.8)
    for k, v in zip(ks, ac):
        a.text(k, v + (0.015 if v >= 0 else -0.015), f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
    a.set(title="B  人数的自相关", xlabel="滞后阶数 k", ylabel="ACF$_k$", ylim=(-0.45, 0.2))
    a = ax[1, 0]
    d = J("s22_预测检查.json"); b, z = np.array(d["真实"]["系数"]), np.array(d["真实"]["z"])
    se = np.abs(b / z); lab = ["t-1", "t-2", "t-3", "t-4", "t-5"]
    a.errorbar(np.arange(5), b[:5], yerr=1.96 * se[:5], fmt="o", color=BLUE, ms=5, capsize=0, lw=1.5)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(np.arange(5)); a.set_xticklabels(lab)
    a.set(title="C  自己过去选择的影响（95% CI）", xlabel="自己的选择", ylabel="回归系数（logit）")
    a = ax[1, 1]
    dh, _, ch, ty = S43.classify(np.ones(T, bool))
    for k in ("反转", "外推", "无方向"):
        m = ty == k
        a.scatter(dh[m], ch[m], s=18, color=TYPE_COL[k], label=f"{k}（{m.sum()}）", edgecolor="white", linewidth=0.6)
    ok = ~np.isnan(dh) & ~np.isnan(ch); r = pearsonr(dh[ok], ch[ok])[0]
    a.axhline(0, color=INK2, lw=0.8); a.axvline(0, color=INK2, lw=0.8)
    a.text(0.02, 0.96, f"r = {r:.2f}", transform=a.transAxes, va="top", color=INK)
    a.legend(loc="lower right", fontsize=8, title="预测方向", title_fontsize=8)
    a.set(title="D  预测方向与选择方向", xlabel="预测侧 $\\hat\\delta_i$（不挤后减挤后，人）", ylabel="选择侧 $D_i$（挤后减不挤后去的比例）")
    fig.tight_layout(); fig.savefig(FIG / "图1_观测指纹.png"); plt.close(fig)


def fig2():
    MODELS = ["0", "H", "F", "B", "HF", "HB", "FB", "HFB"]
    f = PL.OUT / "s45_恢复扩充.json"
    if not f.exists():
        print("图 2：缺 s45_恢复扩充.json，跳过"); return
    r = J("s45_恢复扩充.json")
    P = np.array([[r["模型恢复"]["选中概率"][g][m] if r["模型恢复"]["选中概率"][g] else np.nan for m in MODELS] for g in MODELS])
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2), gridspec_kw=dict(width_ratios=[1.05, 1]))
    a = ax[0]
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"])
    a.imshow(P, cmap=cm, vmin=0, vmax=1); a.grid(False)
    for i in range(8):
        for j in range(8):
            if P[i, j] > 0:
                a.text(j, i, f"{P[i, j]:.1f}".replace("1.0", "1"), ha="center", va="center", fontsize=8, color="white" if P[i, j] > .5 else INK)
    a.set_xticks(range(8)); a.set_xticklabels(MODELS); a.set_yticks(range(8)); a.set_yticklabels(MODELS)
    n = r["模型恢复"]["套数"]
    a.set(title=f"A  模型恢复（每个生成模型 {min(n.values())}–{max(n.values())} 套）", xlabel="按 iBIC 选中的模型", ylabel="生成模型")
    a = ax[1]
    pr = r["参数恢复"].get("HB", {})
    names = ["lnσ", "δ", "w_I", "w_B", "b", "α_H", "η"]
    show = {"lnσ": "$\\sigma_y$", "δ": "$\\delta$", "w_I": "$w_I$", "w_B": "$w_B$", "b": "$b$", "α_H": "$\\alpha_H$", "η": "$\\eta$"}
    y = np.arange(len(names))[::-1]
    med = [pr[k]["各套中位数"] for k in names]; lo = [pr[k]["范围"][0] for k in names]; hi = [pr[k]["范围"][1] for k in names]
    a.hlines(y, lo, hi, color=GRAY, lw=2)
    a.scatter(med, y, color=BLUE, s=30, zorder=3)
    for yy, m in zip(y, med):
        a.text(m, yy + 0.25, f"{m:.2f}", ha="center", fontsize=8, color=INK)
    a.set_yticks(y); a.set_yticklabels([show[k] for k in names]); a.set_xlim(0.5, 1.02)
    a.set(title="B  HB 的参数恢复（Spearman ρ：中位数与范围）", xlabel="真值与估计值的相关")
    fig.tight_layout(); fig.savefig(FIG / "图2_可识别性.png"); plt.close(fig)


def fig3():
    MODELS = ["0", "H", "F", "B", "HF", "HB", "FB", "HFB"]
    s = J("s22_真实_汇总.json")["iBIC"]; best = min(s[m] for m in MODELS)
    dl = {m: s[m] - best for m in MODELS}
    order = sorted(MODELS, key=lambda m: dl[m])
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1.2, 0.8]))
    a = ax[0]
    y = np.arange(len(order))[::-1]
    a.barh(y, [dl[m] for m in order], color=[BLUE if m == "HB" else GRAY for m in order], height=0.6)
    for yy, m in zip(y, order):
        a.text(dl[m] * 1.15 + 0.3, yy, f"{dl[m]:.1f}", va="center", fontsize=8, color=INK)
    a.set_xscale("symlog", linthresh=1); a.set_xlim(0, 3e4); a.xaxis.set_major_formatter(PLAIN)
    a.set_yticks(y); a.set_yticklabels(order)
    a.set(title="A  ΔiBIC（相对最优）", xlabel="ΔiBIC（对称对数刻度）")
    a = ax[1]
    real = s["HFB"] - s["HB"]
    f45 = PL.OUT / "s45_恢复扩充.json"
    vals = {}
    for g in ("HB", "HFB"):
        v = []
        for r in range(1, 31):
            fa, fb = PL.OUT / f"s22_{g}_{r}_HFB.json", PL.OUT / f"s22_{g}_{r}_HB.json"
            if fa.exists() and fb.exists():
                v.append(J(fa.name)["iBIC"] - J(fb.name)["iBIC"])
        vals[g] = np.array(v)
    rng = np.random.default_rng(1)
    for k, (g, col) in enumerate((("HB", BLUE), ("HFB", ORANGE))):
        a.scatter(vals[g], k + rng.uniform(-0.12, 0.12, len(vals[g])), color=col, s=20, edgecolor="white", linewidth=0.5,
                  label=f"{g} 为真（{len(vals[g])} 套）")
    a.axvline(real, color=INK, lw=1.5); a.text(real, 1.45, f"真实数据 {real:.1f}", ha="center", fontsize=8, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    a.axvline(0, color=INK2, lw=0.8, ls=":")
    a.set_yticks([0, 1]); a.set_yticklabels(["HB 为真", "HFB 为真"]); a.set_ylim(-0.5, 1.7)
    a.set(title="B  无模型强化的校准", xlabel="Δ = iBIC(HFB) - iBIC(HB)（负 = 支持 F）")
    a = ax[2]
    o = J("s42_MF样本外检验.json")
    v = [o["前后半程"]["HFB − HB"]["样本外增益_nats"], o["交错区组"]["HFB − HB"]["样本外增益_nats"]]
    a.bar([0, 1], v, color=ORANGE, width=0.55)
    for x_, vv in zip([0, 1], v):
        a.text(x_, vv + 3, f"+{vv:.0f}", ha="center", fontsize=8, color=INK)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks([0, 1]); a.set_xticklabels(["前后半程", "交错区组"]); a.set_ylim(0, 150)
    a.set(title="C  样本外：加入 F 的损失", ylabel="HFB 减 HB（nats；正 = F 更差）")
    fig.tight_layout(); fig.savefig(FIG / "图3_成分检验.png"); plt.close(fig)


def fig4():
    S43 = _load("s43_框架补充.py", "s43")
    X = np.array(J("s22_真实_HB.json")["X"])
    dh, _, _, ty = S43.classify(np.ones(PL.T, bool))
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))
    a = ax[0, 0]
    for k in ("反转", "外推", "无方向"):
        m = ty == k
        a.scatter(X[m, 6], dh[m], s=18, color=TYPE_COL[k], label=k, edgecolor="white", linewidth=0.6)
    ok = ~np.isnan(dh); r = pearsonr(X[ok, 6], dh[ok])[0]
    lim = [min(X[:, 6].min(), np.nanmin(dh)) - 1, max(X[:, 6].max(), np.nanmax(dh)) + 1]
    a.plot(lim, lim, color=INK2, lw=0.8, ls="--")
    a.text(0.02, 0.96, f"r = {r:.2f}", transform=a.transAxes, va="top", color=INK)
    a.legend(loc="lower right", fontsize=8, title="预测方向（不依赖模型）", title_fontsize=8)
    a.set(title="A  模型 δ 与陈述信念", xlabel="模型估计的 $\\delta_i$（人）", ylabel="预测侧 $\\hat\\delta_i$（人）")
    a = ax[0, 1]
    aH = expit(X[:, 8]); hl = np.log(0.5) / np.log(1 - np.clip(aH, 1e-6, 1 - 1e-6))
    bins = np.logspace(np.log10(0.1), np.log10(200), 22)
    a.hist(np.clip(hl, 0.1, 199), bins=bins, color=BLUE, edgecolor="white", linewidth=0.6)
    a.set_xscale("log"); a.xaxis.set_major_formatter(PLAIN); a.axvline(np.median(hl), color=INK, lw=1.2, ls="--")
    a.text(np.median(hl) * 1.1, a.get_ylim()[1] * 0.92, f"中位数 {np.median(hl):.1f} 轮", color=INK, fontsize=8)
    a.set(title="B  习惯记忆的半衰期", xlabel="半衰期（轮，对数刻度）", ylabel="人数")
    a = ax[1, 0]
    wI = X[:, 1]
    a.hist(wI, bins=np.linspace(-4, 6, 26), color=[BLUE][0], edgecolor="white", linewidth=0.6)
    a.axvline(0, color=INK, lw=1.2)
    a.text(0.03, 0.95, f"$w_I<0$（交替）：{(wI < 0).sum()} 人", transform=a.transAxes, va="top", fontsize=8, color=INK)
    a.text(0.97, 0.95, f"$w_I>0$（重复）：{(wI > 0).sum()} 人", transform=a.transAxes, va="top", ha="right", fontsize=8, color=INK)
    a.set(title="C  习惯权重", xlabel="$w_I$", ylabel="人数")
    a = ax[1, 1]
    d = J("s22_预测检查.json"); b, z = np.array(d["真实"]["系数"]), np.array(d["真实"]["z"]); se = np.abs(b / z)
    hb, hsd = np.array(d["HB"]["均值"]), np.array(d["HB"]["模拟间标准差"])
    lab = ["c t-1", "c t-2", "c t-3", "c t-4", "c t-5", "s t-1", "s t-2", "c×s t-1", "c×s t-2"]
    x = np.arange(9)
    a.errorbar(x - 0.15, b, yerr=1.96 * se, fmt="D", color=INK, ms=4, lw=1.2, label="真实（95% CI）")
    a.errorbar(x + 0.15, hb, yerr=1.96 * hsd, fmt="o", color=BLUE, ms=4, lw=1.2, label="HB 模拟（±1.96 SD）")
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(x); a.set_xticklabels(lab, rotation=40, ha="right", fontsize=8)
    a.legend(fontsize=8, loc="upper right")
    a.set(title="D  开环后验预测：滞后回归", ylabel="回归系数（logit）")
    fig.tight_layout(); fig.savefig(FIG / "图4_主模型机制.png"); plt.close(fig)


def fig5():
    h = J("s38_联合估计_HB.json")["历史"]
    f44 = PL.OUT / "s44_联合估计恢复.json"
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    a = ax[0]
    it = [x["轮"] for x in h]
    a.plot(it, [x["γ"] for x in h], "-o", color=BLUE, ms=4, lw=1.8, label="γ（共同纠偏）")
    a.plot(it, [x["σ"] for x in h], "-o", color=ORANGE, ms=4, lw=1.8, label="σ$_c$（共同冲击）")
    a.text(0, h[0]["γ"] - 0.03, "两步法", fontsize=8, color=INK2)
    a.legend(fontsize=8, loc="lower right")
    a.set(title="A  真实数据：联合估计的收敛", xlabel="条件期望最大化的轮次（0 = 两步法起点）", ylabel="估计值", ylim=(0, 0.42))
    a = ax[1]
    if f44.exists():
        rs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(PL.OUT.glob("s44_真_*.json"))]
        tg, ts = rs[0]["真值"]["γ"], rs[0]["真值"]["σ"]
        rng = np.random.default_rng(2)
        for k, (meth, col) in enumerate((("两步法", GRAY), ("联合", BLUE))):
            g = np.array([r[meth]["γ"] for r in rs]); s = np.array([r[meth]["σ"] for r in rs])
            a.scatter(g, s, color=col, s=22, edgecolor="white", linewidth=0.5, label=f"{meth}（{len(rs)} 套）")
        a.scatter([tg], [ts], marker="*", s=160, color=INK, label="真值", zorder=4)
        a.legend(fontsize=8, loc="center")
        a.set(title="B  模拟数据：两步法与联合估计的恢复", xlabel="γ 的估计", ylabel="σ$_c$ 的估计")
    else:
        a.text(0.5, 0.5, "等待 s44", transform=a.transAxes, ha="center")
    fig.tight_layout(); fig.savefig(FIG / "图5_共同成分.png"); plt.close(fig)


def fig6():
    d = J("s41_联合估计ABM.json")
    pops = [("完整模型", d["A 拟合检验"]), ("去习惯", d["B 成分敲除"]["去习惯（w_I = 0）"]),
            ("去世界模型", d["B 成分敲除"]["世界模型不进选择（w_B = 0）"]), ("去共同纠偏 γ", d["B 成分敲除"]["去共同分级反应（γ = 0）"]),
            ("去共同冲击 σ", d["B 成分敲除"]["去共同冲击（σ = 0）"]), ("只留共同成分", d["B 成分敲除"]["只留共同因素（w_I = w_B = 0）"])]
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 6.6))
    for a, key, title in ((ax[0, 0], "ACF1", "A  一阶自相关"), (ax[0, 1], "SD", "B  人数的标准差")):
        y = np.arange(len(pops))[::-1]
        for yy, (nm, e) in zip(y, pops):
            m, sd = e["指标"][key]["均值"], e["指标"][key]["模拟间SD"]
            a.errorbar(m, yy, xerr=1.96 * sd, fmt="o", color=BLUE if nm == "完整模型" else GRAY, ms=5, lw=1.5)
        real = pops[0][1]["指标"][key]["真实"]
        a.axvline(real, color=INK, lw=1.5); a.text(real, len(pops) - 0.4, f"真实 {real:.2f}", ha="center", fontsize=8, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1))
        a.set_yticks(y); a.set_yticklabels([p[0] for p in pops]); a.set_ylim(-0.6, len(pops) - 0.1)
        a.set(title=title, xlabel=f"{key}（模拟均值 ± 1.96 SD）")
    a = ax[1, 0]
    st = J("s43_自稳.json"); dd = d["D 扰动自稳"]
    gm = lambda r: np.mean([v["自稳增益"] for k, v in r.items() if k.startswith("Δb")])
    rows = [("HB 完整", gm(dd["完整"]), BLUE), ("HB 去世界模型", gm(dd["世界模型不进选择"]), GRAY), ("HB 去共同纠偏", gm(dd["去共同分级反应"]), GRAY),
            ("HB 去习惯", gm(dd["去习惯"]), GRAY), ("H + 共同成分", gm(st["H_γσ"]["结果"]), ORANGE), ("H + 共同成分，去 γ", gm(st["H_γσ_去γ"]["结果"]), ORANGE)]
    y = np.arange(len(rows))[::-1]
    a.barh(y, [r_[1] for r_ in rows], color=[r_[2] for r_ in rows], height=0.6)
    for yy, r_ in zip(y, rows):
        a.text(r_[1] + (0.02 if r_[1] >= 0 else -0.02), yy, f"{r_[1]:.2f}", va="center", ha="left" if r_[1] >= 0 else "right", fontsize=8, color=INK)
    a.axvline(0, color=INK2, lw=0.8)
    a.set_yticks(y); a.set_yticklabels([r_[0] for r_ in rows]); a.set_xlim(-0.6, 1.0)
    a.set(title="C  扰动自稳增益（四个 Δb 的平均）", xlabel="1 = 完全拉回，0 = 不拉回，< 0 = 放大")
    a = ax[1, 1]
    ks = [1, 2, 3, 4]
    real = [pops[0][1]["指标"][f"ACF{k}"]["真实"] for k in ks]
    m = [pops[0][1]["指标"][f"ACF{k}"]["均值"] for k in ks]; sd = [pops[0][1]["指标"][f"ACF{k}"]["模拟间SD"] for k in ks]
    a.errorbar(np.array(ks) + 0.08, m, yerr=1.96 * np.array(sd), fmt="o-", color=BLUE, ms=5, lw=1.5, label="完整模型（± 1.96 SD）")
    a.plot(np.array(ks) - 0.08, real, "D-", color=INK, ms=5, lw=1.5, label="真实")
    a.axhline(0, color=INK2, lw=0.8); a.axvspan(3.5, 4.5, color=GRID, zorder=0)
    a.text(4, 0.25, "ACF4 只作描述", ha="center", fontsize=8, color=INK2)
    a.set_xticks(ks); a.legend(fontsize=8, loc="lower right")
    a.set(title="D  完整模型的自相关形状", xlabel="滞后阶数 k", ylabel="ACF$_k$", ylim=(-0.6, 0.35))
    fig.tight_layout(); fig.savefig(FIG / "图6_闭环检验.png"); plt.close(fig)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for k in todo:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6}[k]()
        print("图", k, "完成", flush=True)
