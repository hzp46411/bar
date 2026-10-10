# -*- coding: utf-8 -*-
"""第 16 步：真实的人是否像 Arthur 设想的那样"按最近最准的假设行动"？——预测规则的混合模型
  Arthur（1994）：每人持有多条预测规则，按近期准确度挑一条行动；被采用的规则决定人数，人数历史又决定哪些规则被采用（规则的"生态"）。
  预测规则（用 t 轮之前的人数预测本轮人数）：
    H1 延续：N(t−1)            H2 镜像（以容量 60 为轴）：120 − N(t−1)       H3 两轮周期：N(t−2)
    H4 近四轮均值               H5 近八轮趋势外推（截在 30–90）              H6 锚定：60
    H7 重复自己上一轮的预测      H0 其他（每人一个宽的正态分布 N(μ_i, s_i²)，代表上述规则之外的想法）
  模型：P_it ~ Σ_h π_ith · N(Ĥ_h,t, σ_i²)（H0 用 N(μ_i, s_i²)），π_ith ∝ exp(α_ih + θ·acc_h,t)
        acc_h,t = −（规则 h 过去绝对误差的指数加权，速率 r）/ 10；H7 的准确度用本人过去的预测误差；H0 没有准确度项
        α_ih：每人对各规则的固定偏好；θ：对"近期准确度"的敏感度——Arthur 的机制要求 θ > 0
  估计：θ、r 取网格，逐人最大似然（α、log σ、μ、log s）；θ 的轮廓似然与 θ = 0 的似然比检验；另估逐人 θ_i。
  全体与"如实报告者"（s10 的 ρ_f < 0.05，排除合理化的污染）分别估计。
  输出：规则的使用份额（人群的"规则生态"）、各规则本身的准确度、θ 的估计与检验
用法：python3 s16_Arthur机制检验.py 网格 <r>     —— 对一个记忆速率 r 扫 θ 网格（三个 r 可并行）
      python3 s16_Arthur机制检验.py 精修 <r>     —— 每个 (r, θ) 都从所有格点的逐人解中挑最好的两个重新优化（消除局部最优造成的轮廓噪声）
      python3 s16_Arthur机制检验.py 合并           —— 汇总各 r 的轮廓似然（有精修结果时用精修值），在最佳 (r, θ) 下计算规则生态
输出：结果/s16_Arthur机制检验_r<r>.json；结果/s16_Arthur机制检验.json
"""
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, expit
from scipy.stats import chi2
import pred_lib as PL
P, N, OK, n, T = PL.P, PL.N.astype(float), PL.OK, PL.n, PL.T
t0 = 8
HN = ["延续", "镜像", "两轮周期", "四轮均值", "八轮趋势", "锚定60", "重复自己"]
H = len(HN)


def predictors():
    pr = np.full((H - 1, T), np.nan)
    for t in range(t0, T):
        past = N[:t]
        pr[0, t] = past[-1]; pr[1, t] = 120 - past[-1]; pr[2, t] = past[-2]; pr[3, t] = past[-4:].mean()
        x = np.arange(8); b = np.polyfit(x, past[-8:], 1); pr[4, t] = np.clip(np.polyval(b, 8), 30, 90); pr[5, t] = 60.0
    return pr


PR = predictors()                                            # (6, T) 公共规则
OWN = np.c_[np.full((n, 1), np.nan), np.where(OK, P, np.nan)[:, :-1]]   # 自己上一轮的预测


def accuracy(r):
    """acc[h, t]（公共规则）与 accown[i, t]（重复自己）：t 之前绝对误差的指数加权，取负、除以 10。"""
    acc = np.zeros((H - 1, T)); e = np.full(H - 1, 6.0)
    for t in range(t0, T):
        acc[:, t] = -e / 10; e = (1 - r) * e + r * np.abs(PR[:, t] - N[t])
    accown = np.zeros((n, T)); eo = np.full(n, 6.0)
    for t in range(t0, T):
        accown[:, t] = -eo / 10
        err = np.abs(OWN[:, t] - N[t]); eo = np.where(np.isnan(err), eo, (1 - r) * eo + r * np.nan_to_num(err))
    return acc, accown


def person_data(i, acc, accown):
    ts = np.array([t for t in range(t0, T) if OK[i, t] and not np.isnan(OWN[i, t])])
    y = np.clip(P[i, ts], 30, 90)
    mu = np.vstack([PR[:, ts], OWN[i, ts][None]])                         # (H, n_t)
    ac = np.vstack([acc[:, ts], accown[i, ts][None]])
    return y, mu, ac


def nll_i(x, y, mu, ac, theta):
    a = np.r_[0.0, x[:H]]; ls, m0, ls0 = x[H], x[H + 1], x[H + 2]
    sd, sd0 = np.exp(ls), np.exp(ls0)
    logit = a[:, None] + np.vstack([np.zeros((1, len(y))), theta * ac])  # (H+1, n_t)
    logpi = logit - logsumexp(logit, axis=0, keepdims=True)
    lf = np.vstack([-0.5 * np.log(2 * np.pi) - ls0 - 0.5 * ((y - m0) / sd0) ** 2,
                    -0.5 * np.log(2 * np.pi) - ls - 0.5 * ((y[None] - mu) / sd) ** 2])
    return -float(logsumexp(logpi + lf, axis=0).sum())


BND = [(-8, 8)] * H + [(np.log(0.5), np.log(20)), (30, 90), (np.log(2), np.log(40))]


def fit_i(y, mu, ac, theta, x0=None):
    starts = [x0] if x0 is not None else []
    starts += [np.r_[np.zeros(H), np.log(2.0), np.mean(y), np.log(np.std(y) + 1)],
               np.r_[np.full(H, -2.0), np.log(1.0), np.mean(y), np.log(np.std(y) + 1)]]
    best = None
    for s in starts:
        r = minimize(nll_i, s, args=(y, mu, ac, theta), method="L-BFGS-B", bounds=BND)
        if best is None or r.fun < best.fun: best = r
    return best.x, best.fun


def responsibilities(x, y, mu, ac, theta):
    a = np.r_[0.0, x[:H]]; ls, m0, ls0 = x[H], x[H + 1], x[H + 2]
    logit = a[:, None] + np.vstack([np.zeros((1, len(y))), theta * ac]); logpi = logit - logsumexp(logit, axis=0, keepdims=True)
    lf = np.vstack([-0.5 * np.log(2 * np.pi) - ls0 - 0.5 * ((y - m0) / np.exp(ls0)) ** 2,
                    -0.5 * np.log(2 * np.pi) - ls - 0.5 * ((y[None] - mu) / np.exp(ls)) ** 2])
    lp = logpi + lf; return np.exp(lp - logsumexp(lp, axis=0, keepdims=True))   # (H+1, n_t)


if __name__ == "__main__":
    import sys
    d10 = json.loads((PL.OUT / "s10_合理化联合模型_ρ逐人.json").read_text(encoding="utf-8"))
    honest = expit(np.array(d10["X"])[:, 7]) < 0.05
    THETAS = [-1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    if sys.argv[1] == "网格":
        r = float(sys.argv[2]); acc, accown = accuracy(r)
        D = [person_data(i, acc, accown) for i in range(n)]
        Xr = [None] * n; row = {}
        for th in THETAS:
            fi = np.zeros(n); Xt = []
            for i in range(n):
                x, fv = fit_i(*D[i], th, Xr[i]); fi[i] = fv; Xt.append(x)
            if th == 0.0: X0th = [x.copy() for x in Xt]
            Xr = Xt
            row[str(th)] = dict(全体=float(fi.sum()), 如实报告者=float(fi[honest].sum()), 逐人=fi.tolist(), X=[x.tolist() for x in Xt])
            print(f"r={r} θ={th}: 全体 {fi.sum():.1f}  如实 {fi[honest].sum():.1f}", flush=True)
            PL.save(dict(r=r, θ网格=THETAS, 各θ=row), f"s16_Arthur机制检验_r{r}.json")
        print("完成")
    elif sys.argv[1] == "精修":
        r = float(sys.argv[2]); acc, accown = accuracy(r)
        D = [person_data(i, acc, accown) for i in range(n)]
        cand = [[] for _ in range(n)]
        for rr in (0.1, 0.3, 0.6):
            fp = PL.OUT / f"s16_Arthur机制检验_r{rr}.json"
            if fp.exists():
                for k, v in json.loads(fp.read_text(encoding="utf-8"))["各θ"].items():
                    for i in range(n): cand[i].append(np.array(v["X"][i]))
        row = {}
        for th in THETAS:
            fi = np.zeros(n); Xt = []
            for i in range(n):
                vals = [nll_i(c, *D[i], th) for c in cand[i]]
                best = None
                for j in np.argsort(vals)[:2]:
                    rres = minimize(nll_i, cand[i][j], args=(*D[i], th), method="L-BFGS-B", bounds=BND)
                    if best is None or rres.fun < best.fun: best = rres
                fi[i] = best.fun; Xt.append(best.x)
            row[str(th)] = dict(全体=float(fi.sum()), 如实报告者=float(fi[honest].sum()), 逐人=fi.tolist(), X=[x.tolist() for x in Xt])
            print(f"精修 r={r} θ={th}: 全体 {fi.sum():.1f}  如实 {fi[honest].sum():.1f}", flush=True)
            PL.save(dict(r=r, θ网格=THETAS, 各θ=row), f"s16_Arthur机制检验_精修_r{r}.json")
        print("完成")
    else:
        out = {"规则": HN, "如实报告者人数": int(honest.sum())}
        mae = {HN[h]: float(np.nanmean(np.abs(PR[h, t0:] - N[t0:]))) for h in range(H - 1)}
        cat = {HN[h]: float(np.mean((PR[h, t0:] <= 60) == (N[t0:] <= 60))) for h in range(H - 1)}
        out["规则本身的准确度"] = {"平均绝对误差": mae, "挤/不挤判对比例": cat}
        prof = {}; best = None
        for r in (0.1, 0.3, 0.6):
            fp = PL.OUT / f"s16_Arthur机制检验_精修_r{r}.json"
            if not fp.exists(): fp = PL.OUT / f"s16_Arthur机制检验_r{r}.json"
            if not fp.exists(): continue
            d = json.loads(fp.read_text(encoding="utf-8"))["各θ"]
            f0 = d["0.0"]; ths = [float(k) for k in d]
            fb_k = min(d, key=lambda k: d[k]["全体"]); fh_k = min(d, key=lambda k: d[k]["如实报告者"])
            fi0 = np.array(f0["逐人"]); allf = {k: np.array(v["逐人"]) for k, v in d.items()}
            pos = np.min([allf[k] for k in d if float(k) > 0], 0); neg = np.min([allf[k] for k in d if float(k) < 0], 0)
            prof[str(r)] = dict(轮廓={k: dict(全体=v["全体"], 如实报告者=v["如实报告者"]) for k, v in d.items()},
                                全体_最佳θ=float(fb_k), 全体_LR=2 * (f0["全体"] - d[fb_k]["全体"]),
                                如实_最佳θ=float(fh_k), 如实_LR=2 * (f0["如实报告者"] - d[fh_k]["如实报告者"]),
                                逐人_θ为正显著更好=int(np.sum(pos + 1.92 < fi0)), 逐人_θ为负显著更好=int(np.sum(neg + 1.92 < fi0)))
            prof[str(r)]["全体_p"] = float(chi2.sf(max(prof[str(r)]["全体_LR"], 0), 1)); prof[str(r)]["如实_p"] = float(chi2.sf(max(prof[str(r)]["如实_LR"], 0), 1))
            if best is None or d[fb_k]["全体"] < best[0]: best = (d[fb_k]["全体"], r, float(fb_k), [np.array(x) for x in d[fb_k]["X"]])
        out["θ 的轮廓似然与检验"] = prof
        _, rb, thb, Xb = best
        acc, accown = accuracy(rb)
        resp = np.zeros((n, H + 1))
        for i in range(n):
            y, mu, ac = person_data(i, acc, accown); resp[i] = responsibilities(Xb[i], y, mu, ac, thb).mean(1)
        labs = ["其他"] + HN; main = resp.argmax(1)
        out["规则生态"] = dict(最佳r=rb, 最佳θ=thb, 各规则的平均后验份额={labs[k]: float(resp[:, k].mean()) for k in range(H + 1)},
                           以该规则为主的人数={labs[k]: int((main == k).sum()) for k in range(H + 1)},
                           如实报告者_各规则份额={labs[k]: float(resp[honest, k].mean()) for k in range(H + 1)})
        print(json.dumps(out, ensure_ascii=False)[:3000], flush=True)
        PL.save(dict(out, 逐人后验份额=resp), "s16_Arthur机制检验.json")
        print("完成")
