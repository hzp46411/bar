# -*- coding: utf-8 -*-
"""
第 43 步：按定稿的 Results 框架补齐三个小项（只读已有结果或做少量模拟，不重新拟合）
  类型（3.1.3、3.4.3）：不依赖模型的预测方向类型，与 s31 同一口径
     预测侧 δ̂ = 上轮不挤之后的平均预测 − 上轮挤之后的平均预测（> 0 = 预期反转，< 0 = 外推）
     逐人 Welch t，|t| > 1.96 判为反转 / 外推，其余为无方向；全程计数、前后半程的转移表、各类型的选择侧状态效应
  自稳（3.6.4）：H + γ + σ（s38 联合估计）的扰动自稳增益，与 s41 中 HB + γ + σ 的结果并列；另做 H + γ + σ 去 γ（保持水平）
     自稳增益 = 1 − 平移 / 无反馈平移（同 s41 的 gain）
  对称表（3.6.2）：{H, F, B, HB} × {无共同成分（s37）, 有共同成分（s38 联合估计）}；两步法结果（s37）作对照
  描述（3.1.1）：拥挤比例、状态转移概率 P(S′|S)、个人得分（去且不挤 1 分、不去且挤 0.7 分）、去的比例、换选择比例，
     及换选择比例、上轮挤后去的比例与得分的相关；随机基线 = 每轮以 .5 概率去的期望得分
  拟合优度（3.2 / 3.3）：8 个标准模型（s22 的真实拟合，参数不重新估计）的选择部分 McFadden 伪 R² = 1 − NLL_选择 / NLL_随机，
     NLL_随机 = 选择数 × ln 2；全体与逐人
用法：python3 s43_框架补充.py 类型 | 自稳 | 对称表 | 描述 | 拟合优度
输出：结果/s43_类型.json、结果/s43_自稳.json、结果/s43_对称表.json、结果/s43_描述.json、结果/s43_拟合优度.json
"""
import sys, json, importlib.util, pathlib, collections
import numpy as np
import pred_lib as PL

J = lambda f: json.loads((PL.OUT / f).read_text(encoding="utf-8"))


def _load(fname, mod):
    spec = importlib.util.spec_from_file_location(mod, str(pathlib.Path(__file__).with_name(fname)))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    return M


def types():
    T, A, P, OK = PL.T, PL.A, PL.P, PL.OK
    N = PL.N.astype(float)
    CP = (np.r_[PL._raw.iloc[0, 1:].astype(float).values[-T - 1], N[:-1]] >= 61)          # 上一轮挤（与 s31 相同）

    def classify(t_mask):
        d, tv, ch = (np.full(A.shape[0], np.nan) for _ in range(3))
        for i in range(A.shape[0]):
            ok = OK[i] & t_mask
            x, y = P[i, ok & ~CP], P[i, ok & CP]
            if len(x) >= 5 and len(y) >= 5:
                d[i] = x.mean() - y.mean()
                tv[i] = d[i] / np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            ch[i] = A[i, t_mask & CP].mean() - A[i, t_mask & ~CP].mean()
        ty = np.where(tv > 1.96, "反转", np.where(tv < -1.96, "外推", "无方向"))
        return d, tv, ch, ty

    allr = np.ones(T, bool); half = np.arange(T) < T // 2
    d, tv, ch, ty = classify(allr)
    _, _, _, t1 = classify(half); _, _, _, t2 = classify(~half)
    tr = collections.Counter(zip(t1, t2))
    out = dict(口径="预测侧 δ̂ = 上轮不挤后平均预测 − 上轮挤后平均预测；逐人 Welch t，|t| > 1.96",
               全程计数={k: int((ty == k).sum()) for k in ("反转", "外推", "无方向")},
               前后半程转移={f"{a}→{b}": int(tr[(a, b)]) for a in ("反转", "外推", "无方向") for b in ("反转", "外推", "无方向")},
               方向翻转人数=int(tr[("反转", "外推")] + tr[("外推", "反转")]),
               两半同为反转=int(tr[("反转", "反转")]), 两半同为外推=int(tr[("外推", "外推")]),
               各类型的选择侧状态效应={k: round(float(np.nanmean(ch[ty == k])), 4) for k in ("反转", "外推", "无方向")},
               选择侧状态效应说明="上轮挤之后去的比例 − 上轮不挤之后去的比例；反转者应 > 0，外推者应 < 0",
               全体选择侧状态效应均值=round(float(np.nanmean(ch)), 4))
    PL.save(out, "s43_类型.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def self_stab():
    S41 = _load("s41_联合估计ABM.py", "s41"); S27 = S41.S27
    r = J("s38_联合估计_H.json"); g, s = r["γ"], r["σ"]
    X = S27.load("s38:H"); X[:, 13] = -g
    Y = X.copy(); Y[:, 13] = 0.0; Y = S27.recenter(X, Y)
    out = dict(说明="自稳增益 = 1 − 平移 / 无反馈平移；同 s41 的 gain（各 200 次闭环模拟）",
               H_γσ=dict(模型="H + γ + σ（s38 联合估计）", γ=g, σ=s, 结果=S41.gain(X, s)),
               H_γσ_去γ=dict(模型="H + γ + σ，去 γ（保持水平）", 结果=S41.gain(Y, s)),
               HB_γσ_对照=J("s41_联合估计ABM.json")["D 扰动自稳"])
    PL.save(out, "s43_自稳.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def sym_table():
    s37 = J("s37_对称联合检验.json")
    keep = ("马氏距离平方", "联合p", "合成对数似然", "D", "D的p", "换选择比例", "换选择p")
    pick = lambda d: {k: d[k] for k in keep if k in d}
    out = {"说明": "无共同成分取 s37；有共同成分取 s38 的联合估计（个体参数与 γ、σ 一起估计）；两步法（s37）只作对照"}
    for m in ("H", "F", "B", "HB"):
        row = {"无共同成分": pick(s37["联合检验"][m])}
        f = PL.OUT / f"s38_联合检验_{m}.json"
        if f.exists():
            e = J(f"s38_联合估计_{m}.json")
            row["有共同成分（联合估计）"] = dict(γ=round(e["γ"], 4), σ=round(e["σ"], 4), **pick(J(f"s38_联合检验_{m}.json")["联合检验"]))
        else:
            row["有共同成分（联合估计）"] = "尚未完成"
        row["有共同成分（两步法，对照）"] = dict(γ=round(s37["共同因素"][m]["γ"], 4), σ=round(s37["共同因素"][m]["σ"], 4),
                                        **pick(s37["联合检验"][f"{m} + γ + σ"]))
        out[m] = row
    PL.save(out, "s43_对称表.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def describe():
    from scipy.stats import pearsonr
    T, A = PL.T, PL.A.astype(float)
    N = PL.N.astype(float)
    Nprev = np.r_[PL._raw.iloc[0, 1:].astype(float).values[-T - 1], N[:-1]]
    cr, cp = N >= 61, Nprev >= 61
    G, S = (~cr).astype(float), cr.astype(float)                                       # 去且不挤 1 分；不去且挤 0.7 分
    score = (A * G + (1 - A) * 0.7 * S).sum(1)
    rand = float((0.5 * G + 0.5 * 0.7 * S).sum())
    sw = (A[:, 1:] != A[:, :-1]).mean(1)
    go_after_cr = np.array([A[i, cp].mean() for i in range(A.shape[0])])
    go_after_nc = np.array([A[i, ~cp].mean() for i in range(A.shape[0])])
    r = lambda x, y: dict(r=round(float(pearsonr(x, y)[0]), 3), p=float(f"{pearsonr(x, y)[1]:.3g}"))
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [25, 50, 75])]
    out = dict(轮数=T, 人数=int(A.shape[0]),
               群体=dict(平均人数=round(float(N.mean()), 2), SD=round(float(N.std(ddof=1)), 2), 拥挤比例=round(float(cr.mean()), 3),
                       转移概率={"不挤→不挤": round(float((~cr[cp == 0]).mean()), 3), "不挤→挤": round(float(cr[cp == 0].mean()), 3),
                             "挤→不挤": round(float((~cr[cp == 1]).mean()), 3), "挤→挤": round(float(cr[cp == 1].mean()), 3)}),
               个人=dict(得分_均值=round(float(score.mean()), 1), 得分_SD=round(float(score.std(ddof=1)), 1), 得分_四分位=q(score),
                       随机基线得分=round(rand, 1), 高于随机基线人数=int((score > rand).sum()),
                       去的比例_四分位=q(A.mean(1)), 换选择比例_四分位=q(sw),
                       上轮挤后去的比例_四分位=q(go_after_cr), 上轮不挤后去的比例_四分位=q(go_after_nc)),
               与得分的相关=dict(换选择比例=r(sw, score), 去的比例=r(A.mean(1), score),
                           上轮挤后去的比例=r(go_after_cr, score), 上轮不挤后去的比例=r(go_after_nc, score)))
    PL.save(out, "s43_描述.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))


def pseudo_r2():
    S22 = _load("s22_习惯痕迹审计.py", "s22")
    A = S22.S20.A
    base = np.log(2) * A.shape[1]
    out = {"说明": "选择部分的 McFadden 伪 R² = 1 − NLL_选择 / (选择数 × ln 2)；参数取 s22 的层级拟合（不重新估计）"}
    for m in ["0", "H", "F", "B", "HF", "HB", "FB", "HFB"]:
        X = np.array(J(f"s22_真实_{m}.json")["X"])
        nll = S22.run(X, A, S22.S20.P, np.zeros_like(S22.OKP))
        r2 = 1 - nll / base
        out[m] = dict(全体=round(float(1 - nll.sum() / (base * len(nll))), 4), 逐人中位数=round(float(np.median(r2)), 4),
                      逐人四分位=[round(float(x), 4) for x in np.percentile(r2, [25, 75])])
    PL.save(out, "s43_拟合优度.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"类型": types, "自稳": self_stab, "对称表": sym_table, "描述": describe, "拟合优度": pseudo_r2}[sys.argv[1]]()
