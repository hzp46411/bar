# -*- coding: utf-8 -*-
"""
第 42 步：MF 的样本外增益——是 MF 本身，还是惯性设得过简？
  背景：外部报告称"MB + 惯性"加入 MF 后样本外好 247 nats、91/100 人受益（p = 3.95e−15）。
        s21 中一步惯性下 MF 的群体增益为 240 iBIC，换成 choice kernel 习惯痕迹后只剩 1.7。
  模型（s22 的标准公式，层级 EM）：
    一步惯性：IB（惯性 + 世界模型）对 IFB（再加 MF）
    习惯痕迹：HB（choice kernel + 世界模型）对 HFB（再加 MF）
  切分：前后半程（跨时间）；交错区组（每 40 轮一组，奇数组 / 偶数组；同一时期内）
    只用训练轮次的选择与预测估计，起点为该模型真实拟合的群体均值（不用个人信息）；
    在检验轮次上计算选择的负对数似然（状态用真实历史从第 1 轮跑起）
  报告：MF 的样本外增益（两折之和，nats；负 = MF 更好）、受益人数、逐人配对检验（Wilcoxon、t）
用法：python3 s42_MF样本外检验.py 拟合 <模型> <前半|后半|奇|偶> | 汇总
输出：结果/s42_<模型>_<折>.json、结果/s42_MF样本外检验.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr
from scipy.stats import wilcoxon, ttest_1samp
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s22", str(pathlib.Path(__file__).with_name("s22_习惯痕迹审计.py")))
S22 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S22)
S21, S20 = S22.S21, S22.S20
T, n, G, S, OKP, CROWD_PREV = S22.T, S22.n, S22.G, S22.S, S22.OKP, S22.CROWD_PREV
CW = np.ones(T)
FOLDS = {"前半": np.arange(T) < T // 2, "后半": np.arange(T) >= T // 2,
         "奇": (np.arange(T) // 40) % 2 == 1, "偶": (np.arange(T) // 40) % 2 == 0}


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """同 s22 的 run；选择似然乘以轮次权重 CW。"""
    rows = X.shape[0]
    G_ = G if G_ is None else G_; S_ = S if S_ is None else S_
    b, wI, wF, aF, wB = X[:, 0], X[:, 1], X[:, 2], expit(X[:, 3]), X[:, 4]
    eta, delta, sig, aH = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7]), expit(X[:, 8])
    Mn, Mc = 60 + delta / 2, 60 - delta / 2
    QG, QS = np.full(rows, 0.5), np.full(rows, 0.35)
    H = np.zeros(rows); nll = np.zeros(rows)
    for t in range(T):
        sp = CROWD_PREV[t]
        Mp = Mc if sp else Mn
        p = ndtr((60.5 - Mp) / sig)
        z = b + wI * H + wF * (QG - QS) + wB * (1.7 * p - 0.7)
        a = A_[:, t]
        nll += CW[t] * (np.logaddexp(0, z) - a * z)
        nll += OKP_[:, t] * (0.5 * np.log(2 * np.pi) + np.log(sig) + 0.5 * ((P_[:, t] - Mp) / sig) ** 2)
        if sp:
            Mc = Mc + eta * (S20.N[t] - Mc)
        else:
            Mn = Mn + eta * (S20.N[t] - Mn)
        r = a * G_[:, t] + (1 - a) * 0.7 * S_[:, t]
        QG = QG + aF * a * (r - QG); QS = QS + aF * (1 - a) * (r - QS)
        H = H + aH * (2 * a - 1 - H)
    return nll


S21.run = run

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "拟合":
        m, fold = sys.argv[2], sys.argv[3]
        train = FOLDS[fold]
        r = json.loads((PL.OUT / f"s22_真实_{m}.json").read_text(encoding="utf-8"))
        mu, s2 = np.array(r["mu"]), np.array(r["s2"])
        CW[:] = train; S21.OKP = OKP * train
        logf = open(PL.W / "日志" / f"s42_{m}_{fold}.log", "a", encoding="utf-8")
        log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
        R = S22.em(m, S20.A, S20.P, np.tile(mu, (n, 1)), mu, s2, log)
        CW[:] = ~train; test = run(R["X"], S20.A, S20.P, np.zeros_like(OKP))
        PL.save(dict(模型=m, 训练=fold, 检验选择NLL=test, X=R["X"]), f"s42_{m}_{fold}.json")
    else:
        L = lambda m, f: np.array(json.loads((PL.OUT / f"s42_{m}_{f}.json").read_text(encoding="utf-8"))["检验选择NLL"])
        out = {}
        for scheme, folds in (("前后半程", ("前半", "后半")), ("交错区组", ("奇", "偶"))):
            out[scheme] = {}
            for base, full in (("IB", "IFB"), ("HB", "HFB")):
                d = sum(L(full, f) - L(base, f) for f in folds)        # 负 = 加入 MF 后检验更好
                out[scheme][f"{full} − {base}"] = dict(样本外增益_nats=round(float(d.sum()), 1), MF受益人数=int((d < 0).sum()),
                                                      Wilcoxon_p=float(f"{wilcoxon(d).pvalue:.3g}"), t检验_p=float(f"{ttest_1samp(d, 0).pvalue:.3g}"),
                                                      每人中位数=round(float(np.median(d)), 3))
            out[scheme]["各模型检验NLL"] = {m: round(float(sum(L(m, f).sum() for f in folds)), 1) for m in ("IB", "IFB", "HB", "HFB")}
        PL.save(out, "s42_MF样本外检验.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
