# -*- coding: utf-8 -*-
"""
第 40 步：两个系统怎样组合——加性整合 vs 混合切换（主模型 HB 的稳健性检验）
  加性（主模型）：logit P(去) = b + w_I·H + w_B·v_B，v_B = 1.7p − 0.7（每次选择是两个系统的加权合成）
  混合：P(去) = π·expit(b + w_I·H) + (1 − π)·expit(b + w_B·v_B)（每次选择以概率 π 听习惯，否则听世界模型）
        每人多一个 π（logit 尺度）；两个模型不嵌套
  比较：
    A 群体层面 iBIC（层级 EM，与 s22 相同的设置；加性取 s22 的结果）
    B 冲突轮次：习惯建议去（H > 0）而世界模型建议不去（p < .5），或反之；在冲突与一致轮次上分别比较两个模型的选择对数似然
      加性模型在冲突时给出折中的概率，混合模型给出两个系统概率的线性插值；冲突轮次最能区分二者
  起点：s22 中 HB 的解，π 逐人取网格（.2、.5、.8）
用法：python3 s40_加性与混合.py 拟合 | 汇总
输出：结果/s40_HB混合.json、结果/s40_加性与混合.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr, logit
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s22", str(pathlib.Path(__file__).with_name("s22_习惯痕迹审计.py")))
S22 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S22)
S21, S20 = S22.S21, S22.S20
T, n, G, S, OKP, CROWD_PREV, ONE_STEP = S22.T, S22.n, S22.G, S22.S, S22.OKP, S22.CROWD_PREV, S22.ONE_STEP
M = "HBmix"
FREE = dict(S22.FREE); FREE[M] = S22.FREE["HB"] + [9]
ULO = np.r_[S22.ULO, -5.0]; UHI = np.r_[S22.UHI, 5.0]
OUT_P = None                                                            # 需要逐轮概率时设为列表


def system_terms(X, A_):
    """真实历史上两个系统的 logit（行, T）与世界模型的 p（供冲突分析）。"""
    rows = X.shape[0]
    b, wI, wB = X[:, 0], X[:, 1], X[:, 4]
    eta, delta, sig, aH = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7]), expit(X[:, 8])
    Mn, Mc = 60 + delta / 2, 60 - delta / 2; H = np.zeros(rows)
    zH, zB, pp, HH = (np.zeros((rows, T)) for _ in range(4))
    for t in range(T):
        sp = CROWD_PREV[t]; Mp = Mc if sp else Mn; p = ndtr((60.5 - Mp) / sig)
        zH[:, t] = b + wI * H; zB[:, t] = b + wB * (1.7 * p - 0.7); pp[:, t] = p; HH[:, t] = H
        if sp:
            Mc = Mc + eta * (S20.N[t] - Mc)
        else:
            Mn = Mn + eta * (S20.N[t] - Mn)
        H = H + aH * (2 * A_[:, t] - 1 - H)
    return zH, zB, pp, HH


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """混合模型的负对数似然（选择 + 预测）；X：(行, 10)，第 10 列为 logit π。"""
    rows = X.shape[0]
    b, wI, wB = X[:, 0], X[:, 1], X[:, 4]
    eta, delta, sig, aH, pi = expit(X[:, 5]), X[:, 6], np.exp(X[:, 7]), expit(X[:, 8]), expit(X[:, 9])
    Mn, Mc = 60 + delta / 2, 60 - delta / 2
    H = np.zeros(rows); nll = np.zeros(rows)
    for t in range(T):
        sp = CROWD_PREV[t]
        Mp = Mc if sp else Mn
        p = ndtr((60.5 - Mp) / sig)
        P = pi * expit(b + wI * H) + (1 - pi) * expit(b + wB * (1.7 * p - 0.7))
        P = np.clip(P, 1e-12, 1 - 1e-12)
        a = A_[:, t]
        nll -= a * np.log(P) + (1 - a) * np.log(1 - P)
        nll += OKP_[:, t] * (0.5 * np.log(2 * np.pi) + np.log(sig) + 0.5 * ((P_[:, t] - Mp) / sig) ** 2)
        if sp:
            Mc = Mc + eta * (S20.N[t] - Mc)
        else:
            Mn = Mn + eta * (S20.N[t] - Mn)
        H = H + aH * (2 * a - 1 - H)
    return nll


def to_X(U, m):
    X = np.zeros((U.shape[0], 10)); X[:, 8] = ONE_STEP
    for j, col in enumerate(FREE[m]):
        X[:, col] = np.exp(U[:, j]) if col in S22.LOGW else U[:, j]
    return X


def to_U(X, m):
    idx = FREE[m]
    U = np.column_stack([np.log(np.maximum(X[:, c], np.exp(-8))) if c in S22.LOGW else X[:, c] for c in idx])
    return np.clip(U, ULO[idx], UHI[idx])


S21.run, S21.FREE, S21.ULO, S21.UHI, S21.to_X, S21.to_U = run, FREE, ULO, UHI, to_X, to_U
S22.FREE, S22.ULO, S22.UHI, S22.to_X = FREE, ULO, UHI, to_X


def choice_ll(P, A_):
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return A_ * np.log(P) + (1 - A_) * np.log(1 - P)


if __name__ == "__main__":
    mode = sys.argv[1]
    Ntot = float((T + OKP.sum(1)).sum())
    if mode == "拟合":
        logf = open(PL.W / "日志" / "s40_HB混合.log", "a", encoding="utf-8")
        log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
        r = json.loads((PL.OUT / "s22_真实_HB.json").read_text(encoding="utf-8"))
        X = np.zeros((n, 10)); X[:, :9] = np.array(r["X"])
        f = []
        for c in (-1.4, 0.0, 1.4):
            Xa = X.copy(); Xa[:, 9] = c; f.append(run(Xa, S20.A, S20.P, OKP))
        X[:, 9] = np.array([-1.4, 0.0, 1.4])[np.argmin(np.array(f), 0)]
        U = to_U(X, M)
        R = S22.em(M, S20.A, S20.P, U, U.mean(0), np.maximum(U.var(0), 1e-2), log)
        PL.save(dict(模型=M, iBIC=float(-2 * R["LE"].sum() + 2 * len(FREE[M]) * np.log(Ntot)), **R), "s40_HB混合.json")
        log("完成")
    else:
        add = json.loads((PL.OUT / "s22_真实_HB.json").read_text(encoding="utf-8"))
        mix = json.loads((PL.OUT / "s40_HB混合.json").read_text(encoding="utf-8"))
        Xa, Xm = np.array(add["X"]), np.array(mix["X"])
        A = S20.A
        zH, zB, pp, HH = system_terms(Xa, A)
        P_add = expit(zH + zB - Xa[:, [0]])                             # 加性：b + w_I·H + w_B·v_B
        zHm, zBm, ppm, HHm = system_terms(Xm[:, :9], A)
        pim = expit(Xm[:, [9]])
        P_mix = pim * expit(zHm) + (1 - pim) * expit(zBm)
        conflict = ((HH > 0) & (pp < 0.5)) | ((HH < 0) & (pp > 0.5))
        agree = ((HH > 0) & (pp > 0.5)) | ((HH < 0) & (pp < 0.5))
        la, lm = choice_ll(P_add, A), choice_ll(P_mix, A)
        per = lambda L, msk: np.array([L[i][msk[i]].sum() for i in range(n)])
        out = dict(iBIC=dict(加性=round(add["iBIC"], 1), 混合=round(mix["iBIC"], 1), 混合减加性=round(mix["iBIC"] - add["iBIC"], 1)),
                   π=dict(四分位数=[round(float(x), 3) for x in np.percentile(pim, [25, 50, 75])]),
                   冲突轮次=dict(占比=round(float(conflict.mean()), 3), 加性对数似然=round(float(la[conflict].sum()), 1),
                              混合对数似然=round(float(lm[conflict].sum()), 1), 混合更好的人数=int((per(lm, conflict) > per(la, conflict)).sum()),
                              观测去的比例_习惯建议去=round(float(A[conflict & (HH > 0)].mean()), 3),
                              加性预测=round(float(P_add[conflict & (HH > 0)].mean()), 3), 混合预测=round(float(P_mix[conflict & (HH > 0)].mean()), 3),
                              观测去的比例_世界模型建议去=round(float(A[conflict & (HH < 0)].mean()), 3),
                              加性预测_2=round(float(P_add[conflict & (HH < 0)].mean()), 3), 混合预测_2=round(float(P_mix[conflict & (HH < 0)].mean()), 3)),
                   一致轮次=dict(占比=round(float(agree.mean()), 3), 加性对数似然=round(float(la[agree].sum()), 1), 混合对数似然=round(float(lm[agree].sum()), 1)))
        PL.save(out, "s40_加性与混合.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
