# -*- coding: utf-8 -*-
"""
第 24 步：逐人最大似然下的信息准则比较（AIC、HQC、BIC）：习惯痕迹 vs MF
  s20 的 8 个模型（一步惯性）已有逐人最大似然解；这里补上习惯痕迹的模型（H、HF、HB、HFB，以及 s23 的 HDB、HDFB），
  前向模型用 s23 的 run（习惯痕迹 + 慢漂移痕迹；不含的成分权重为 0）。
  准则（逐人计算后求和；n_i = 选择数 + 有效预测数，与 s20 相同）：
    AIC = 2·NLL + 2k        HQC = 2·NLL + 2k·ln(ln n_i)        BIC = 2·NLL + k·ln n_i
  起点（保证大模型不差于它嵌套的小模型）：s22 / s23 层级估计的解；对应一步惯性模型的 s20 解（logit α_H = 7，即 α_H ≈ 1）；
        嵌套的小模型的最大似然解
用法：python3 s24_信息准则比较.py 拟合 <模型>   （先 H，再 HF、HB，再 HFB；HDB、HDFB 需 s23 的结果）
      python3 s24_信息准则比较.py 汇总
输出：结果/s24_最大似然_<模型>.json、结果/s24_信息准则汇总.json
"""
import sys, json, time, importlib.util, pathlib
import numpy as np
from scipy.optimize import minimize
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s23", str(pathlib.Path(__file__).with_name("s23_慢漂移检验.py")))
S23 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S23)
S20, run, n, T, OKP, G, S = S23.S20, S23.run, S23.n, S23.T, S23.OKP, S23.G, S23.S
A, P = S20.A, S20.P
ONE = S20.MODELS                                                         # 一步惯性的 8 个模型（s20）
HAB = ["H", "HF", "HB", "HFB", "HDB", "HDFB"]
FREE = {m: [0, 5, 6, 7, 1, 8] + ([9, 10] if "D" in m else []) + ([2, 3] if "F" in m else []) + ([4] if "B" in m else []) for m in HAB}
SUB = {"H": [], "HF": ["H"], "HB": ["H"], "HFB": ["HF", "HB"], "HDB": ["HB"], "HDFB": ["HDB", "HFB"]}
LO = np.r_[S20.LO, -7.0, -10.0, -7.0]
HI = np.r_[S20.HI, 7.0, 10.0, 7.0]
NOBS = T + OKP.sum(1)


def base():
    X = np.zeros((n, 11)); X[:, 8] = S23.ONE_STEP
    return X


def fit(m, starts):
    idx = FREE[m]; k = len(idx); h = 1e-5
    st = lambda M_: np.tile(M_, (k + 1, 1)); As, Ps, Os, Gs, Ss = st(A), st(P), st(OKP), st(G), st(S)
    B0 = base()

    def obj(v):
        X = B0.copy(); X[:, idx] = v.reshape(n, k); Xs = st(X)
        for j, col in enumerate(idx):
            Xs[(j + 1) * n:(j + 2) * n, col] += h
        f = run(Xs, As, Ps, Os, Gs, Ss).reshape(k + 1, n)
        return f[0].sum(), ((f[1:] - f[0]) / h).T.ravel()
    bestX, bestf = B0.copy(), np.full(n, np.inf)
    for X0 in starts:
        res = minimize(obj, np.clip(X0[:, idx], LO[idx], HI[idx]).ravel(), jac=True, method="L-BFGS-B",
                       bounds=list(zip(np.tile(LO[idx], n), np.tile(HI[idx], n))), options={"maxiter": 3000})
        X = B0.copy(); X[:, idx] = res.x.reshape(n, k)
        f = run(X, A, P, OKP)
        better = f < bestf; bestX[better], bestf[better] = X[better], f[better]
    return bestX, bestf


def starts_for(m):
    out = []
    src = PL.OUT / (f"s23_真实_{m}.json" if "D" in m else f"s22_真实_{m}.json")
    Xh = np.array(json.loads(src.read_text(encoding="utf-8"))["X"])
    out.append(Xh if Xh.shape[1] == 11 else S23.widen(Xh))
    one = m.replace("H", "I").replace("D", "")
    X1 = np.array(json.loads((PL.OUT / "s20_真实拟合.json").read_text(encoding="utf-8"))["X"][one])
    X1 = S23.widen(np.column_stack([X1, np.full(n, 7.0)]))
    out.append(X1)
    for s_ in SUB[m]:
        out.append(np.array(json.loads((PL.OUT / f"s24_最大似然_{s_}.json").read_text(encoding="utf-8"))["X"]))
    return out


def crit(nll, k):
    return dict(AIC=2 * nll + 2 * k, HQC=2 * nll + 2 * k * np.log(np.log(NOBS)), BIC=2 * nll + k * np.log(NOBS))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "拟合":
        m = sys.argv[2]; t0 = time.time()
        X, f = fit(m, starts_for(m))
        PL.save(dict(模型=m, X=X, nll_i=f), f"s24_最大似然_{m}.json")
        print(f"{m}：NLL {f.sum():.1f}（{time.time() - t0:.0f}s）", flush=True)
    else:
        real = json.loads((PL.OUT / "s20_真实拟合.json").read_text(encoding="utf-8"))
        nll = {m: np.array(real["nll_i"][m]) for m in ONE}
        K = {m: len(S20.FREE[m]) for m in ONE}
        for m in HAB:
            p = PL.OUT / f"s24_最大似然_{m}.json"
            if p.exists():
                nll[m] = np.array(json.loads(p.read_text(encoding="utf-8"))["nll_i"]); K[m] = len(FREE[m])
        C = {m: crit(nll[m], K[m]) for m in nll}
        ms = list(nll)
        tot = {c: {m: round(float(C[m][c].sum()), 1) for m in ms} for c in ("AIC", "HQC", "BIC")}
        rel = {c: {m: round(v - min(tot[c].values()), 1) for m, v in tot[c].items()} for c in tot}
        cnt = {c: dict(zip(ms, np.bincount(np.argmin(np.column_stack([C[m][c] for m in ms]), 1), minlength=len(ms)).tolist())) for c in tot}
        pairs = [("H", "F"), ("HB", "FB"), ("HB", "IFB"), ("HFB", "HB"), ("HFB", "FB"), ("HB", "IB"), ("HDB", "HB"), ("HDFB", "HDB")]
        pair = {}
        for a, b in pairs:
            if a in C and b in C:
                pair[f"{a} 对 {b}"] = {c: dict(总差=round(float((C[a][c] - C[b][c]).sum()), 1),
                                              前者更好的人数=int(((C[a][c] - C[b][c]) < 0).sum())) for c in tot}
        out = dict(参数个数=K, NLL={m: round(float(nll[m].sum()), 1) for m in ms}, 总和=tot, 相对最优=rel, 逐人最优人数=cnt, 两两比较=pair)
        PL.save(out, "s24_信息准则汇总.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
