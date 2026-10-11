# -*- coding: utf-8 -*-
"""
第 38 步：个体参数与共同因素（γ、σ）的联合估计，及其 ABM 联合检验
  背景：s36 用两步法（个体参数固定为不含共同因素时的估计，再估计 γ、σ）。世界模型 B 对上一轮状态的反应与 γ（全体对上一轮偏离的
        共同反应）都依赖上一轮的人数；两步法下 B 的参数可能吸收了本属于共同反应的部分，或反过来，二者的分配可能有偏。
        联合估计让个体参数与 γ、σ 在同一个似然里一起确定。
  模型：z_it = d_it(θ_i) − γ·(N_{t−1} − 60)/10 + σ·ε_t，ε_t ~ N(0, 1)（同一轮所有人共享）；θ_i 有群体先验（层级）
  算法（条件期望最大化，交替进行；每一轮存检查点）：
    1 ε 的后验：给定当前 θ、γ、σ，每轮用 40 点高斯–埃尔米特积分求 ε_t 的后验，压缩为 K = 8 个等权的后验分位点
    2 更新 θ：层级 EM（s22 的 em，最多 10 轮，从上一轮的解出发），选择似然取 K 个共同偏移下的平均（期望完全数据对数似然）
    3 更新 γ、σ：给定 θ，对 ε 积分的边际似然最大化（同 s36）
    重复至 |Δγ|、|Δσ| < 0.005 或 6 轮
  模型：HB（主模型）与 H（对照：只有习惯 + 共同因素在两步法下通过了联合检验）
  ABM：联合估计的参数 + γ、σ，1000 次闭环模拟，用 s36 的事先固定的联合检验（指纹：均值、SD、ACF1–3）
用法：python3 s38_共同因素联合估计.py 估计 <HB|H>     （断点续跑：从 检查点/s38_<模型>_轮<k>.json 继续）
      python3 s38_共同因素联合估计.py 检验 <HB|H>
输出：结果/s38_联合估计_<模型>.json、结果/s38_联合检验_<模型>.json
"""
import sys, json, time, importlib.util, pathlib
import numpy as np
from scipy.special import expit, ndtr, logit
import pred_lib as PL


def _load(fname, mod):
    spec = importlib.util.spec_from_file_location(mod, str(pathlib.Path(__file__).with_name(fname)))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    return M


S36 = _load("s36_共同因素层.py", "s36")
S27 = S36.S27
S22 = _load("s22_习惯痕迹审计.py", "s22")
S21, S20 = S22.S21, S22.S20
T, n, G, S, OKP, CROWD_PREV = S22.T, S22.n, S22.G, S22.S, S22.OKP, S22.CROWD_PREV
LAG = S36.LAG
K = 8
OFF = np.zeros((T, K))                                                  # 每轮 K 个共同偏移 −γ·lag + σ·ε（等权）
CK = PL.W / "检查点"


def run(X, A_, P_, OKP_, G_=None, S_=None, rng=None):
    """同 s22 的 run；选择似然取每轮 K 个共同偏移下的平均（OFF 为模块全局）。"""
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
        zk = z[:, None] + OFF[t][None, :]
        nll += (np.logaddexp(0, zk) - a[:, None] * zk).mean(1)
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


def logits(m, X9):
    """当前 θ 在真实历史上的个体 logit d_it（不含共同因素）。"""
    Y = np.zeros((n, 18)); Y[:, :9] = X9; Y[:, 11] = -50.0; Y[:, 12] = 0.0
    _, P = S27.open_loop(Y)
    return logit(np.clip(P, 1e-9, 1 - 1e-9))


def posterior_offsets(d, g, s):
    """ε_t 的后验（40 点积分），压缩为 K 个等权的后验分位点；返回 (T, K) 的共同偏移。"""
    z = d[None] - g * LAG + s * S36.NODES[:, None, None]
    L = (S36.A[None] * z - np.logaddexp(0, z)).sum(1)                   # (节点, 轮)
    L = L - L.max(0); w = S36.W[:, None] * np.exp(L); w = w / w.sum(0)
    cdf = np.cumsum(w, 0); qs = (np.arange(K) + 0.5) / K
    eps = np.array([[S36.NODES[np.searchsorted(cdf[:, t], q)] for q in qs] for t in range(T)])
    return -g * LAG[:, None] + s * eps


def total_marginal(m, X9, g, s):
    """联合模型的对数似然：选择（对 ε 积分）+ 预测。"""
    d = logits(m, X9)
    OFF[:] = 0.0
    fc = run(X9, S20.A, S20.P, OKP).sum() - run(X9, S20.A, S20.P, np.zeros_like(OKP)).sum()
    return S36.loglik(d, S36.A, g, s) - fc


if __name__ == "__main__":
    mode, m = sys.argv[1], sys.argv[2]
    if mode == "估计":
        logf = open(PL.W / "日志" / f"s38_{m}.log", "a", encoding="utf-8")
        log = lambda s_: (logf.write(s_ + "\n"), logf.flush(), print(s_, flush=True))
        r0 = json.loads((PL.OUT / f"s22_真实_{m}.json").read_text(encoding="utf-8"))
        s36 = json.loads((PL.OUT / ("s36_共同因素层.json" if m == "HB" else "s37_对称联合检验.json")).read_text(encoding="utf-8"))
        g, s = (s36["A 估计"]["γ+σ"]["γ"], s36["A 估计"]["γ+σ"]["σ"]) if m == "HB" else (s36["共同因素"][m]["γ"], s36["共同因素"][m]["σ"])
        X9 = np.array(r0["X"]); U = S22.to_U(X9, m); mu, s2 = np.array(r0["mu"]), np.array(r0["s2"])
        hist = [dict(轮=0, γ=g, σ=s, 联合对数似然=total_marginal(m, X9, g, s))]
        start = 1
        for k in range(6, 0, -1):                                      # 断点续跑
            f = CK / f"s38_{m}_轮{k}.json"
            if f.exists():
                c = json.loads(f.read_text(encoding="utf-8"))
                X9, mu, s2, g, s, hist = np.array(c["X"]), np.array(c["mu"]), np.array(c["s2"]), c["γ"], c["σ"], c["历史"]
                U = S22.to_U(X9, m); start = k + 1; log(f"从第 {k} 轮的检查点继续"); break
        log(f"{m} 起点：γ {g:.4f}，σ {s:.4f}，联合对数似然 {hist[0]['联合对数似然']:.2f}")
        for it in range(start, 7):
            t0 = time.time()
            OFF[:] = posterior_offsets(logits(m, X9), g, s)
            R = S22.em(m, S20.A, S20.P, U, mu, s2, log, max_it=10)
            X9, mu, s2 = R["X"], R["mu"], R["s2"]; U = S22.to_U(X9, m)
            e = S36.estimate(logits(m, X9), S36.A)["γ+σ"]
            dg, ds = abs(e[0] - g), abs(e[1] - s); g, s = e[0], e[1]
            ll = total_marginal(m, X9, g, s)
            hist.append(dict(轮=it, γ=g, σ=s, 联合对数似然=ll))
            log(f"第 {it} 轮：γ {g:.4f}，σ {s:.4f}，联合对数似然 {ll:.2f}（{time.time() - t0:.0f}s）")
            (CK / f"s38_{m}_轮{it}.json").write_text(json.dumps(dict(X=X9.tolist(), mu=mu.tolist(), s2=s2.tolist(), γ=g, σ=s, 历史=hist),
                                                               ensure_ascii=False), encoding="utf-8")
            if dg < 0.005 and ds < 0.005:
                break
        q = lambda v: [round(float(x), 3) for x in np.percentile(v, [25, 50, 75])]
        X0 = np.array(r0["X"])
        par = {nm: dict(两步=q(X0[:, c]), 联合=q(X9[:, c]), 跨人相关=round(float(np.corrcoef(X0[:, c], X9[:, c])[0, 1]), 3))
               for nm, c in (("b", 0), ("w_I", 1), ("w_B", 4), ("δ", 6), ("lnσ_预测", 7)) if c in S22.FREE[m]}
        PL.save(dict(模型=m, X=X9, mu=mu, s2=s2, γ=g, σ=s, 历史=hist, 参数_两步对联合=par), f"s38_联合估计_{m}.json")
        log("完成")
    else:
        r = json.loads((PL.OUT / f"s38_联合估计_{m}.json").read_text(encoding="utf-8"))
        g, s = r["γ"], r["σ"]
        S27.SRC[m + "联合"] = f"s38_联合估计_{m}.json"; S27.FIXED[m + "联合"] = (-50.0, 0.0)
        X = S27.load(m + "联合"); X[:, 13] = -g
        OBS = S36.fingerprint(S27.N_REAL[None])[0]
        OBS_D, OBS_SW = S36.micro(S27.N_REAL[None], S36.A[None].astype(np.int8))
        fps, Ds, sws, css, acf4 = [], [], [], [], []
        for b in range(4):
            Ns, As = S27.simulate(X, 250, 100 + 10 * b, s)
            fps.append(S36.fingerprint(Ns)); d_, sw_ = S36.micro(Ns, As); Ds.append(d_); sws.append(sw_)
            css.append((Ns >= 61).mean(1)); acf4.append(S27.acf(Ns, 4))
        SIM = tuple(np.concatenate(x) for x in (fps, Ds, sws, css, acf4))
        out = dict(模型=m + " + γ + σ（联合估计）", γ=g, σ=s, 联合检验=S36.joint(OBS, OBS_D[0], OBS_SW[0], *SIM))
        PL.save(out, f"s38_联合检验_{m}.json")
        print(json.dumps(out, ensure_ascii=False, indent=1))
