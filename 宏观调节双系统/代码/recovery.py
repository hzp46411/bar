# -*- coding: utf-8 -*-
"""
第 4 层 · 步骤 2：完整流程的恢复检验与零分布（开环：他人人数取真实数据）
  真值套（4）：用主模型 RG 的估计（个体参数、θ、λ、σ）模拟选择 → 从头联合估计 M0 与 RG
  零套（8）：用 M0 的估计（θ = 0）模拟 → 同样估计 M0 与 RG
输出：结果/恢复/<套名>.json（每套一个检查点）；全部完成后写 结果/恢复汇总.json
回答：θR、θG 能否恢复（偏差、相关）；θ = 0 时 θR 的零分布有多宽（给真实估计一个完整流程的 z）；
      含 σ 的似然比在零下是否校准（假阳性率）
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L

DIR = L.OUT / "恢复"
FIT = lambda n: json.loads((L.OUT / "拟合" / f"{n}.json").read_text(encoding="utf-8"))
SPEC_RG = L.Spec(ratio=L.MODS, gain=L.MODS, habit=True, push=["stab", "dev"])   # 主模型：HRGP（习惯痕迹 + 比例/增益调节 + 加法重复推力）
SPEC_0 = L.Spec(habit=True)                                  # 对照：习惯痕迹、无调节（H0）


def staged_fit(spec, A, X0, phi0, seed, ck, log):
    """分阶段拟合，每一段都存检查点：
    0 个体多起点（共用参数取起点）→ 1 全参数 L-BFGS（每 200 步存一次）→ 2 新共用参数下再做个体多起点 → 3 再精修。"""
    st = dict(np.load(ck)) if Path(ck).exists() else {}
    stage = int(st.get("stage", 0)); X = st.get("X"); phi = st.get("phi")
    save = lambda: np.savez(ck, stage=stage, X=X, phi=phi)
    if stage > 0:
        log(f"  从检查点继续：阶段 {stage}")
    while stage < 4:
        if stage in (0, 2):
            base = X0 if stage == 0 else X
            ph = phi0 if stage == 0 else phi
            X, _ = L.fit_individuals(spec, ph, A, L.G_REAL, L.S_REAL, L.ATT, [base] + L.basin_starts(base), seed=seed + 2 + stage)
            phi = np.array(ph, float); stage += 1; save()
        else:
            X, phi, f, info = L.joint_lbfgs(spec, A, L.G_REAL, L.S_REAL, L.ATT, X, phi, maxiter=200)
            if "LIMIT" not in info["message"].upper():
                stage += 1
            save()
            log(f"  阶段 {stage}：NLL = {f.sum():.3f}（{info['message'][:40]}）")
    f = L.run(X, spec, phi, A, L.G_REAL, L.S_REAL, L.ATT)
    return X, phi, f


def job(args):
    kind, i = args
    name = f"{kind}_{i:02d}"
    out = DIR / f"{name}.json"
    if out.exists():
        return name, "已完成（跳过）"
    src = FIT("HRGP" if kind == "真值" else "H0")
    spec_src = SPEC_RG if kind == "真值" else SPEC_0
    Xs, phis = np.array(src["X"]), np.array([src["shared"][k] for k in spec_src.names()])
    seed = 500 + 37 * i + (0 if kind == "真值" else 10000)
    rng = np.random.default_rng(seed)
    eps = np.random.default_rng(seed + 1).standard_normal(L.T)
    A = np.zeros(L.A_REAL.shape)
    L.run(Xs, spec_src, phis, A, L.G_REAL, L.S_REAL, L.ATT, rng=rng, sigma=src["sigma"], eps=eps)
    # 从头估计：个体起点 = 全 0 加 4 个盆地起点（不用真值），共用起点 = 原项目 λ、θ = 0
    X0 = np.zeros_like(Xs)
    logf = open(L.W / "日志" / f"recovery_{name}.log", "a", encoding="utf-8")
    log = lambda s: (logf.write(s + "\n"), logf.flush())
    part = DIR / f"{name}.partial.json"                    # 每个模型拟合完就存（容器重启后不必重做）
    res = json.loads(part.read_text(encoding="utf-8")) if part.exists() else \
        dict(name=name, kind=kind, seed=seed, true=dict(zip(spec_src.names(), phis.tolist())), true_sigma=src["sigma"])
    for mname, spec in (("M0", SPEC_0), ("RG", SPEC_RG)):
        if mname in res:
            continue
        phi0 = np.r_[L.base_lambda(), np.zeros(spec.k - 1)]
        phi0[-1] = 1.0                                   # 习惯痕迹速率起点 ≈ 0.73
        X, phi, f = staged_fit(spec, A, X0, phi0, seed, L.CKPT / f"rec_{name}_{mname}.npz", log)
        z = L.run(X, spec, phi, A, L.G_REAL, L.S_REAL, L.ATT, out="z")
        sig, lls = L.marginal_sigma(z, A)
        res[mname] = dict(shared=dict(zip(spec.names(), phi.tolist())), nll=float(f.sum()), sigma=sig, loglik_sigma=lls)
        L.save_json(res, part)
        log(f"{name} {mname}: NLL={f.sum():.2f} σ={sig:.3f}")
    res["LR"] = 2 * (res["M0"]["nll"] - res["RG"]["nll"])
    res["LR_sigma"] = 2 * (res["RG"]["loglik_sigma"] - res["M0"]["loglik_sigma"])
    L.save_json(res, out)
    return name, f"LR_σ={res['LR_sigma']:.2f}"


def summarize():
    from scipy.stats import chi2
    rs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(DIR.glob("*.json")) if not p.name.endswith(".partial.json")]
    keys = SPEC_RG.names()[1:]
    S = {"n_真值": sum(r["kind"] == "真值" for r in rs), "n_零": sum(r["kind"] == "零" for r in rs)}
    tru = [r for r in rs if r["kind"] == "真值"]; nul = [r for r in rs if r["kind"] == "零"]
    S["真值套"] = {k: dict(真值=tru[0]["true"][k], 估计均值=float(np.mean([r["RG"]["shared"][k] for r in tru])),
                          估计SD=float(np.std([r["RG"]["shared"][k] for r in tru], ddof=1))) for k in keys} if tru else {}
    S["零套"] = {k: dict(均值=float(np.mean([r["RG"]["shared"][k] for r in nul])), SD=float(np.std([r["RG"]["shared"][k] for r in nul], ddof=1)))
                 for k in keys} if len(nul) > 1 else {}
    if nul:
        lr = np.array([r["LR_sigma"] for r in nul])
        S["零套_LR_sigma"] = dict(均值=float(lr.mean()), 名义自由度=10, 说明="结果字段 M0 = H0（习惯痕迹无调节），RG = HRGP", 假阳性率=float(np.mean(chi2.sf(np.maximum(lr, 0), 10) < .05)), 值=lr.tolist())
    if tru:
        S["真值套_LR_sigma"] = [r["LR_sigma"] for r in tru]
    L.save_json(S, L.OUT / "恢复汇总.json")
    return S


if __name__ == "__main__":
    DIR.mkdir(parents=True, exist_ok=True)
    jobs = [("真值", i) for i in range(4)] + [("零", i) for i in range(8)]
    with Pool(4) as pool:
        for name, msg in pool.imap_unordered(job, jobs):
            print(name, msg, flush=True)
    summarize()
    (L.CKPT / "recovery.done").write_text("ok")
