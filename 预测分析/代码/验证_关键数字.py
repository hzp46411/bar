# -*- coding: utf-8 -*-
"""快速核对：用保存的参数与相同的随机种子，重算报告中的关键数字（约 3–5 分钟）
  V1 数据事实：选择—预测一致率、独立基线、平均预测、比"永远报 60"更准的人数
  V2 工具变量主规格（s9）：重跑交叉拟合的 κ̂ 与两阶段最小二乘
  V3 合理化联合模型（s10）：用保存的逐人参数重算五个变体的负对数似然
  V4 BBL 信念形成（s12）：重算四种信念方式的负对数似然
  V5 完整联合模型（s14）：重算 J0 / J1 / J2 的总负对数似然与选择部分；关掉类别信念通道后与 HRGPR 逐人一致
  V6 Arthur 机制检验（s16）：用精修后的逐人参数重算 r = 0.1 时 θ = 0 与 θ = 1 的负对数似然
  V7 4 轮项（s13b）：两步估计的 δ4 与似然比
  V8 闭环（s17）：用相同种子重跑"完整"与"去信念"各 300 次
  V9 频率依赖（s18）：用相同种子重跑外推者比例扫描
  V10 恒定平移联合模型（s7）：重算类别 / 分级写法的负对数似然
  某一步的结果文件不存在时跳过该项。
用法：python3 验证_关键数字.py
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit
import pred_lib as PL
HERE = pathlib.Path(__file__).resolve().parent
OUT = PL.OUT
J = lambda f: json.loads((OUT / f).read_text(encoding="utf-8"))
ok_all = []


def check(name, got, want, tol):
    good = bool(np.all(np.abs(np.asarray(got, float) - np.asarray(want, float)) <= tol))
    ok_all.append(good)
    print(f"  {'通过' if good else '不通过'}  {name}：重算 {np.round(got, 4).tolist() if np.ndim(got) else round(float(got), 4)}  保存 {np.round(want, 4).tolist() if np.ndim(want) else round(float(want), 4)}", flush=True)


def load(fname, argv):
    sys.argv = [sys.argv[0]] + argv
    spec = importlib.util.spec_from_file_location(fname.replace(".py", "") + "_".join(argv), str(HERE / fname))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def has(f): return (OUT / f).exists()


P, N, A, OK = PL.P, PL.N, PL.A, PL.OK
print("V1 数据事实")
cons = np.nanmean(np.where(OK, A == (P <= 60), np.nan)); pg = np.nanmean(A); pb = np.nanmean((P <= 60)[OK])
check("选择—预测一致率", cons, 0.7773, 1e-3); check("独立基线", pg * pb + (1 - pg) * (1 - pb), 0.5107, 1e-3)
check("平均预测", np.nanmean(P), 60.029, 1e-2)
mae_i = np.nanmean(np.abs(P - N[None])[:, 1:], 1); check("比'永远报 60'更准的人数", int((mae_i < np.mean(np.abs(60 - N[1:]))).sum()), 0, 0)

if has("s9_选择对预测的因果效应.json"):
    print("V2 工具变量主规格（s9）")
    src = (HERE / "s9_选择对预测的因果效应.py").read_text(encoding="utf-8")
    ns = {"__name__": "s9"}; exec(src.split("out = {")[0], ns)
    ii, tt, a, Xc, nm, K_half, c = ns["ii"], ns["tt"], ns["a"], ns["Xc"], ns["nm"], ns["K_half"], ns["c"]
    yN = P[ii, tt] - 60.0; yK = (P[ii, tt] <= 60).astype(float); Z = (K_half[ii, tt] * c)[:, None]
    g1 = PL.iv_fe(yN, a, Xc, Z, ii, nm)["内生变量"]["b"]; g2 = PL.iv_fe(yK, a, Xc, Z, ii, nm)["内生变量"]["b"]
    d9 = J("s9_选择对预测的因果效应.json")["主：κ̂（前后半交叉拟合）× c"]
    check("γ（人数）", g1, d9["人数"]["γ"]["b"], 1e-6); check("报不挤的效应", g2, d9["不挤"]["γ"]["b"], 1e-6)

if has("s10_合理化联合模型_ρ逐人.json"):
    print("V3 合理化联合模型（s10）")
    for v in ("ρ0", "ρ共用", "ρ逐人", "s0", "s0ρ逐人"):
        if not has(f"s10_合理化联合模型_{v}.json"): continue
        m = load("s10_合理化联合模型.py", [v]); d = J(f"s10_合理化联合模型_{v}.json")
        X = np.array(d["X"]); sh = np.array([d["共用"][k] for k in m.SH])
        check(f"{v} 的 NLL", sum(m.nll_person(X[i], sh, i) for i in range(m.n)), d["总NLL"], 1e-3)

if has("s12_BBL信念与预测_BBL.json"):
    print("V4 BBL 信念形成（s12）")
    for v in ("BBL", "BBL固定ρ", "大小", "BBL+上轮"):
        if not has(f"s12_BBL信念与预测_{v}.json"): continue
        m = load("s12_BBL信念与预测.py", [v]); d = J(f"s12_BBL信念与预测_{v}.json")
        X = np.array(d["X"]); sh = np.array([d["共用"][k] for k in m.SH])
        check(f"{v} 的 NLL", sum(m.nll_person(X[i], sh, i) for i in range(m.n)), d["总NLL"], 1e-3)

if has("s14_HRGPR预测联合模型_J1.json"):
    print("V5 完整联合模型（s14）")
    for v in ("J0", "J1", "J2"):
        if not has(f"s14_HRGPR预测联合模型_{v}.json"): continue
        m = load("s14_HRGPR预测联合模型.py", [v]); d = J(f"s14_HRGPR预测联合模型_{v}.json")
        X = np.array(d["X"]); phi = np.array([d["共用"][k] for k in m.SP.names()])
        DATA = (m.A, m.PP, m.OKP, m.G, m.S)
        nll, nlc = m.run_joint(X, phi, *DATA, out="both")
        check(f"{v} 总 NLL / 选择部分", [nll.sum(), nlc.sum()], [d["总NLL"], d["选择部分的边际NLL"]], 1e-3)
        if v == "J1":
            X0 = m.init_X(m.X4, DATA, rho_f=np.full(m.n, 0.3))
            _, nlc0 = m.run_joint(X0, m.PHI4, *DATA, out="both")
            ref = m.L.run(m.X4, m.SP, m.PHI4, m.A, m.G, m.S, m.N, std=m.STD)
            check("关掉类别信念通道后与 HRGPR 逐人最大差", np.abs(nlc0 - ref).max(), 0.0, 1e-8)

if has("s16_Arthur机制检验_精修_r0.1.json"):
    print("V6 Arthur 机制检验（s16，r = 0.1）")
    src = (HERE / "s16_Arthur机制检验.py").read_text(encoding="utf-8")
    ns = {"__name__": "s16"}; exec(src.split('if __name__ == "__main__":')[0], ns)
    acc, accown = ns["accuracy"](0.1); D = [ns["person_data"](i, acc, accown) for i in range(PL.n)]
    d = J("s16_Arthur机制检验_精修_r0.1.json")["各θ"]
    for th in ("0.0", "1.0"):
        X = [np.array(x) for x in d[th]["X"]]
        check(f"θ = {th} 的 NLL", sum(ns["nll_i"](X[i], *D[i], float(th)) for i in range(PL.n)), d[th]["全体"], 1e-3)

if has("s13b_四轮项与闭环ACF4.json"):
    print("V7 4 轮项（s13b，两步）")
    from scipy.optimize import minimize_scalar
    L = PL.L; z = PL.model_states("HRGPR")["z"]
    lag4 = np.r_[np.zeros(4), (N[:-4] - 60) / 10]
    s0, ll0 = L.marginal_sigma(z, A)
    r = minimize_scalar(lambda dd: -L.marginal_sigma(z + dd * lag4[None], A)[1], bounds=(-1, 1), method="bounded")
    _, ll1 = L.marginal_sigma(z + r.x * lag4[None], A); d = J("s13b_四轮项与闭环ACF4.json")["两步"]
    check("δ4 / LR", [r.x, 2 * (ll1 - ll0)], [d["δ4"], d["LR"]], [1e-3, 1e-2])

if has("s17_宏观秩序的生成.json"):
    print("V8 闭环（s17，相同种子各 300 次）")
    S17 = load("s17_宏观秩序的生成.py", ["300", "150"]); d = J("s17_宏观秩序的生成.json")["A 成分拆分"]
    Xnb = S17.X0.copy(); Xnb[:, 1] = 0.0
    for lab, X in (("完整人群", S17.X0), ("去信念（β = 0）", Xnb)):
        r = S17.run_many(X, S17.PHI0, S17.SIG0, 300, 100000)
        check(f"{lab} 平均人数 / SD / 效率", [r["均值"]["均值"], r["SD"]["均值"], r["效率"]["均值"]], [d[lab]["均值"]["均值"], d[lab]["SD"]["均值"], d[lab]["效率"]["均值"]], 1e-9)

if has("s18_内部模型为什么会产生.json"):
    print("V9 频率依赖（s18，外推者比例扫描，相同种子）")
    S18 = load("s18_内部模型为什么会产生.py", []); S18.B = 120
    r = S18.freq_scan(1, "β"); d = J("s18_内部模型为什么会产生.json")["A 频率依赖：选择中的信念方向（β > 0 外推）"]
    check("得分相等的比例 f*", r["得分相等的比例"], d["得分相等的比例"], 1e-9)

if has("s7_联合模型_类别.json"):
    print("V10 恒定平移联合模型（s7）")
    for v in ("类别", "分级"):
        if not has(f"s7_联合模型_{v}.json"): continue
        m = load("s7_联合模型.py", [v]); d = J(f"s7_联合模型_{v}.json")
        X = np.array(d["X"]); sh = np.array([d["共用"][k] for k in m.SH])
        check(f"{v} 的 NLL", sum(m.nll_person(X[i], sh, i) for i in range(m.n)), d["总NLL"], 1e-2)

print(f"\n共 {len(ok_all)} 项，通过 {sum(ok_all)} 项。" + ("全部通过。" if all(ok_all) else "有项目未通过，请检查。"))
