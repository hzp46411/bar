# -*- coding: utf-8 -*-
"""
第 37 步：对称的联合检验——单习惯、单 MF、单世界模型与主模型，各自"不加 / 加共同因素"
  与原项目第 3 阶段相同：每个个体模型都在自己的开环 logit 上用两步法重新估计共同因素（γ、σ），
  再做闭环 ABM（各 1000 次），用事先固定的宏观指纹（均值、SD、ACF1–3）做联合检验（马氏距离² + Hotelling 预测检验），
  另报告合成似然（Wood, 2010）、个体指标 D 与换选择比例、锁定、ACF4（描述）。
  个体模型：H（只有习惯 choice kernel）、F（只有 MF，Q 学习）、B（只有世界模型）、HB（主模型），均为 s22 的层级估计
  问题：单 MF 能否产生真实的宏观模式？共同因素本身能做多少？
用法：python3 s37_对称联合检验.py
输出：结果/s37_对称联合检验.json
"""
import json, importlib.util, pathlib
import numpy as np
from scipy.special import logit
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s36", str(pathlib.Path(__file__).with_name("s36_共同因素层.py")))
S36 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S36)
S27 = S36.S27
MODELS = ["H", "F", "B", "HB"]
B_SIM = 1000

if __name__ == "__main__":
    for m in MODELS:
        S27.SRC[m] = f"s22_真实_{m}.json"; S27.FIXED[m] = (-50.0, 0.0)
    OBS = S36.fingerprint(S27.N_REAL[None])[0]
    OBS_D, OBS_SW = S36.micro(S27.N_REAL[None], S36.A[None].astype(np.int8))
    out = {"真实": dict(指纹=dict(zip(S36.FP_NAMES, np.round(OBS, 4).tolist())), D=round(float(OBS_D[0]), 4), 换选择比例=round(float(OBS_SW[0]), 4)),
           "共同因素": {}, "联合检验": {}}
    SIM = {}
    for m in MODELS:
        X = S27.load(m)
        _, P = S27.open_loop(X)
        d = logit(np.clip(P, 1e-9, 1 - 1e-9))
        E = S36.estimate(d, S36.A)
        g, s = E["γ+σ"][0], E["γ+σ"][1]
        out["共同因素"][m] = dict(γ=round(g, 4), σ=round(s, 4), 个体模型的选择对数似然=round(E["无"][2], 2),
                              加共同因素后=round(E["γ+σ"][2], 2), 似然比检验=S36.lr_tests(E))
        for tag, (Y, sc) in {"": (X, 0.0), " + γ + σ": (X.copy(), s)}.items():
            if tag:
                Y[:, 13] = -g
            fps, Ds, sws, css, acf4 = [], [], [], [], []
            for b in range(4):
                Ns, As = S27.simulate(Y, B_SIM // 4, 100 + 10 * b, sc)
                fps.append(S36.fingerprint(Ns)); d_, sw_ = S36.micro(Ns, As); Ds.append(d_); sws.append(sw_)
                css.append((Ns >= 61).mean(1)); acf4.append(S27.acf(Ns, 4))
            SIM[m + tag] = (np.concatenate(fps), np.concatenate(Ds), np.concatenate(sws), np.concatenate(css), np.concatenate(acf4))
            out["联合检验"][m + tag] = S36.joint(OBS, OBS_D[0], OBS_SW[0], *SIM[m + tag])
            print(m + tag, out["联合检验"][m + tag]["联合p"], out["联合检验"][m + tag]["合成对数似然"], flush=True)
    out["合成似然之差（HB + γ + σ 减其他；bootstrap 95% 区间）"] = {k: S36.synth_diff(OBS, SIM["HB + γ + σ"][0], SIM[k][0]) for k in SIM if k != "HB + γ + σ"}
    PL.save(out, "s37_对称联合检验.json")
    print(json.dumps(out, ensure_ascii=False, indent=1))
