# -*- coding: utf-8 -*-
"""
稳定调节的形状放进闭环：用 翻转检验_<模型>.json 的"分类"（稳定 0–4 各一级）与"两者"（翻转 + 线性）估计，
替换模型里线性的稳定作用，看闭环能否同时再现：换人率按稳定的剖面、换人率波动、长期稳定后的回调 φs。
输出：结果/第三环_稳定形状_<模型>.json
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import abm_arb as Ab
import third_arrow as TA

MODEL = sys.argv[1] if len(sys.argv) > 1 else "HRGPRS"; B = int(sys.argv[2]) if len(sys.argv) > 2 else 400
F = json.loads((L.OUT / f"翻转检验_{MODEL}.json").read_text(encoding="utf-8"))
st = L.public_mods(L.ATT)[0]["stab"]; STD = L.load_std()


def terms(lab):
    x = np.array(F[lab]["参数"])
    if lab == "分类":
        M = np.array([[(k == j) - (st == j).mean() for j in (1, 2, 3, 4)] for k in range(5)])
    elif lab == "两者":
        M = np.array([[(k - STD["stab"][0]) / STD["stab"][1], (k == 0) - (st == 0).mean()] for k in range(5)])
    elif lab == "翻转":
        M = np.array([[(k == 0) - (st == 0).mean()] for k in range(5)])
    else:
        M = np.array([[(k - STD["stab"][0]) / STD["stab"][1]] for k in range(5)])
    m = M.shape[1]
    return dict(eB=M @ x[:m], eH=M @ x[m:2 * m], psi=M @ x[2 * m:3 * m])


def swprof(N, A):
    s = L.public_mods(N)[0]["stab"][1:]; sw = (A[:, 1:] != A[:, :-1]).mean(0)
    return [float(sw[s == k].mean()) if (s == k).any() else np.nan for k in range(5)]


def one(args):
    lab, seed = args
    mod = {} if lab == "原模型(线性)" else {"zero_mods": ["stab"], "stab_terms": terms(lab)}
    N, A, _ = Ab.simulate(MODEL, mod, seed)
    s = TA.structure(N, A); s.update({f"换人率_稳定{k}": v for k, v in enumerate(swprof(N, A))})
    return lab, s


if __name__ == "__main__":
    labs = ["原模型(线性)", "线性", "翻转", "两者", "分类"]
    with Pool(4) as pool:
        res = pool.map(one, [(lab, 910000 + 7919 * i + 101 * j) for j, lab in enumerate(labs) for i in range(B)], chunksize=20)
    obs = TA.structure(L.ATT, L.A_REAL); obs.update({f"换人率_稳定{k}": v for k, v in enumerate(swprof(L.ATT, L.A_REAL))})
    out = {"模型": MODEL, "B": B, "真实": obs, "变体": {}, "两步估计": {lab: F[lab] for lab in ("线性", "翻转", "两者", "分类")}}
    for lab in labs:
        rr = [r for l, r in res if l == lab]
        s, _ = TA.summarize(rr, obs); out["变体"][lab] = s
    L.save_json(out, L.OUT / f"第三环_稳定形状_{MODEL}.json")
    keys = ["φ0", "φs", "φn", "sq_acf1", "换人率_sd", "换人率_acf1", "换人→下轮偏离"] + [f"换人率_稳定{k}" for k in range(5)]
    print("真实", {k: round(obs[k], 3) for k in keys})
    for lab, s in out["变体"].items():
        print(lab, {k: (round(s[k]["均值"], 3), round(s[k]["真实分位"], 2)) for k in keys})
