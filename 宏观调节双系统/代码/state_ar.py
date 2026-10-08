# -*- coding: utf-8 -*-
"""宏观层面的状态依赖反馈：N_t − 60 = a + φ0·d + φs·d·稳定 + φn·d·|d|/10，d = N_{t−1} − 60（稳定 = 决策前同一状态已持续轮数，标准化）
真实数据 vs 闭环模拟（完整 HRGPR / 关宏观调节 / 三条都关，各 300 次）。"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import abm_arb as Ab

STD = L.load_std()
def coefs(N):
    pm, _ = L.public_mods(N)
    d = np.r_[np.nan, N[:-1] - L.CAP]; y = N - L.CAP
    st = (pm["stab"] - STD["stab"][0]) / STD["stab"][1]
    sl = slice(5, None)
    Xm = np.column_stack([np.ones(L.T), d, d * st, d * np.abs(d) / 10])[sl]
    b, *_ = np.linalg.lstsq(Xm, y[sl], rcond=None)
    return b[1:]
if __name__ == "__main__":
    Ab.fingerprint = lambda N, A: {"N": N.copy()}
    out = {"真实": dict(zip(["φ0", "φs_稳定", "φn_非线性"], np.round(coefs(L.ATT), 4).tolist()))}
    for name in ("HRGPR", "HRGPR_关宏观调节", "HRGPR_三条都关", "M0"):
        C = np.array([coefs(Ab.abm_one((name, 900000 + k))[1]["N"]) for k in range(300)])
        real = coefs(L.ATT)
        out[name] = {lab: dict(均值=round(float(C[:, j].mean()), 4), 区间95=[round(float(np.percentile(C[:, j], q)), 4) for q in (2.5, 97.5)],
                               真实值的分位=round(float(np.mean(C[:, j] <= real[j])), 3)) for j, lab in enumerate(["φ0", "φs_稳定", "φn_非线性"])}
        print(name, out[name], flush=True)
    print("真实", out["真实"])
    L.save_json(out, L.OUT / "宏观状态依赖反馈.json")
