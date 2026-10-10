# -*- coding: utf-8 -*-
"""
第 34 步：修正"去快习惯 H"的模型（s28 的 DWLG1）的参数化
  s28 中 DWLG1 去掉 w_I 但保留 α_H 与 a_D，而慢漂移的学习率是 α_D = α_H · expit(a_D)，两者只以乘积出现，
  多出一个不可识别的方向，使拉普拉斯证据多罚一个参数。这里固定 a_D = +50（expit = 1），让 α_H 直接作为慢漂移的学习率。
  做法：真实拟合（起点为 s28 中 DWLG1 的解）与前后半程交叉验证（与 s28 相同）
用法：python3 s34_修正去快习惯模型.py 真实 | 交叉验证 <前半|后半>
输出：结果/s34_真实_DWLG1c.json、结果/s34_交叉验证_DWLG1c_<折>.json
"""
import sys, json, importlib.util, pathlib
import numpy as np
from scipy.special import expit
import pred_lib as PL

spec = importlib.util.spec_from_file_location("s28", str(pathlib.Path(__file__).with_name("s28_最终模型验证.py")))
S28 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S28)
S22, S21, S20 = S28.S22, S28.S21, S28.S20
M = "DWLG1c"
S28.FREE[M] = [0, 5, 6, 7, 8, 9, 2, 3, 4, 14, 13]
_to_X = S28.to_X


def to_X(U, m):
    X = _to_X(U, m)
    if m == M:
        X[:, 10] = 50.0
    return X


S21.to_X, S22.to_X = to_X, to_X
n, T, OKP = S28.n, S28.T, S28.OKP


def start_from_s28():
    r = json.loads((PL.OUT / "s28_真实_DWLG1.json").read_text(encoding="utf-8"))
    X = np.array(r["X"])
    X[:, 8] = np.log(np.clip(expit(X[:, 8]) * expit(X[:, 10]), 1e-6, 1 - 1e-6) / (1 - np.clip(expit(X[:, 8]) * expit(X[:, 10]), 1e-6, 1 - 1e-6)))
    X[:, 10] = 50.0
    U = S28.to_U(X, M)
    return U, U.mean(0), np.maximum(U.var(0), 1e-2)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "真实":
        U, mu, s2 = start_from_s28()
        X0 = to_X(U, M); print("起点 NLL", round(S28.run(X0, S20.A, S20.P, OKP).sum(), 1), flush=True)
        R = S28.fit(M, S20.A, S20.P, U, mu, s2, f"真实_{M}")
        PL.save(dict(模型=M, iBIC=float(-2 * R["LE"].sum() + 2 * len(S28.FREE[M]) * np.log(S28.Ntot)), **R), f"s34_真实_{M}.json")
        print("iBIC", R["LE"].sum(), flush=True)
    else:
        fold = sys.argv[2]
        train = (np.arange(T) < T // 2) if fold == "前半" else (np.arange(T) >= T // 2)
        r = json.loads((PL.OUT / f"s34_真实_{M}.json").read_text(encoding="utf-8"))
        mu, s2 = np.array(r["mu"]), np.array(r["s2"])
        S28.CW[:] = train; S21.OKP = OKP * train
        R = S28.fit(M, S20.A, S20.P, np.tile(mu, (n, 1)), mu, s2, f"交叉验证_{M}_{fold}")
        X = R["X"]
        S28.CW[:] = ~train; choice = S28.run(X, S20.A, S20.P, np.zeros_like(OKP))
        S28.CW[:] = 0.0; fc = S28.run(X, S20.A, S20.P, OKP * ~train)
        PL.save(dict(模型=M, 训练=fold, 检验选择NLL=choice, 检验预测NLL=fc, X=X), f"s34_交叉验证_{M}_{fold}.json")
