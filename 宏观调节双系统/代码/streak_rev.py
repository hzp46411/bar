# -*- coding: utf-8 -*-
"""微观检验：连续同一状态后，个人是否预期反转（"该轮到另一边了"）？
在 HRGPR 上（两步）加一项 ψrev·稳定·sign(N_{t−1} − 60)：ψrev < 0 = 挤得越久越不去、空得越久越去（与习惯推力方向无关）。"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust as R
import arb_lib as L

rest, bp, hp = R.parts([])
s = np.broadcast_to(np.sign(np.r_[0.0, L.ATT[:-1] - L.CAP]), R.A.shape)
base = rest + bp + hp
_, ll0 = L.marginal_sigma(base, R.A)
x, ll1 = R.fit(lambda p: base + p[0] * R.Mst * s, 1)
x2, ll2 = R.fit(lambda p: base + p[0] * R.Mst * s + p[1] * R.Mst * R.c, 2)      # 同时再给习惯推力一次机会
out = dict(ψrev=round(float(x[0]), 4), LR_df1=round(2 * (ll1 - ll0), 2), p=float(chi2.sf(max(2 * (ll1 - ll0), 0), 1)),
           同时重估_ψrev=round(float(x2[0]), 4), 同时重估_习惯推力增量=round(float(x2[1]), 4))
print(out); L.save_json(out, L.OUT / "连续后反转预期.json")
