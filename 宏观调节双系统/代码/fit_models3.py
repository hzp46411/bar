# -*- coding: utf-8 -*-
"""第 4 层 · 步骤 1（补 2）：偏离对信念 / 惯性两个系统分开的联合检验，完成后对新模型做全参数精修。"""
import sys
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import fit_models as FM
import polish as P

if __name__ == "__main__":
    names = ["HRG_dev仅惯性", "HRG_dev仅信念"]
    with Pool(2) as pool:
        for name, msg in pool.imap_unordered(FM.job, names):
            print(name, msg, flush=True)
    with Pool(2) as pool:
        for name, msg in pool.imap_unordered(P.job, [L.OUT / "拟合" / f"{n}.json" for n in names]):
            print("精修", name, msg, flush=True)
    (L.CKPT / "fit_models3.done").write_text("ok")
