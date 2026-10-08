# -*- coding: utf-8 -*-
"""第 4 层 · 步骤 1（补 3）：加法"重复推力"的联合估计（HRGP、HRGP−stab、HRGP−dev），完成后精修。"""
import sys
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import fit_models as FM
import polish as P

if __name__ == "__main__":
    names = ["HRGP", "HRGP-stab", "HRGP-dev"]
    with Pool(3) as pool:
        for name, msg in pool.imap_unordered(FM.job, names):
            print(name, msg, flush=True)
    with Pool(3) as pool:
        for name, msg in pool.imap_unordered(P.job, [L.OUT / "拟合" / f"{n}.json" for n in names]):
            print("精修", name, msg, flush=True)
    (L.CKPT / "fit_models4.done").write_text("ok")
