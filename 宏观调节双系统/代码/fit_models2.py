# -*- coding: utf-8 -*-
"""第 4 层 · 步骤 1（补）：习惯痕迹主模型 HRG 的逐项检验 —— HRG 去掉某一个变量对比例的作用（4 个模型）。"""
import sys
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import fit_models as FM

if __name__ == "__main__":
    with Pool(4) as pool:
        for name, msg in pool.imap_unordered(FM.job, [f"HRG-{m}" for m in L.MODS]):
            print(name, msg, flush=True)
    (L.CKPT / "fit_models2.done").write_text("ok")
