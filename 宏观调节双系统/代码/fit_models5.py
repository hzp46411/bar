# -*- coding: utf-8 -*-
"""第 4 层 · 可靠性仲裁的正式联合检验：HRGP−rel（不含可靠性）、HRGPR（信念权重看信念建议成绩、惯性权重看习惯建议成绩）、
HRGPR4（两种成绩都作用于两个系统，检验交叉作用）。拟合 → 精修 → 补充优化，各阶段有检查点。"""
import sys
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parent))
import arb_lib as L
import fit_models as FM
import polish as P
import refine as RF

NAMES = ["HRGP-rel", "HRGPR", "HRGPR4"]
if __name__ == "__main__":
    with Pool(3) as pool:
        for name, msg in pool.imap_unordered(FM.job, NAMES):
            print(name, msg, flush=True)
    paths = [L.OUT / "拟合" / f"{n}.json" for n in NAMES]
    with Pool(3) as pool:
        for name, msg in pool.imap_unordered(P.job, paths):
            print("精修", name, msg, flush=True)
    with Pool(3) as pool:
        for name, msg in pool.imap_unordered(RF.job, paths):
            print("补充优化", name, msg, flush=True)
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).with_name("compare.py"))], check=True, stdout=subprocess.DEVNULL)
    (L.CKPT / "fit_models5.done").write_text("ok")
