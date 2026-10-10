#!/usr/bin/env bash
# s16：三个 r 的网格跑完后，依次精修并合并（检查点：结果/s16_Arthur机制检验_r*.json）
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
until [ "$(grep -l 完成 ../日志/s16_r0.1.out ../日志/s16_r0.3.out ../日志/s16_r0.6.out 2>/dev/null | wc -l)" -eq 3 ]; do sleep 30; done
for r in 0.1 0.3 0.6; do python3 s16_Arthur机制检验.py 精修 $r > ../日志/s16_精修_r$r.out 2>&1 & done; wait
python3 s16_Arthur机制检验.py 合并 > ../日志/s16_合并.out 2>&1
echo 全部完成 >> ../日志/s16_合并.out
