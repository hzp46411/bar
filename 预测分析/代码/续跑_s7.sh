#!/bin/bash
# 两个联合模型各自跑完前 12 轮后，从检查点继续到 40 轮（收敛判据：一轮改善 < 0.5）
cd "$(dirname "$0")"
for m in 分级 类别; do
  ( until [ -f ../检查点/s7_$m.done ]; do sleep 20; done
    rm -f ../检查点/s7_$m.done
    python3 s7_联合模型.py $m 40 >> ../日志/s7_$m.out 2>&1 ) &
done
wait
