#!/usr/bin/env bash
# 按顺序运行全部分析；已完成的步骤（检查点/*.done）自动跳过，中断后重跑即可断点续跑。
# 用法：bash run_all.sh            （在 代码/ 目录下运行）
#       FORCE=1 bash run_all.sh    （忽略 .done 标记，全部重跑）
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CKPT=../检查点
LOG=../日志
mkdir -p "$CKPT" "$LOG"

for f in ../数据/allpredict.csv ../数据/allpredict_rt.csv; do
  [ -f "$f" ] || { echo "缺少数据文件 $f（把 allpredict.csv 与 allpredict_rt.csv 放进 数据/）"; exit 1; }
done

for step in 01_describe 02_fit 03_compare 04_recovery_sim 05_recovery_analysis 07_supplement; do
  if [ -z "${FORCE:-}" ] && [ -f "$CKPT/$step.done" ]; then
    echo "跳过 $step（已完成）"
    continue
  fi
  echo "运行 $step ..."
  python3 "$step.py" 2>&1 | tee "$LOG/$step.log"
done
python3 06_figures.py 2>&1 | { grep -v findfont || true; } | tee "$LOG/06_figures.log"
echo "全部完成：结果在 ../结果/，图在 ../图/"
