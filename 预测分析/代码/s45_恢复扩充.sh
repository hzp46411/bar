#!/usr/bin/env bash
# 第 45 步（运行部分）：恢复扩充——用 s22 的拟合程序在新的假数据集上拟合；已有结果的模型跳过（复用已有拟合）
# 用法：bash s45_恢复扩充.sh <生成模型_编号> <候选模型…>      例：bash s45_恢复扩充.sh HB_7 0 H F B HF HB FB HFB
ds=$1; shift
for m in "$@"; do
  [ -f "../结果/s22_${ds}_${m}.json" ] && continue
  python3 s22_习惯痕迹审计.py 拟合 "$ds" "$m" || exit 1
done
