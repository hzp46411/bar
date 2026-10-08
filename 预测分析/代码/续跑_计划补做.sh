#!/usr/bin/env bash
# 下次启动后继续"按计划补做"中未完成的部分（需先把 断点包 解压回 预测分析/，恢复 数据/ 结果/ 检查点/ 日志/）
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# J2：从 检查点/s14_J2.npz 自动继续
nohup python3 s14_HRGPR预测联合模型.py J2 20 > ../日志/s14_J2.out 2>&1 &
# 恢复检验：没有中途检查点，整套重跑（每套约 1 小时）
for k in 0 1; do
  [ -f ../结果/s14b_联合模型恢复_套$k.json ] || nohup python3 s14b_联合模型恢复.py $k > ../日志/s14b_$k.out 2>&1 &
done
echo "已在后台启动；完成后：python3 图_预测分析.py，并补完 交付/预测分析/按计划的分析报告.md 中的【J2】【s14b】与第 5 节"
