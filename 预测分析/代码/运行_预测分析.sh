#!/usr/bin/env bash
# 预测分析：按顺序运行全部步骤（需要第 4 层的 宏观调节双系统/代码/arb_lib.py 与 HRGPR 拟合结果；数据放在 预测分析/数据/allpredict.csv）
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
for s in s1_信念形成 s2_信念与行动 s3_不依赖模型的双系统检验 s4_信念系统的直接测量 s5_群体层面 s6_真实预测误差与可靠性仲裁 s8_经验与个体差异; do
  python3 "$s.py" > /dev/null; echo "完成 $s"
done
python3 s7_联合模型.py 类别 40 > /dev/null; echo "完成 s7 类别"        # 可选：恒定平移的联合模型（分级写法较慢：python3 s7_联合模型.py 分级 40）
python3 s9_选择对预测的因果效应.py > /dev/null; echo "完成 s9"
python3 s9b_γ稳健性与报告分布.py > /dev/null; echo "完成 s9b"
for v in ρ0 ρ共用 ρ逐人 s0 s0ρ逐人; do
  python3 s10_合理化联合模型.py "$v" 40 > /dev/null; echo "完成 s10 $v"
done
python3 s10b_合理化模型恢复.py > /dev/null; echo "完成 s10b"
python3 s11_合理化对前面结论的影响.py > /dev/null; echo "完成 s11"
python3 图_预测分析.py > /dev/null; echo "完成 图"
