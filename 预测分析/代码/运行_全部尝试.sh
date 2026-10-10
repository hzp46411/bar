#!/usr/bin/env bash
# 预测分析 · 全部尝试：从头重跑研究中实际走过的全部路径（含被取代与旁支）（4 核约 3–4 小时；可分段运行，每段都会写出结果 JSON）
# 依赖：本文件夹的 数据/allpredict.csv 与 依赖/宏观调节双系统/（第 4 层的 arb_lib.py、HRGPR / HRGPRS 拟合、原项目数据）
# 从头重跑前请清空 ../检查点/（否则联合模型会从检查点继续，立即结束）
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { echo "[$(date +%H:%M:%S)] $*"; python3 "$@" > "../日志/$(echo "$*" | tr ' /' '__').out" 2>&1; }
# 1 描述、信念与行动、双系统、群体、可靠性、经验（各约 1–3 分钟）
for s in s1_信念形成 s2_信念与行动 s3_不依赖模型的双系统检验 s4_信念系统的直接测量 s5_群体层面 s6_真实预测误差与可靠性仲裁 s8_经验与个体差异; do run "$s.py"; done
# 2 类别式与分级联合模型（类别约 5 分钟；分级约 40 分钟）
run s7_联合模型.py 类别 40 & run s7_联合模型.py 分级 40 & wait
# 3 工具变量（约 2 + 10 分钟）
run s9_选择对预测的因果效应.py; run s9b_γ稳健性与报告分布.py
# 4 合理化联合模型（五个变体并行，约 5 分钟）与恢复（约 30 分钟）
for v in ρ0 ρ共用 ρ逐人 s0 s0ρ逐人; do run s10_合理化联合模型.py "$v" 40 & done; wait
run s10b_合理化模型恢复.py
run s11_合理化对前面结论的影响.py
# 5 BBL 信念形成（四种并行，约 3 分钟）
for v in BBL BBL固定ρ 大小 BBL+上轮; do run s12_BBL信念与预测.py "$v" 30 & done; wait
# 6 ACF4（约 1 分钟）
run s13_四轮前与ACF4.py; run s13b_四轮项与闭环ACF4.py
# 7 完整联合模型（J0、J1 并行约 10 分钟；J2 从 J1 出发）与恢复（每套约 8–15 分钟）
run s14_HRGPR预测联合模型.py J0 20 & run s14_HRGPR预测联合模型.py J1 20 & wait
run s14_HRGPR预测联合模型.py J2 20
run s14b_联合模型恢复.py 0 & run s14b_联合模型恢复.py 1 & run s14c_两种学习率下的恢复.py 0 & run s14c_两种学习率下的恢复.py 1 & wait
# 8 Arthur 机制检验：三个记忆速率的网格并行（约 1 小时）→ 精修（约 1 小时）→ 合并
for r in 0.1 0.3 0.6; do run s16_Arthur机制检验.py 网格 $r & done; wait
for r in 0.1 0.3 0.6; do run s16_Arthur机制检验.py 精修 $r & done; wait
run s16_Arthur机制检验.py 合并
# 9 普查、闭环、频率依赖（约 6 分钟）
run s15_内部模型普查.py; run s17_宏观秩序的生成.py 300 150; run s17b_双重分离.py; run s18_内部模型为什么会产生.py
# 10 旁支与探索（被取代的"信念份额外推"；研究中在命令行里临时算过的核对）
run s7c_信念份额外推.py; run 探索_杂项.py
# 11 图与核对
run 图_预测分析.py
python3 验证_关键数字.py
