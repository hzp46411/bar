# 守护进程：按 队列.txt 顺序运行脚本；脚本进程消失且没有 检查点/<脚本名>.done 时从检查点重启。
# 用法：nohup bash 守护.sh > ../日志/守护.log 2>&1 &
D="$(cd "$(dirname "$0")" && pwd)"
W="$(cd "$D/.." && pwd)"
bash "$D/恢复原项目.sh"
while true; do
  all_done=1
  while read -r s; do
    [ -z "$s" ] && continue
    base="${s%.py}"
    if [ -f "$W/检查点/$base.done" ]; then continue; fi
    all_done=0
    if ! ps aux | grep -v grep | grep -q "[p]ython3 $s"; then
      echo "$(date '+%F %T') 启动 $s"
      (cd "$D" && nohup python3 "$s" >> "$W/日志/$base.out" 2>&1 &)
    fi
    break            # 按顺序：前一个没完成就不启动下一个
  done < "$D/队列.txt"
  [ "$all_done" = 1 ] && { echo "$(date '+%F %T') 队列全部完成"; exit 0; }
  sleep 60
done
