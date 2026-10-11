#!/usr/bin/env bash
# 守护进程：按 日志/队列.txt 运行任务，并发上限 P（默认 3）；任务结束且成功时写 检查点/<标记>.done。
# 进程消失且没有 .done 的任务会在下一轮检查时重启（脚本自己负责从检查点续跑）；同一任务最多启动 3 次，之后记为失败、不再重启。
# 每轮把 结果/、日志/、进度.md 同步到持久工作区。全部完成后退出。
# 队列每行：<标记> <命令…>（命令在 代码/ 下执行）。用法：nohup bash 守护.sh > ../日志/守护.log 2>&1 &
W=$(cd "$(dirname "$0")/.." && pwd)
Q="$W/日志/队列.txt"; CK="$W/检查点"; OUT=/mnt/user-data/outputs/第二轮建模_工作区
P=${P:-3}
mkdir -p "$CK" "$OUT"
cd "$W/代码"
while true; do
  pending=0; running=0
  while read -r tag cmd; do
    [ -z "$tag" ] && continue
    [ -f "$CK/$tag.done" ] && continue
    [ "$(cat "$CK/$tag.tries" 2>/dev/null || echo 0)" -ge 3 ] && ! { [ -f "$CK/$tag.pid" ] && kill -0 "$(cat "$CK/$tag.pid")" 2>/dev/null; } && continue
    pending=$((pending + 1))
    if [ -f "$CK/$tag.pid" ] && kill -0 "$(cat "$CK/$tag.pid")" 2>/dev/null; then running=$((running + 1)); fi
  done < "$Q"
  while read -r tag cmd; do
    [ -z "$tag" ] && continue
    [ -f "$CK/$tag.done" ] && continue
    if [ -f "$CK/$tag.pid" ] && kill -0 "$(cat "$CK/$tag.pid")" 2>/dev/null; then continue; fi
    tries=$(cat "$CK/$tag.tries" 2>/dev/null || echo 0)
    [ "$tries" -ge 3 ] && { echo "$(date -u +%H:%M) 失败（已试 3 次）$tag"; continue; }
    [ "$running" -ge "$P" ] && break
    echo $((tries + 1)) > "$CK/$tag.tries"
    nohup bash -c "$cmd && touch '$CK/$tag.done'" < /dev/null >> "$W/日志/守护_$tag.log" 2>&1 &
    echo $! > "$CK/$tag.pid"; running=$((running + 1))
    echo "$(date -u +%H:%M) 启动 $tag"
  done < "$Q"
  cp -ru "$W/结果" "$W/日志" "$OUT/" 2>/dev/null; cp -f "$W/进度.md" "$OUT/" 2>/dev/null
  [ "$pending" -eq 0 ] && { echo "$(date -u +%H:%M) 队列全部完成"; break; }
  sleep 60
done
