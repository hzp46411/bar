# 若环境重置后原项目丢失：从上传的压缩包重新解压（压缩包也可能不在，则需重新上传）
W="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -f "$W/原项目/BBL惯性建模/共用代码/model.py" ]; then
  mkdir -p "$W/原项目"
  Z="$W/原项目/BBL惯性建模.zip"
  [ -f "$Z" ] || cp /root/.claude/uploads/*/*BBL*.zip "$Z"
  (cd "$W/原项目" && unzip -q -o "$Z" -x '*__pycache__*')
fi
