#!/bin/bash
# 本地跑 qlib 模式生成信号，自动 push 到 GitHub，Streamlit Cloud 实时更新
# 用法：bash dashboard/push_signals.sh
# 建议加入 crontab：0 0 * * 1-5 bash /Users/stevenwick/qlib-main/dashboard/push_signals.sh

set -e
DASH_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DASH_DIR/results/run_$(date +%Y%m%d).log"

echo "=== $(date '+%Y-%m-%d %H:%M') 开始生成信号 ===" | tee "$LOG"

# 1. 生成信号（qlib 模式，利用 Alpha158）
cd "$DASH_DIR"
python run_daily.py --mode qlib 2>&1 | tee -a "$LOG"

# 2. push results/ 到 GitHub
cd "$DASH_DIR"
git add results/
git diff --staged --quiet && echo "无变更，跳过 push" && exit 0

git commit -m "📊 qlib信号更新 $(TZ='Asia/Shanghai' date +'%Y-%m-%d %H:%M CST')"
git push

echo "=== 推送完成，Streamlit Cloud 将在 1 分钟内更新 ===" | tee -a "$LOG"
