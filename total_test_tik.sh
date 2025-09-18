#!/bin/bash
# run_baseline_loop.sh

LOG_FILE="./test_yzx/time_log.txt"
# 清空文件
cat /dev/null > "$LOG_FILE"
echo "开始批量执行baseline实验"
echo "日志保存到: $LOG_FILE"

# 删除旧文件
rm -f ./baseline/output/example_results.json

# 运行baseline
echo "运行baseline..."
uv run -m baseline.run_conversation \
    --input_path ./baseline/data/example_queries.json \
    --output_path ./baseline/output/example_results.json \


if [ $? -eq 0 ]; then
    echo "baseline执行完成" | tee -a "$LOG_FILE"
else
    echo " baseline执行失败" | tee -a "$LOG_FILE"
fi

echo ""
echo "全部实验执行完成！"