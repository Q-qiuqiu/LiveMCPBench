#!/bin/bash
# run_baseline_loop.sh

# 定义参数数组
max_tools_list=(10 200)
insert_numbers=(0)

# 计数器
total_runs=$(( ${#max_tools_list[@]} * ${#insert_numbers[@]} ))
current_run=1
LOG_FILE="./test_yzx/time_log.txt"
# 清空文件
cat /dev/null > "$LOG_FILE"
echo "开始批量执行baseline实验" |
echo "日志保存到: $LOG_FILE"
for max_tools in "${max_tools_list[@]}"; do
    for insert_number in "${insert_numbers[@]}"; do
        echo "参数: max_tools=$max_tools, insert_number=$insert_number"
        # 删除旧文件
        rm -f ./baseline/output/example_results.json

        # 运行example
        echo "运行example..."

        uv run -m baseline.run_conversation \
            --input_path ./baseline/data/example_queries.json \
            --output_path ./baseline/output/example_results.json \
            --max_tools $max_tools \
            --insert_number $insert_number
        
        if [ $? -eq 0 ]; then
            echo "example执行完成" | tee -a "$LOG_FILE"

            # 运行draw.py
            echo "运行draw.py..."
            python3 ./test_yzx/draw.py

            if [ $? -eq 0 ]; then
                echo "参数max_tools=$max_tools, insert_number=$insert_number draw完成"
            else
                echo "draw.py执行失败" | tee -a "$LOG_FILE"
            fi
        else
            echo "example执行失败" | tee -a "$LOG_FILE"
        fi
        
        ((current_run++))
        sleep 2  # 可选延迟
    done
done

echo ""
echo "全部实验执行完成！"