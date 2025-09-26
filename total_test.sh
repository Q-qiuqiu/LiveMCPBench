#!/bin/bash
# run_baseline_loop.sh

# 定义 TOP_TOOLS 与 insert_number 的组合数组，格式为 "TOP_TOOLS:insert_number"

test_combinations=(
    "20:5"
    "20:10"
    "20:15"
    "20:20"
    "30:0"
    "30:10"
    "30:20"
    "30:30"
    "40:0"
    "40:10"
    "40:20"
    "40:30"
    "40:40"
    "50:0"
    "50:10"
    "50:20"
    "50:30"
    "50:40"
    "50:50"
    "100:0"
    "100:10"
    "100:20"
    "100:30"
    "100:40"
    "100:50"
    "100:100"
    "150:0"
    "150:20"
    "150:40"
    "150:50"
    "150:100"
    "150:150"
    "200:0"
    "200:20"
    "200:40"
    "200:50"
    "200:100"
    "200:150"
    "200:200"
    "250:0"
    "250:20"
    "250:40"
    "250:50"
    "250:100"
    "250:150"
    "250:200"
    "250:250"
    "300:0"
    "300:20"
    "300:40"
    "300:50"
    "300:100"
    "300:150"
    "300:200"
    "300:250"
    "300:300"
    "350:0"
    "350:20"
    "350:40"
    "350:50"
    "350:100"
    "350:150"
    "350:200"
    "350:250"
    "350:300"
    "350:350"
    "400:0"
    "400:20"
    "400:40"
    "400:50"
    "400:100"
    "400:150"
    "400:200"
    "400:250"
    "400:300"
    "400:350"
    "400:400"
    "450:0"
    "450:20"
    "450:40"
    "450:50"
    "450:100"
    "450:150"
    "450:200"
    "450:250"
    "450:300"
    "450:350"
    "450:400"
    "450:450"
    "500:0"
    "500:20"
    "500:40"
    "500:50"
    "500:100"
    "500:150"
    "500:200"
    "500:250"
    "500:300"
    "500:350"
    "500:400"
    "500:450"
    "500:500"
)

# 计数器
total_runs=${#test_combinations[@]}
current_run=1
LOG_FILE="./test_yzx/experiment_result.txt"
# 清空文件
cat /dev/null > "$LOG_FILE"
echo "开始批量执行baseline实验" | tee -a "$LOG_FILE"
echo "总共 $total_runs 个参数" | tee -a "$LOG_FILE"
echo "日志保存到: $LOG_FILE"

# 检查 .env 文件是否存在
if [ ! -f .env ]; then
    echo "错误: .env 文件不存在，请先创建 .env 文件"
    exit 1
fi

for combo in "${test_combinations[@]}"; do
    #拆分参数
    IFS=":" read -r max_tools insert_number <<< "$combo"

    echo ""| tee -a "$LOG_FILE"
    echo "============================================================"| tee -a "$LOG_FILE"
    echo "第 $current_run/$total_runs 次执行"| tee -a "$LOG_FILE"
    echo "参数:MAX_TOOLS=$max_tools, insert_number=$insert_number"| tee -a "$LOG_FILE"

    # 删除旧文件
    rm -f ./baseline/output/baseline_results.json

    # 运行baseline
    echo "运行baseline..."
    # 清空文件
    cat /dev/null > ./test_yzx/selected_tools.txt

    uv run -m baseline.run_conversation \
        --input_path ./annotated_data/all_annotations.json \
        --output_path ./baseline/output/baseline_results.json \
        --insert_number $insert_number \
        --max_tools $max_tools

    if [ $? -eq 0 ]; then
        echo "baseline执行完成" | tee -a "$LOG_FILE"
        
        # 运行judge
        echo "📊 运行judge.py..."
        python3 ./test_yzx/judge.py 2>&1 | tee -a "$LOG_FILE"
        
        if [ $? -eq 0 ]; then
            echo "参数MAX_TOOLS=$max_tools ,insert_number=$insert_number judge完成" | tee -a "$LOG_FILE"
        else
            echo "judge.py执行失败" | tee -a "$LOG_FILE"
        fi
    else
        echo " baseline执行失败" | tee -a "$LOG_FILE"
    fi
    current_run=$((current_run + 1))
    sleep 2  # 可选延迟
done


echo ""
echo "全部实验执行完成！"