#!/bin/bash
# run_baseline_loop.sh

# 定义 TOP_TOOLS 与 insert_number 的组合数组，格式为 "TOP_TOOLS:insert_number"

test_combinations=(
    "10:0"
)

# 计数器
total_runs=${#test_combinations[@]}
current_run=1
LOG_FILE="./test_yzx/experiment_result.txt"
LOG_FILE2="./test_yzx/rag_gt.txt"
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
    IFS=":" read -r top_tools insert_number <<< "$combo"

    echo ""| tee -a "$LOG_FILE"
    echo "============================================================"| tee -a "$LOG_FILE"
    echo "第 $current_run/$total_runs 次执行"| tee -a "$LOG_FILE"
    echo "参数:TOP_TOOLS=$top_tools, insert_number=$insert_number"| tee -a "$LOG_FILE"

    echo "============================================================"| tee -a "$LOG_FILE2"
    echo "第 $current_run/$total_runs 次执行"| tee -a "$LOG_FILE2"
    echo "参数:TOP_TOOLS=$top_tools, insert_number=$insert_number"| tee -a "$LOG_FILE2"

    # 删除旧文件
    rm -f ./baseline/output/baseline_results.json

    # 运行baseline
    echo "运行baseline..."
    # 清空文件
    cat /dev/null > ./test_yzx/selected_tools.txt

    # 修改 .env 文件中的 TOP_TOOLS 值
    sed -i "s/^TOP_TOOLS=.*/TOP_TOOLS=$top_tools/" .env
    echo "修改 .env 文件: TOP_TOOLS=$top_tools"

    # # 调用 env_reset.sh 更新环境
    # echo "调用 env_reset.sh 更新环境..."
    # bash ./scripts/env_reset.sh

    # 显示当前 .env 中的 TOP_TOOLS 值以确认修改成功
    current_top_tools=$(grep "^TOP_TOOLS=" .env | cut -d'=' -f2)
    echo "当前 .env 中 TOP_TOOLS=$current_top_tools"

    uv run -m baseline.run_conversation \
        --input_path ./annotated_data/all_annotations.json \
        --output_path ./baseline/output/baseline_results.json \
        --insert_number $insert_number \
        --top_tools $top_tools

    if [ $? -eq 0 ]; then
        echo "baseline执行完成" | tee -a "$LOG_FILE"
        
        # 运行judge
        echo "📊 运行judge.py..."
        python3 ./test_yzx/judge.py 2>&1 | tee -a "$LOG_FILE"
        
        if [ $? -eq 0 ]; then
            echo "参数TOP_TOOLS=$top_tools ,insert_number=$insert_number judge完成" | tee -a "$LOG_FILE"
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