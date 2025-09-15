#!/bin/bash
# run_baseline_loop.sh

# 定义 TOP_TOOLS 与 insert_number 的组合数组，格式为 "TOP_TOOLS:insert_number"

    # "45:40"
    # "50:40"  
    # "55:40"
    # "70:40"
    # "75:40"
    # "80:40"
    # "85:40"
    # "90:40"
    # "95:40"
    # "100:40"
    # "150:40"
    # "200:40"
    # "250:40"
    # "300:40"
    # "350:40"
    # "400:40"
    # "450:40"
    # "500:40"
    # "45:44"
    # "50:45"
    # "55:45"
    # "60:45"
    # "65:45"
    # "80:45"
    # "85:45"
    # "90:45"
    # "95:45"
    # "100:45"
    # "150:45"
    # "200:45"
    # "250:45"
    # "300:45"
    # "350:45"
    # "400:45"
    # "450:45"
    # "500:45"
    # "50:49"
    # "55:50"
    # "60:50"
    # "65:50"
    # "80:50"
    # "85:50"
    # "90:50"
    # "95:50"
    # "100:50"
    # "150:50"
    # "200:50"
    # "250:50"
    # "300:50"
    # "350:50"
    # "400:50"
    # "450:50"
    # "500:50"
    # "55:54"
    # "60:55"
    # "65:55"
    # "80:55"
    # "85:55"
    # "90:55"
    # "95:55"
    # "100:55"
    # "150:55"
    # "200:55"
    # "250:55"
    # "300:55"
    # "350:55"
    # "400:55"
    # "450:55"
    # "500:55"
    # "100:95"
    # "150:100"
    # "200:100"
    # "250:100"
    # "300:100"
    # "350:100"
    # "400:100"
    # "450:100"
    # "500:100"
    # "150:145"
    # "200:150"
    # "250:150"
    # "300:150"
    # "350:150"
    # "400:150"
    # "450:150"
    # "500:150"
    # "200:195"
    # "250:200"
    # "300:200"
    # "350:200"
    # "400:200"
    # "450:200"
    # "500:200"
    # "250:245"
    # "300:250"
    # "350:250"
    # "400:250"
    # "450:250"
    # "500:250"
    # "300:295"
    # "350:300"
    # "400:300"
    # "450:300"
    # "500:300"
    # "350:345"
    # "400:350"
    # "450:350"
    # "500:350"
    # "400:395"
    # "450:400"
    # "500:400"
    # "450:445"
    # "500:450"
    # "500:495"

test_combinations=(
    "200:0"
    "200:50"
    "200:100"
    "200:150"
    "200:195"
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
    IFS=":" read -r top_tools insert_number <<< "$combo"

    echo ""| tee -a "$LOG_FILE"
    echo "============================================================"| tee -a "$LOG_FILE"
    echo "第 $current_run/$total_runs 次执行"| tee -a "$LOG_FILE"
    echo "参数:TOP_TOOLS=$top_tools, insert_number=$insert_number"| tee -a "$LOG_FILE"

    # 删除旧文件
    rm -f ./baseline/output/baseline_results.json

    # 运行baseline
    echo "运行baseline..."
    # 清空文件
    cat /dev/null > ./test_yzx/selected_tools.txt

    # 修改 .env 文件中的 TOP_TOOLS 值
    sed -i "s/^TOP_TOOLS=.*/TOP_TOOLS=$top_tools/" .env
    echo "修改 .env 文件: TOP_TOOLS=$top_tools"

    # 调用 env_reset.sh 更新环境
    echo "调用 env_reset.sh 更新环境..."
    bash ./scripts/env_reset.sh

    # 显示当前 .env 中的 TOP_TOOLS 值以确认修改成功
    current_top_tools=$(grep "^TOP_TOOLS=" .env | cut -d'=' -f2)
    echo "当前 .env 中 TOP_TOOLS=$current_top_tools"

    uv run -m baseline.run_conversation \
        --input_path ./annotated_data/all_annotations.json \
        --output_path ./baseline/output/baseline_results.json \
        --insert_number $insert_number

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

    ((current_run++))
    sleep 2  # 可选延迟
done


echo ""
echo "全部实验执行完成！"