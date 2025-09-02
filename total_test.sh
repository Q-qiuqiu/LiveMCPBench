#!/bin/bash
# run_baseline_loop.sh

# 定义参数数组
max_tools_list=(100 150 200 300 350 400 450 500 )
insert_numbers=(0 30 50 70 90 )

# 计数器
total_runs=$(( ${#max_tools_list[@]} * ${#insert_numbers[@]} ))
current_run=1
LOG_FILE="./test_yzx/experiment_result.txt"
# 清空文件
cat /dev/null > "$LOG_FILE"
echo "开始批量执行baseline实验" | tee -a "$LOG_FILE"
echo "总共 $total_runs 个参数组合" | tee -a "$LOG_FILE"
echo "日志保存到: $LOG_FILE"
for max_tools in "${max_tools_list[@]}"; do
    for insert_number in "${insert_numbers[@]}"; do
        echo ""| tee -a "$LOG_FILE"
        echo "============================================================"| tee -a "$LOG_FILE"
        echo "第 $current_run/$total_runs 次执行"| tee -a "$LOG_FILE"
        echo "参数: max_tools=$max_tools, insert_number=$insert_number"| tee -a "$LOG_FILE"
        echo "============================================================"| tee -a "$LOG_FILE"
        
        # 删除旧文件
        rm -f ./baseline/output/baseline_results.json
        
        # 运行baseline
        echo "运行baseline..."
        # 清空文件
        cat /dev/null > ./test_yzx/selected_tools.txt
        uv run -m baseline.run_conversation \
            --input_path ./annotated_data/all_annotations.json \
            --output_path ./baseline/output/baseline_results.json \
            --max_tools $max_tools \
            --insert_number $insert_number
        
        if [ $? -eq 0 ]; then
            echo "baseline执行完成" | tee -a "$LOG_FILE"
            
            # 运行judge
            echo "📊 运行judge.py..."
            python3 ./test_yzx/judge.py 2>&1 | tee -a "$LOG_FILE"
            
            if [ $? -eq 0 ]; then
                echo "judge.py执行完成" | tee -a "$LOG_FILE"
            else
                echo "judge.py执行失败" | tee -a "$LOG_FILE"
            fi
        else
            echo " baseline执行失败" | tee -a "$LOG_FILE"
        fi
        
        ((current_run++))
        sleep 2  # 可选延迟
    done
done

echo ""
echo "全部实验执行完成！"