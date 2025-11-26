#!/bin/bash
# start_models_env.sh

set -m

echo "🚀 启动所有模型..."

cd /home/yzx/LiveMCPBench/test_yzx

echo "📁 当前工作目录: $(pwd)"

# 创建日志目录
mkdir -p ./logs

# 存储进程PID和状态
declare -A PIDS
declare -A STATUS
FAILED_MODELS=()

# 清理函数
cleanup() {
    echo ""
    echo "🛑 收到终止信号，正在清理进程..."
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            echo "📛 终止进程: $pid"
            kill $pid
        fi
    done
    echo "✅ 清理完成"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 启动模型函数
start_model() {
    local model_name=$1
    local gpu=$2
    local model_path="/home/yzx/LiveMCPBench/test_yzx/embedding/$model_name"
    
    echo ""
    echo "🎯 启动: $model_name -> GPU $gpu"
    
    # 检查文件是否存在
    if [ ! -f "$model_path" ]; then
        echo "❌ 错误: 文件不存在 - $model_path"
        STATUS["$model_name"]="FILE_NOT_FOUND"
        FAILED_MODELS+=("$model_name")
        return 1
    fi
    
    # 启动模型并记录PID
    env CUDA_VISIBLE_DEVICES=$gpu python3 "$model_path" > "./logs/${model_name}.log" 2>&1 &
    local pid=$!
    PIDS["$model_name"]=$pid
    
    # 等待2秒检查进程是否存活
    sleep 2
    if kill -0 $pid 2>/dev/null; then
        echo "✅ 启动成功: $model_name (PID: $pid)"
        STATUS["$model_name"]="RUNNING"
        return 0
    else
        echo "❌ 启动失败: $model_name (进程已退出)"
        STATUS["$model_name"]="FAILED"
        FAILED_MODELS+=("$model_name")
        return 1
    fi
}

# 启动所有模型
echo "📋 开始启动模型..."

# GPU 1 的模型
# start_model "qwen_embeding.py" 1
# start_model "bge-base-en-v1.5_embeding.py" 1
# start_model "bge-large-en-v1.5_embeding.py" 1
# start_model "e5-base-v2_embeding.py" 1
# start_model "e5-large-v2_embeding.py" 1

# GPU 2 的模型
start_model "tool-bge-base-en-v1.5_embeding.py" 2
start_model "tool-bge-large-en-v1.5_embeding.py" 2
start_model "tool-e5-base-v2_embeding.py" 2
start_model "tool-e5-large-v2_embeding.py" 2
start_model "toolbench_embeding.py" 2

# 显示启动总结
echo ""
echo "📊 ===== 启动总结 ====="
echo "✅ 成功启动: $(( ${#PIDS[@]} - ${#FAILED_MODELS[@]} )) 个模型"
echo "❌ 启动失败: ${#FAILED_MODELS[@]} 个模型"

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "📛 失败的模型:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "   - $model (状态: ${STATUS[$model]})"
    done
    echo ""
    echo "💡 提示: 查看日志文件了解失败原因:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "   - tail -n 20 ./logs/${model}.log"
    done
fi

echo ""
echo "🔄 运行中的模型PID:"
for model in "${!PIDS[@]}"; do
    if [ "${STATUS[$model]}" = "RUNNING" ]; then
        echo "   - $model: ${PIDS[$model]}"
    fi
done

echo ""
echo "⏳ 所有模型运行中... (按 Ctrl+C 停止所有进程)"
echo "📝 查看日志: tail -f ./logs/*.log"

# 健康监控循环（可选）
echo ""
echo "🔍 开始健康监控..."
while [ ${#PIDS[@]} -gt 0 ]; do
    sleep 30
    ALIVE_COUNT=0
    for model in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$model]}" 2>/dev/null; then
            ((ALIVE_COUNT++))
        else
            echo "❌ 进程异常退出: $model (PID: ${PIDS[$model]})"
            unset PIDS["$model"]
        fi
    done
    echo "📊 健康检查: $ALIVE_COUNT 个模型正常运行"
    
    # 如果所有进程都退出了，退出循环
    if [ $ALIVE_COUNT -eq 0 ]; then
        echo "💥 所有模型进程已退出"
        break
    fi
done

wait