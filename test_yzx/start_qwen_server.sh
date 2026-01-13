export CUDA_VISIBLE_DEVICES=0

# python -m vllm.entrypoints.openai.api_server \
#     --model /data/labshare/Param/Qwen/qwen3-30b \
#     --tensor-parallel-size 4 \
#     --tool-call-parser hermes \
#     --enable-auto-tool-choice \
#     --port 7001

python -m vllm.entrypoints.openai.api_server \
    --model /home/yzx/models_weight/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.8 \
    --port 7001