export CUDA_VISIBLE_DEVICES=4,5,6,7

# python -m vllm.entrypoints.openai.api_server \
#     --model /data/labshare/Param/Qwen/qwen3-30b \
#     --tensor-parallel-size 4 \
#     --tool-call-parser hermes \
#     --enable-auto-tool-choice \
#     --port 7001

python -m vllm.entrypoints.openai.api_server \
    --model /data/labshare/Param/toolllama \
    --tensor-parallel-size 4 \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --chat-template ./test_yzx/toolllama_chat_template.jinja \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.8 \
    --port 7001
