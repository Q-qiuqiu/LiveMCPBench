export CUDA_VISIBLE_DEVICES=0,1,2,3

# python -m vllm.entrypoints.openai.api_server \
#     --model /data/labshare/Param/Qwen/qwen3-30b \
#     --tensor-parallel-size 4 \
#     --tool-call-parser hermes \
#     --enable-auto-tool-choice \
#     --port 7001

python -m vllm.entrypoints.openai.api_server \
    --model /data/labshare/Param/Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --tool-call-parser hermes \
    --enable-auto-tool-choice \
    --max-model-len 196608 \
    --port 7001