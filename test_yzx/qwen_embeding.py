# 文件名：embedding_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import uvicorn

# 模型路径
model_path = "/data/labshare/Param/Qwen/Qwen3-Embedding-0.6B"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()

# API 服务
app = FastAPI()

class EmbeddingRequest(BaseModel):
    input: list[str]

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    inputs = tokenizer(request.input, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": emb.cpu().tolist(),
                "index": i
            }
            for i, emb in enumerate(embeddings)
        ],
        "model": "qwen3-embedding-0.6b",
        "usage": {
            "prompt_tokens": int(inputs.input_ids.numel()),
            "total_tokens": int(inputs.input_ids.numel())
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7002)
