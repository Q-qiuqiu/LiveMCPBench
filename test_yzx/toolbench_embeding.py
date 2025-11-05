
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import uvicorn

# 模型路径
model_path = "/data/labshare/Param/toolbench_retriver"

# 加载 SentenceTransformer 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_path, device=device)

# API 服务
app = FastAPI()

class EmbeddingRequest(BaseModel):
    input: list[str]

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    # 计算嵌入向量
    embeddings = model.encode(
        request.input,
        convert_to_tensor=True,
        normalize_embeddings=True,  # 可选：归一化嵌入向量
        device=device
    )

    # 转为列表以便返回 JSON
    embeddings_list = embeddings.cpu().tolist()

    # 统计 token 数量（近似值）
    # SentenceTransformer 没有 tokenizer.input_ids.numel()，我们简单估计
    total_tokens = sum(len(text.split()) for text in request.input)

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": emb,
                "index": i
            }
            for i, emb in enumerate(embeddings_list)
        ],
        "model": "toolbench_retriver",
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7007)
