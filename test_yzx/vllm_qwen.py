# SPDX-License-Identifier: Apache-2.0
# # SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# from vllm import LLM, SamplingParams

# # Sample prompts.
# prompts = [
#     "你好，介绍一下你自己。",
# ]
# # Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# # 指定本地模型路径
# model_path = "/data/labshare/Param/Qwen/qwen3-30b"

# def main():
#     # Create an LLM.
#     llm = LLM(
#         model=model_path,
#         tensor_parallel_size=8   # 八卡并行
#         #dtype="bfloat16",         # 30B 建议用 bf16 节省显存
#         )
#     # Generate texts from the prompts.
#     # The output is a list of RequestOutput objects
#     # that contain the prompt, generated text, and other information.
#     outputs = llm.generate(prompts, sampling_params)
#     # Print the outputs.
#     print("\nGenerated Outputs:\n" + "-" * 60)
#     for output in outputs:
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         print(f"Prompt:    {prompt!r}")
#         print(f"Response:    {generated_text!r}")
#         print("-" * 60)


# if __name__ == "__main__":
#     main()







from openai import OpenAI

# 配置本地 API
client = OpenAI(
    api_key="EMPTY",                  # vLLM 不校验 key
    base_url="http://localhost:7001/v1/"
)

# 聊天补全调用
response = client.chat.completions.create(
    model="/data/labshare/Param/Qwen/QwQ-32B",   # 模型名称随意填，vLLM 会忽略
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
