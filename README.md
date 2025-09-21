## 环境启动
conda activate LiveMCPBench
cd LiveMCPBench
bash test_yzx/start_qwen_server.sh #启动vllm服务
bash test_yzx/start_qwen_embeding.sh #启动embeding模型

## 测试脚本
bash total_test.sh #无rag模块，测试prompt拼接
bash total_test_rag.sh #包含rag模块，且加入了rag结果的cache复用

## 结果示例
selected_tools.txt #记录每轮的llm挑选结果，新一轮结果会覆盖
judge.py #每轮进行正错判断
experiment.txt #所有测试轮次的正确率结果
rag_gt.txt #所有测试轮次中每个任务的标准答案、RAG结果和LLM挑选结果

## 结果处理
rag_gt_process.py #处理rag_gt.txt中的所有结果，并计算rag筛选的正确率，保存至rag_gt_result.txt
log2excel.py #将rag_gt_result.txt的数据生成excel

## 主要代码文件
baseline/run_conversation.py #主要代码
baseline/scripts/run_baselines.sh #全部任务测试脚本
annotated_data/all_annotations.json #全部任务的问题
baseline/scripts/run_example.sh #一个任务测试脚本
baseline/data/examole_queries.json #示例问题，仅包含一个任务
