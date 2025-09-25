import re
import pandas as pd

def log_to_excel(input_file, output_file):
    # 读取日志
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []

    for line in lines:
        line = line.strip()
        # 修改正则表达式匹配新格式
        match = re.match(r"TOP_TOOLS=(\d+), insert_number=(\d+), 总任务数=\d+, RAG正确率=([\d.]+)%\([^)]+\), LLM正确率=([\d.]+)%", line)
        if match:
            top_tools = int(match.group(1))
            insert_number = int(match.group(2))
            rag_accuracy = float(match.group(3))  # RAG正确率
            llm_accuracy = float(match.group(4))  # LLM正确率
            
            data.append((top_tools, insert_number, rag_accuracy, llm_accuracy))

    # 创建DataFrame
    df = pd.DataFrame(data, columns=["TOP_TOOLS", "insert_number", "RAG_accuracy", "LLM_accuracy"])

    # 去重，只保留第一次出现的组合
    df = df.drop_duplicates(subset=["TOP_TOOLS", "insert_number"], keep="first")

    # 创建透视表
    pivot_rag = df.pivot(index="TOP_TOOLS", columns="insert_number", values="RAG_accuracy")
    pivot_llm = df.pivot(index="TOP_TOOLS", columns="insert_number", values="LLM_accuracy")

    # 合并两个表格到一个DataFrame中
    combined_df = pd.concat([pivot_rag, pivot_llm], keys=['RAG正确率', 'LLM正确率'])

    # 保存到Excel
    combined_df.to_excel(output_file, float_format="%.2f")

    print(f"转换完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    log_to_excel("./test_yzx/rag_gt_result.txt", "accuracy_table.xlsx")