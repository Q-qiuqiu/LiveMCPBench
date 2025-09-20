import re
import pandas as pd

def log_to_excel(input_file, output_file):
    # 读取日志
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []

    for line in lines:
        line = line.strip()
        match = re.match(r"TOP_TOOLS=(\d+), insert_number=(\d+), 正确率=([\d.]+)%", line)
        if match:
            top_tools = int(match.group(1))
            insert_number = int(match.group(2))
            accuracy = float(match.group(3))  # 去掉百分号
            data.append((top_tools, insert_number, accuracy))

    # 转换为 DataFrame
    df = pd.DataFrame(data, columns=["TOP_TOOLS", "insert_number", "accuracy"])

    # 去重，只保留第一次出现的组合
    df = df.drop_duplicates(subset=["TOP_TOOLS", "insert_number"], keep="first")

    # 透视表：TOP_TOOLS -> 行，insert_number -> 列
    pivot = df.pivot(index="TOP_TOOLS", columns="insert_number", values="accuracy")

    # 保存为 Excel
    pivot.to_excel(output_file, float_format="%.2f")

    print(f"转换完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    log_to_excel("./test_yzx/rag_gt_result.txt", "accuracy_table.xlsx")
