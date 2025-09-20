import re

def process_log(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    total_tasks = 0
    correct_tasks = 0
    params = ""
    inside_execution = False

    gt_tools = None  # 暂存 ground truth
    for line in lines:
        line = line.strip()

        # 检测到一次新执行
        if line.startswith("第 ") and "次执行" in line:
            # 如果上一段有任务，计算并保存结果
            if inside_execution and total_tasks > 0:
                accuracy = correct_tasks / total_tasks
                results.append(f"{params}, 正确率={accuracy:.2%}")
            
            # 重置计数器
            inside_execution = True
            total_tasks = 0
            correct_tasks = 0
            params = ""
            continue

        # 获取参数
        if line.startswith("参数:"):
            params = line.replace("参数:", "").strip()

        # ground truth
        if "Ground-truth tools:" in line:
            match = re.search(r"Ground-truth tools: \[(.*)\]", line)
            if match:
                gt_tools = [t.strip().strip("'") for t in match.group(1).split(",")]
        # rag selected
        if "RAG selected tools:" in line and gt_tools is not None:
            match = re.search(r"RAG selected tools: \[(.*)\]", line)
            if match:
                rag_tools = [t.strip().strip("'") for t in match.group(1).split(",")]

                if gt_tools:  # ground truth 至少有一个工具
                    first_gt = gt_tools[0]
                    total_tasks += 1
                    if first_gt in rag_tools:
                        correct_tasks += 1
            gt_tools = None  # 处理完一个任务，清空 ground truth

    # 处理最后一次执行
    if inside_execution and total_tasks > 0:
        accuracy = correct_tasks / total_tasks
        results.append(f"{params}, 正确率={accuracy:.2%}")
    print(results)
    # 写入结果文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    process_log("./test_yzx/rag_gt.txt", "./test_yzx/rag_gt_result.txt")
