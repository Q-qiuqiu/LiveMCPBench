import re

def process_log(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = []
    total_tasks = 0
    rag_correct_tasks = 0
    llm_correct_tasks = 0
    llm_correct_when_rag_wrong = 0  # RAG错误时LLM正确的次数
    rag_wrong_cases = 0  # RAG错误的案例总数
    params = ""
    inside_execution = False

    gt_tools = None  # 暂存 ground truth
    rag_tools = None  # 暂存 RAG 选择的工具
    for line in lines:
        line = line.strip()

        # 检测到一次新执行
        if line.startswith("第 ") and "次执行" in line:
            # 如果上一段有任务，计算并保存结果
            if inside_execution and total_tasks > 0:
                rag_acc = rag_correct_tasks / total_tasks
                llm_acc = llm_correct_tasks / rag_correct_tasks
                llm_when_rag_wrong_acc = llm_correct_when_rag_wrong / rag_wrong_cases if rag_wrong_cases > 0 else 0
                results.append(f"{params}, 总任务数={total_tasks}, RAG正确率={rag_acc:.2%}({rag_correct_tasks}/{total_tasks}), LLM正确率={llm_acc:.2%}({llm_correct_tasks}/{rag_correct_tasks}), RAG错误时LLM正确率={llm_when_rag_wrong_acc:.2%}({llm_correct_when_rag_wrong}/{rag_wrong_cases})")
  
            # 重置计数器
            inside_execution = True
            total_tasks = 0
            rag_correct_tasks = 0
            llm_correct_tasks = 0
            llm_correct_when_rag_wrong = 0
            rag_wrong_cases = 0
            params = ""
            gt_tools = None
            rag_tools = None
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
                    # 检查RAG是否正确
                    rag_correct = first_gt in rag_tools
                    if rag_correct:
                        rag_correct_tasks += 1
                    else:
                        rag_wrong_cases += 1  # 记录RAG错误的案例

        # LLM chose
        if "LLM chose" in line and gt_tools is not None:
            match = re.search(r"LLM chose (.+)", line)
            if match and gt_tools:
                chosen = match.group(1).strip()
                first_gt = gt_tools[0]
                # 检查LLM是否正确
                llm_correct = chosen == first_gt
                if llm_correct:
                    llm_correct_tasks += 1
                    
                    # 如果LLM正确且RAG错误，记录这种情况
                    if rag_tools is not None and first_gt not in rag_tools:
                        llm_correct_when_rag_wrong += 1
            gt_tools = None  # 一个任务处理完毕
            rag_tools = None

    # 处理最后一次执行
    if inside_execution and total_tasks > 0:
        rag_acc = rag_correct_tasks / total_tasks
        llm_acc = llm_correct_tasks / rag_correct_tasks
        llm_when_rag_wrong_acc = llm_correct_when_rag_wrong / rag_wrong_cases if rag_wrong_cases > 0 else 0
        results.append(f"{params}, 总任务数={total_tasks}, RAG正确率={rag_acc:.2%}({rag_correct_tasks}/{total_tasks}), LLM正确率={llm_acc:.2%}({llm_correct_tasks}/{rag_correct_tasks}), RAG错误时LLM正确率={llm_when_rag_wrong_acc:.2%}({llm_correct_when_rag_wrong}/{rag_wrong_cases})")
    # 打印和保存
    print(results)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    process_log("./test_yzx/rag_gt.txt", "./test_yzx/rag_gt_result.txt")
