import json

# 读取 JSON 文件
with open("./test_yzx/query_gt.json", "r", encoding="utf-8") as f:
    data = json.load(f)

correct = 0
total = 0

for task in data:
    logs = task["logs"]

    # 找出 ground_truth 和 tool_result
    gt = None
    result = None
    for log in logs:
        if log["type"] == "query":
            gt = log["ground_truth"][0]  # 取第一个工具
        elif log["type"] == "tool_result":
            result = log["tool_names"]

    # 判断是否匹配
    if gt and result:
        total += 1
        if gt in result:
            correct += 1
            print(f" Task {task['task_number']} 命中: {gt}")
        else:
            print(f" Task {task['task_number']} 未命中: {gt}")

# 输出正确率
if total > 0:
    accuracy = correct / total * 100
    print(f"\n总任务数: {total}")
    print(f"命中数: {correct}")
    print(f"正确率: {accuracy:.2f}%")
else:
    print("没有找到有效任务。")
