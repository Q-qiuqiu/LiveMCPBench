import re

def parse_tool_log(log_file):
    """
    解析日志文件，将类似：
    0.get-weread-rank
    1.load
    2.geocode_address
    ...
    转成工具名称列表
    """
    tools = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            # 使用正则提取数字后面的工具名
            match = re.match(r"\d+\.(.+)", line.strip())
            if match:
                tools.append(match.group(1).strip())
    return tools


def compute_accuracy(ground_truth, predicted):
    """
    计算标准答案与预测的匹配正确率
    简单方式：逐个匹配，匹配成功计数 / 总数
    """
    n = len(ground_truth)
    correct = 0
    for gt, pred in zip(ground_truth, predicted):
        if gt == pred:
            correct += 1
    accuracy = correct / n if n > 0 else 0
    return accuracy, correct, n


if __name__ == "__main__":
    gt_file = "./test_yzx/answer_tools.txt"
    pred_file = "./test_yzx/selected_tools.txt"

    gt_tools = parse_tool_log(gt_file)
    pred_tools = parse_tool_log(pred_file)

    accuracy, correct_count, total = compute_accuracy(gt_tools, pred_tools)



    # 可选：打印逐条匹配情况
    # for i, (gt, pred) in enumerate(zip(gt_tools, pred_tools)):
    #     print(f"{i}: {gt} | {pred} | {'✔' if gt==pred else '✘'}")

    print(f"Accuracy: {accuracy:.2%}({correct_count}/{total})")