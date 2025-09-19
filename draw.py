import re
import matplotlib.pyplot as plt
from datetime import datetime
import os
import argparse
def parse_log(filename,task_index):
    """解析日志文件，返回事件列表"""
    events = []
    current_time = 0.0
    llm_count, mcp_route_count, mcp_exec_count = 0, 0, 0
    total_time = None
    inside_task = False
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            # 判断是否进入对应 task
            task_match = re.match(r"task (\d+)", line)
            if task_match:
                current_task = int(task_match.group(1))
                inside_task = (current_task == task_index)
                continue  # task 行不计入事件

            if not inside_task:
                continue  # 跳过不属于目标 task 的行
            
            if "LLM response time" in line:
                duration = float(re.search(r"([\d.]+)s", line).group(1))
                llm_count += 1
                events.append(("llm_call", f"llm_call_{llm_count}", current_time, current_time + duration))
                current_time += duration
            elif "MCP Tool route execution time" in line:
                duration = float(re.search(r"([\d.]+)s", line).group(1))
                mcp_route_count += 1
                events.append(("mcp_route_exec", f"mcp_route_{mcp_route_count}", current_time, current_time + duration))
                current_time += duration

            elif "MCP Tool execute-tool execution time" in line:
                duration = float(re.search(r"([\d.]+)s", line).group(1))
                mcp_exec_count += 1
                events.append(("mcp_exec", f"mcp_exec_{mcp_exec_count}", current_time, current_time + duration))
                current_time += duration

            elif "Total query processing time" in line:
                total_time = float(re.search(r"([\d.]+)s", line).group(1))

    return events, total_time

def plot_timeline(events,max_tools, insert_number,task_index):
    """绘制执行时间分布图"""
    colors = {
        "llm_call": "orange",
        "mcp_route_exec": "red",
        "mcp_exec": "purple"
        # "init": "skyblue",
        # "local_py_exec": "green"
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    yticks, ylabels = [], []
    y = 0

    for phase_type in ["llm_call", "mcp_route_exec", "mcp_exec"]:
        yticks.append(y)
        ylabels.append(phase_type)
        for event in events:
            if event[0] == phase_type:
                ax.barh(y, event[3] - event[2], left=event[2],
                        color=colors[phase_type], edgecolor="black")
                ax.text((event[2] + event[3]) / 2, y, event[1],
                        ha="center", va="center", fontsize=8, color="black")
        y += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time (s)")
    ax.set_title("Execution Trace Timeline by Phase Type")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    filename = f"timeline_{task_index}_{max_tools}_{insert_number}.png"
    save_dir="./test_yzx/timeline_full"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # 保存图片
    print(f"图像已保存到: {save_path}")

def draw_from_log(log_file, max_tools, insert_number, task_index):
    events, total_time = parse_log(log_file,task_index)
    plot_timeline(events, max_tools, insert_number, task_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw execution timeline from log file.")
    parser.add_argument("--max_tools", type=int, required=True, help="最大工具数量")
    parser.add_argument("--insert_number", type=int, required=True, help="插入次数")
    parser.add_argument("--task_index", type=int, required=True, help="任务索引")
    args = parser.parse_args()
    log_file = "./test_yzx/time_log.txt"  # 你的日志文件路径
    draw_from_log(log_file, args.max_tools, args.insert_number, args.task_index)