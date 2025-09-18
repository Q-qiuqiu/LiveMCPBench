import re
import matplotlib.pyplot as plt

def parse_log(filename):
    """解析日志文件，返回事件列表"""
    events = []
    current_time = 0.0
    llm_count, mcp_route_count, mcp_exec_count = 0, 0, 0
    total_time = None
    with open(filename, "r") as f:
        for line in f:
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

def plot_timeline(events):
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
    save_path="./test_yzx/execution_timeline.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # 保存图片
    print(f"图像已保存到: {save_path}")

if __name__ == "__main__":
    log_file = "./test_yzx/time_log.txt"  # 你的日志文件路径
    events, total_time = parse_log(log_file)  #解包两个返回值
    plot_timeline(events)  # 只传 events
