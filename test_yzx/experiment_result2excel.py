import pandas as pd
import re

# 从txt文件读取日志数据
def read_log_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 解析日志数据
def parse_log_data(log_text):
    # 修改正则表达式，匹配MAX_TOOLS而不是TOP_TOOLS
    pattern = r'第 \d+/\d+ 次执行\s*参数:MAX_TOOLS=(\d+),\s*insert_number=(\d+)[\s\S]*?Accuracy:\s*([\d.]+)%'
    matches = re.findall(pattern, log_text)
    
    data = []
    for match in matches:
        max_tools = int(match[0])
        insert_number = int(match[1])
        accuracy = float(match[2])
        data.append((max_tools, insert_number, accuracy))
        print(f"提取到: MAX_TOOLS={max_tools}, insert_number={insert_number}, Accuracy={accuracy}")
    
    return data

# 主程序
def main():
    # 读取日志文件
    log_text = read_log_file('test_yzx/experiment_result.txt')
    
    # 解析数据
    parsed_data = parse_log_data(log_text)
    
    if not parsed_data:
        print("没有提取到任何数据，请检查日志格式")
        # 打印部分日志内容帮助调试
        print("日志前500字符:")
        print(log_text[:500])
        return
    
    # 创建DataFrame
    df = pd.DataFrame(parsed_data, columns=['MAX_TOOLS', 'insert_number', 'Accuracy'])
    
    print("\n原始数据:")
    print(df)
    
    # 重塑数据：MAX_TOOLS作为行索引，insert_number作为列
    pivot_table = df.pivot_table(
        index='MAX_TOOLS', 
        columns='insert_number', 
        values='Accuracy', 
        aggfunc='first'
    )
    
    # 重置索引，使MAX_TOOLS成为普通列
    pivot_table.reset_index(inplace=True)
    pivot_table.columns.name = None  # 移除列名
    
    print("\n生成的表格:")
    print(pivot_table)
    
    # 保存为Excel文件
    pivot_table.to_excel('experiment_result.xlsx', index=False)
    print("\n表格已保存为 'experiment_result.xlsx'")

if __name__ == "__main__":
    main()