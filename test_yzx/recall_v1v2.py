import re
import json
import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()

class RAGPerformanceEvaluator:
    def __init__(self, log_file: str, output_json: str = "./rag_evaluation_log.json"):
        self.log_file = log_file
        self.output_json = output_json
        self.total_tasks = 0
        self.recall_correct_tasks = 0
        self.problem_solved_tasks = 0
        self.evaluation_logs = []
        self.task_queries = {}
        
        # 初始化OpenAI客户端
        self.llm_client = openai.AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL")
        )
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")
        
        # 加载任务查询信息
        self.load_task_queries()
    
    def load_task_queries(self):
        """从all_annotations.json文件加载任务查询信息"""
        annotations_path = "annotated_data/all_annotations.json"
        
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                    # 假设任务索引从1开始，与log文件中的索引对应
                    for i, item in enumerate(annotations, 1):
                        if 'Question' in item:
                            self.task_queries[i] = item['Question']
                print(f"成功加载了 {len(self.task_queries)} 个任务的查询信息")
            except Exception as e:
                print(f"警告: 无法加载任务查询信息 {annotations_path}: {e}")
    
    def load_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """加载工具描述和参数信息"""
        # 这里假设工具信息存储在tools.json文件中
        # 实际实现时可能需要根据项目结构调整路径
        tools_info = {}
        tools_json_path = "./tools/LiveMCPTool/tools.json"
        print("加载工具信息中...")
        if os.path.exists(tools_json_path):
            try:
                with open(tools_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 解析工具信息的逻辑可能需要根据实际数据格式调整
                    for server_entry in data:
                        server_name = list(server_entry['config']['mcpServers'].keys())[0]
                        tools = server_entry['tools'].get(server_name, {}).get('tools', [])
                        for tool in tools:
                            tools_info[tool['name']] = {
                                'description': tool.get('description', ''),
                                'parameters': tool.get('inputSchema', {}).get('properties', {})
                            }
                            #print(f"加载工具 {tool['name']} 信息")
            except Exception as e:
                print(f"警告: 无法加载工具信息文件 {tools_json_path}: {e}")
        
        return tools_info
    
    async def llm_judge_tool_relevance(self, query: str, tool_name: str, tool_info: Dict[str, Any]) -> Dict[str, Any]:
        """使用大模型判断工具是否可以解决问题"""
        try:
            prompt = f"""
            请判断以下工具是否能够解决用户的查询问题。
            
            用户查询: {query}
            
            工具信息:
            - 工具名称: {tool_name}
            - 工具描述: {tool_info.get('description', '无描述')}
            - 工具参数: {json.dumps(tool_info.get('parameters', {}), ensure_ascii=False, indent=2)}
            
            请只回答'能'或'不能'，不要添加任何其他内容。
            """
            
            response = await self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的工具评估专家，负责判断工具是否能够解决用户的查询问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            judge_result = response.choices[0].message.content.strip()
            can_solve = "能" in judge_result
            
            return {
                "tool_name": tool_name,
                "can_solve": can_solve,
                "description": tool_info.get('description', '无描述'),
                "parameters": tool_info.get('parameters', {})
            }
        except Exception as e:
            print(f"LLM判断失败: {e}")
            return {
                "tool_name": tool_name,
                "can_solve": False,
                "description": tool_info.get('description', '无描述'),
                "parameters": tool_info.get('parameters', {})
            }
    
    async def process_file(self):
        """处理日志文件并评估性能"""
        if not os.path.exists(self.log_file):
            print(f"错误: 找不到文件 {self.log_file}")
            return
        
        # 加载工具信息
        tools_info = self.load_tools_info()
        print(f"成功加载了 {len(tools_info)} 个工具的信息")
        with open(self.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # 检查是否是任务行
            if i < len(lines) and re.match(r'^\d+\.Ground-truth tools:', lines[i]):
                self.total_tasks += 1
                
                # 解析任务信息
                task_index = int(re.match(r'^(\d+)\.', lines[i]).group(1))
                print(f"正在处理任务 {task_index}")
                
                # 解析Ground-truth tools
                gt_match = re.search(r"\[(.*?)\]", lines[i])
                gt_tools = eval(f"[{gt_match.group(1)}]") if gt_match else []
                
                # 解析RAG selected tools
                rag_match = re.search(r"\[(.*?)\]", lines[i+1])
                rag_tools = eval(f"[{rag_match.group(1)}]") if rag_match else []
                
                # 解析LLM选择
                llm_choice = lines[i+2].split("LLM chose ")[1].strip()
                
                # 检查recall是否正确
                has_overlap = bool(set(gt_tools) & set(rag_tools))
                if has_overlap:
                    self.recall_correct_tasks += 1
                    # 移除自动增加问题解决任务数的逻辑，只在recall失败时计算
                    
                    # 记录日志，包含查询信息和工具详情
                    recall_log = {
                        "task_index": task_index,
                        "query": self.task_queries.get(task_index, f"任务 {task_index} 的查询"),
                        "ground_truth_tools": [],
                        "rag_selected_tools": [],
                        "llm_choice": llm_choice,
                        "recall_correct": True,
                        "problem_solved": False,
                        "llm_judgments": []
                    }
                    
                    # 为每个工具添加详情
                    for tool_name in gt_tools:
                        tool_info = tools_info.get(tool_name, {'description': '未知', 'parameters': {}})
                        recall_log["ground_truth_tools"].append({
                            "name": tool_name,
                            "description": tool_info.get('description', '无描述'),
                            "parameters": tool_info.get('parameters', {})
                        })
                    
                    for tool_name in rag_tools:
                        tool_info = tools_info.get(tool_name, {'description': '未知', 'parameters': {}})
                        recall_log["rag_selected_tools"].append({
                            "name": tool_name,
                            "description": tool_info.get('description', '无描述'),
                            "parameters": tool_info.get('parameters', {})
                        })
                    
                    self.evaluation_logs.append(recall_log)
                else:
                    # 需要调用LLM判断每个工具
                    llm_judgments = []
                    problem_solved = False
                    
                    # 从加载的任务查询信息中获取真实的query
                    query = self.task_queries.get(task_index, f"任务 {task_index} 的查询")
                    
                    # 对每个RAG选择的工具进行判断
                    for tool_name in rag_tools[:3]:  # 限制判断前3个工具以节省资源
                        tool_info = tools_info.get(tool_name, {'description': '未知', 'parameters': {}})
                        judgment = await self.llm_judge_tool_relevance(query, tool_name, tool_info)
                        llm_judgments.append(judgment)
                        
                        if judgment['can_solve']:
                            problem_solved = True
                    
                    if problem_solved:
                        self.problem_solved_tasks += 1
                    
                    # 记录日志，包含查询信息和工具详情
                    recall_log = {
                        "task_index": task_index,
                        "query": query,
                        "ground_truth_tools": [],
                        "rag_selected_tools": [],
                        "llm_choice": llm_choice,
                        "recall_correct": False,
                        "problem_solved": problem_solved,
                        "llm_judgments": llm_judgments
                    }
                    
                    # 为每个工具添加详情
                    for tool_name in gt_tools:
                        tool_info = tools_info.get(tool_name, {'description': '未知', 'parameters': {}})
                        recall_log["ground_truth_tools"].append({
                            "name": tool_name,
                            "description": tool_info.get('description', '无描述'),
                            "parameters": tool_info.get('parameters', {})
                        })
                    
                    for tool_name in rag_tools:
                        tool_info = tools_info.get(tool_name, {'description': '未知', 'parameters': {}})
                        recall_log["rag_selected_tools"].append({
                            "name": tool_name,
                            "description": tool_info.get('description', '无描述'),
                            "parameters": tool_info.get('parameters', {})
                        })
                    
                    self.evaluation_logs.append(recall_log)
                
                # 移动到下一个任务
                i += 3
            else:
                i += 1
        
        # 保存日志到JSON文件
        self.save_logs()
        
        # 计算并打印结果
        self.print_results()
    
    def save_logs(self):
        """保存评估日志到JSON文件"""
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        with open(self.output_json, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_logs, f, indent=2, ensure_ascii=False)
        print(f"评估日志已保存到 {self.output_json}")
    
    def print_results(self):
        """打印评估结果"""
        recall_accuracy = (self.recall_correct_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0
        failed_recall_tasks = self.total_tasks - self.recall_correct_tasks
        problem_solved_accuracy = (self.problem_solved_tasks / failed_recall_tasks * 100) if failed_recall_tasks > 0 else 0
        
        print("=" * 50)
        print(f"评估结果:")
        print(f"总任务数: {self.total_tasks}")
        print(f"Recall正确任务数: {self.recall_correct_tasks}")
        print(f"Recall失败任务数: {failed_recall_tasks}")
        print(f"问题解决任务数(在Recall失败的任务中): {self.problem_solved_tasks}")
        print(f"Recall正确率: {recall_accuracy:.2f}%")
        print(f"问题解决正确率(在Recall失败任务中的比例): {problem_solved_accuracy:.2f}%")
        print("=" * 50)

async def main():
    evaluator = RAGPerformanceEvaluator(
        log_file="./test_yzx/rag_gt.txt",
        output_json="./test_yzx/rag_evaluation_log.json"
    )
    await evaluator.process_file()

if __name__ == "__main__":
    asyncio.run(main())