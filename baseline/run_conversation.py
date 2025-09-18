import asyncio
import json
import logging
import os
import pathlib
import traceback
from typing import List, Optional, Tuple
import time
import dotenv
from mcp import ClientSession
from tqdm import tqdm
import argparse
import uuid
import re
import yaml
from utils.clogger import _set_logger
from utils.llm_api import ChatModel
from utils.mcp_client import MCPClient

_set_logger(
    exp_dir=pathlib.Path("./logs"),
    logging_level_stdout=logging.INFO,
    logging_level=logging.DEBUG,
    file_name="baseline.log",
)
dotenv.load_dotenv()
logger = logging.getLogger(__name__)

INPUT_QUERIES_FILE = "./baseline/data/example_queries.json"
CONVERSATION_RESULTS_FILE = f"./baseline/output/{os.getenv('MODEL', 'None').replace('/', '_')}_{os.getenv('EMBEDDING_MODEL', 'None').replace('/', '_')}.json"
TOOLS_FILE = "./tools/LiveMCPTool/tools.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default=INPUT_QUERIES_FILE,
        help="Path to the input queries file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=CONVERSATION_RESULTS_FILE,
        help="Path to the output conversation results file.",
    )
    parser.add_argument(
        "--tools_path",
        type=str,
        default=TOOLS_FILE,   # 默认值
        help="Path to the tools.json file.",
    )
    parser.add_argument(
        "--insert_number",
        type=int,
        default=0,
        help="Insert ground-truth tools after N tools (0 = insert at beginning).",
    )
    parser.add_argument(
        "--max_tools",
        type=int,
        default=None,
        help="Maximum number of tools to load from tools.json. If None, load all.",
)
    return parser.parse_args()

def parse_answer_tools(entry):
    """
    从 entry 中解析出正确答案工具列表
    """
    tools_text = entry["Annotator Metadata"]["Tools"]
    answer_tools = []
    for line in tools_text.splitlines():
        # 形如 "1. get-weread-rank"
        parts = line.split(".", 1)
        if len(parts) == 2:
            tool_name = parts[1].strip()
            answer_tools.append(tool_name)
    return answer_tools


def build_prompt_from_rag(tools_file: str, 
                        rag_tools: List[str],     # RAG 筛选出的工具名
                        answer_tools: List[str],
                        insert_number: int = 0,
                        task_index: int = 0) -> str:
    """
    根据 rag_tools 和 tools.json 生成 system prompt
    """

    with open(tools_file, "r", encoding="utf-8") as f:
        tools_data = json.load(f)
    # 构建工具映射: tool_name -> 工具对象
    tools_map = {}
    for entry in tools_data:
        for server_id, server in entry.get("tools", {}).items():
            for tool in server.get("tools", []):
                tools_map[tool["name"]] = {
                    "server_id": server_id,
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {"type":"object","properties":{}})
                }
    #print(f"Ground-truth tools: {answer_tools}")
    #print(f"RAG selected tools: {rag_tools}")
    # 写入日志文件
    with open("./test_yzx/rag_gt.txt", "a", encoding="utf-8") as f:
        f.write(f"{task_index}.Ground-truth tools: {answer_tools}"+ "\n" + f"RAG selected tools: {rag_tools}" + "\n")

    # 先去重保持顺序
    rag_tools_unique = []
    for t in rag_tools:
        if t not in rag_tools_unique:
            rag_tools_unique.append(t)
    # RAG 命中正确答案的工具（交集部分）
    hit_tools = [t for t in rag_tools_unique if t in answer_tools]  

    # 剩余工具（rag_tools 里有但不在正确答案里的）
    remaining_tools = [t for t in rag_tools_unique if t not in hit_tools]

    # 控制插入位置不越界
    insert_number = max(0, min(insert_number, len(remaining_tools)))

    # 拼接最终顺序
    final_tools = remaining_tools[:insert_number] + hit_tools + remaining_tools[insert_number:]

    # 系统提示开头
    prompt_lines = [
        "You are an intelligent assistant designed to help users accomplish tasks using a set of MCP tools.\n",
        "Important rules:\n"
        "1. You have access to a single execution tool called 'execute-tool'. You must always use this tool to invoke any of the available MCP tools.\n"
        "2. Never call MCP tools directly. Always select the appropriate tool and call it via 'execute-tool'.\n"
        "3. Provide accurate and complete responses to the user. You can combine multiple tool calls if necessary, but respond to the user only once.\n"
        "4. For each tool call, specify:\n"
        "   - server_name: the MCP server hosting the target tool\n"
        "   - tool_name: the name of the target tool\n"
        "   - params: a dictionary of input parameters for the tool\n\n"
        "Available MCP tools (choose from these, but always call via 'execute-tool'):\n"
    ]

    # 遍历最终工具列表，拼接到 prompt
    for tool_name in final_tools:
        if tool_name in tools_map:
            t = tools_map[tool_name]
            input_props = t["inputSchema"].get("properties", {})
            prompt_lines.append(
                f"- {tool_name} (server: {t['server_id']}): {t['description']}. Input: {json.dumps(input_props)}"
            )
        else:
            prompt_lines.append(
                f"- {tool_name} (server: unknown): description missing. Input: {{}}"
            )
            logger.warning(f"RAG tool {tool_name} not found in tools.json")

    return "\n".join(prompt_lines)

def clean_tool_description(tool_text):
    """清理工具描述中的非法转义字符"""
    # 修复无效的 Unicode 转义
    tool_text = re.sub(r'\\U(?![\da-fA-F]{8})', r'\\\\U', tool_text)
    # 修复其他非法转义
    tool_text = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', tool_text)
    return tool_text

def reorder_tools(response, answer_tools: list[str], insert_index: int) -> str:
    """
    调整 matched_tools 的顺序，把所有正确答案工具移动到 target_index 开始的位置。
    其他工具保持原顺序。
    
    response: 大模型返回的原始字符串
    answer_tools: 正确答案工具名列表，例如 ["mfcc", "chroma_cqt"]
    insert_index: 正确答案工具放置的位置 (0 表示第一个)
    """
    text = response.content[0].text
    # 提取 matched_tools 下的所有工具块
    tool_pattern = r"(?m)^- server_name:.*?(?=^-\sserver_name:|\Z)"
    tools = re.findall(tool_pattern, text, flags=re.S)
    if not tools:
        return response  # 没有工具，直接返回
    cleaned_tools = [clean_tool_description(tool) for tool in tools]
    # 分离正确答案工具 & 其他
    matched = [t for t in cleaned_tools if any(f"tool_name: {ans}" in t for ans in answer_tools)]
    remaining = [t for t in cleaned_tools if t not in matched]

    # 新顺序
    new_tools = remaining.copy()
    if insert_index < 0:
        insert_index = 0
    if insert_index > len(remaining):
        insert_index = len(remaining)
    new_tools[insert_index:insert_index] = matched


    # 拼回文本（替换原 matched_tools 部分）
    new_text = re.sub(r"matched_tools:\n.*", "matched_tools:\n" + "\n".join(new_tools), text, flags=re.S)

    # 替换 response 内部的 text
    response.content[0].text = new_text
    return response

def show_tools(response, log_text):
    # 获取 text 内容
    if hasattr(response, "content") and len(response.content) > 0:
        text = response.content[0].text
    else:
        raise ValueError("CallToolResult 对象中没有 content 或为空") 
    matched_tools_yaml = "\n".join(text.split("matched_tools:")[1].splitlines())

    # 加载为 Python 对象
    tools_list = yaml.safe_load(matched_tools_yaml)
    print(f"{log_text} tools: {[tool['tool_name'] for tool in tools_list]}")


def extract_tools(response) -> list:
    """
    从 response 中解析 matched_tools，提取工具名并返回列表
    """
    # 获取 text 内容
    if hasattr(response, "content") and len(response.content) > 0:
        text = response.content[0].text
    else:
        raise ValueError("CallToolResult 对象中没有 content 或为空") 

    # 提取 matched_tools 部分
    if "matched_tools:" not in text:
        raise ValueError("response.text 中没有找到 'matched_tools:' 段落")

    matched_tools_yaml = "\n".join(text.split("matched_tools:")[1].splitlines())

    # 加载为 Python 对象
    tools_list = yaml.safe_load(matched_tools_yaml)

    # 提取 tool_name
    rag_tools = [tool["tool_name"] for tool in tools_list if "tool_name" in tool]
    return rag_tools


class LoggingMCPClient(MCPClient):
    def __init__(self):
        super().__init__(timeout=180, max_sessions=9999)
        self.chat_model = ChatModel(
            model_name=os.getenv("MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model_url=os.getenv("BASE_URL"),
        )

    async def connect_copilot(self):
        if "mcp-copilot" not in self.sessions:
            await self.config_connect(
                config={
                    "mcpServers": {
                        "mcp-copilot": {
                            "command": "python",
                            "args": ["-m", "baseline.mcp_copilot"],
                        },
                    }
                },
            )
            logger.info("Connected to MCP Copilot server.")

    async def process_query(
        self,
        query: str,
        answer_tools: list,
        tools_file: str = TOOLS_FILE,
        max_tools: Optional[int] = None,
        insert_number: int = 0,
        task_index: int = 0,
        history: Optional[list] = None,
        max_tool_tokens: int = 10000,
    ) -> Tuple[str, List[dict]]:
        if history is None:
            messages = [
                {
                    "role": "system",
                    "content": """\
You are an agent designed to assist users with daily tasks by using external tools. You have access to two tools: a retrieval tool and an execution tool. The retrieval tool allows you to search a large toolset for relevant tools, and the execution tool lets you invoke the tools you retrieved. Whenever possible, you should use these tools to get accurate, up-to-date information and to perform file operations.

Note that you can only response to user once and only use the retrieval tool once, so you should try to provide a complete answer in your response.
""",
                }
            ]
        else:
            messages = history.copy()

        messages.append({"role": "user", "content": query})
        available_tools = []
        for server in self.sessions:
            session = self.sessions[server]
            assert isinstance(session, ClientSession), (
                "Session must be an instance of ClientSession"
            )
            response = await session.list_tools()
            for tool in response.tools:
                available_tools += [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                ]
        final_text = []
        stop_flag = False
        try:
            while not stop_flag:       
                request_payload = {
                    "messages": messages,
                    "tools": available_tools,
                }
                #print("request_payload",request_payload)
                # === 大模型思考时间 ===

                llm_start = time.perf_counter()
                response = self.chat_model.complete_with_retry(**request_payload)
                llm_end = time.perf_counter()
                logger.info(f"LLM response time: {llm_end - llm_start:.3f}s")
                #写入日志文件
                with open("./test_yzx/time_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"LLM response time: {llm_end - llm_start:.3f}s" + "\n")

                if hasattr(response, "error"):
                    raise Exception(
                        f"Error in OpenAI response: {response.error['metadata']['raw']}"
                    )
                response_message = response.choices[0].message
                if response_message.tool_calls:
                    tool_call_list = []
                    for tool_call in response_message.tool_calls:
                        if not tool_call.id:
                            tool_call.id = str(uuid.uuid4())
                        tool_call_list.append(tool_call)
                    response_message.tool_calls = tool_call_list

                messages.append(response_message.model_dump(exclude_none=True))
               
                content = response_message.content
                if (
                    content
                    and not response_message.tool_calls
                    and not response_message.function_call
                ):
                    final_text.append(content)

                    stop_flag = True
                else:
                    tool_calls = response_message.tool_calls
                    if not tool_calls:
                        logger.warning(
                            "Received empty response from LLM without content or tool calls."
                        )
                        break
                    for tool_call in tool_calls:
                        try:
                            # === 工具调用时间 ===
                            tool_start = time.perf_counter()
                            tool_name = tool_call.function.name
                            tool_args = json.loads(tool_call.function.arguments)
                            tool_id = tool_call.id
                            # There is only one server in our method
                            # We use mcp-copilot to route the servers
                            server_id = "mcp-copilot"
                            session = self.sessions[server_id]

                            logger.info(
                                f"LLM is calling tool: {tool_name}({tool_args})"
                            )          

                            result = await asyncio.wait_for(
                                session.call_tool(tool_name, tool_args), timeout=300
                            )
                            tool_end = time.perf_counter()
                            logger.info(f"MCP Tool {tool_name} execution time: {tool_end - tool_start:.3f}s")
                            #写入日志文件
                            with open("./test_yzx/time_log.txt", "a", encoding="utf-8") as f:
                                f.write(f"MCP Tool {tool_name} execution time: {tool_end - tool_start:.3f}s" + "\n")


                        except asyncio.TimeoutError:
                            logger.error(f"Tool call {tool_name} timed out.")
                            result = "Tool call timed out."
                            await self.cleanup_server("mcp-copilot")
                            await self.connect_copilot()
                            #写入日志文件
                            # with open("./test_yzx/selected_tools.txt", "a", encoding="utf-8") as f:
                            #     f.write(f"{task_index}.{tool_name}" + "\n")
                            # stop_flag = True
                            # break
                        except Exception as e:
                            logger.error(f"Error calling tool {tool_name}: {e}")
                            # error_traceback = traceback.format_exc()
                            # print(error_traceback)
                            result = f"Error: {str(e)}"
                            #写入日志文件
                            # with open("./test_yzx/selected_tools.txt", "a", encoding="utf-8") as f:
                            #     f.write(f"{task_index}.{tool_name}" + "\n")
                            # stop_flag = True
                            # break
                        result = str(result)
                        result = result[:max_tool_tokens]
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": str(result),
                            }
                        )
        except Exception as e:
            # error_traceback = traceback.format_exc()
            # print(error_traceback)
            logger.error(f"Error processing query '{query}': {e}")
            final_text.append(f"Error: {str(e)} ")
            messages.append({"role": "assistant", "content": str(e)})
            # 写入日志文件
            # with open("./test_yzx/selected_tools.txt", "a", encoding="utf-8") as f:
            #     f.write(f"{task_index}.error content: {str(e)}" + "\n")
        self.history = messages
        return "\n".join(final_text), messages


async def main(args):
    if not pathlib.Path(args.input_path).exists():
        logger.error(f"Input queries file {args.input_path} does not exist.")
        return
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"len(queries): {len(data)}")
    client = LoggingMCPClient()
    await client.connect_copilot()
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        exist_ids = {entry["task_id"] for entry in all_results}
    else:
        all_results = []
        exist_ids = set()
    error_queries = set()
    try:
        for idx, entry in tqdm(enumerate(data), total=len(data)):
            task_id = entry["task_id"]
            if task_id in exist_ids:
                continue
            query = entry["Question"]
            logger.info(f"{query}")
            # 从 entry 解析出答案工具
            answer_tools = parse_answer_tools(entry)
            #写入日志文件
            with open("./test_yzx/time_log.txt", "a", encoding="utf-8") as f:
                f.write(f"task {idx}" + "\n")
            try:
                query_start = time.perf_counter()
                response, messages = await client.process_query(query=query,
                                                                 answer_tools=answer_tools,
                                                                 tools_file=TOOLS_FILE,
                                                                 max_tools=args.max_tools,
                                                                 insert_number=args.insert_number,
                                                                 task_index=idx )
                                
                query_end = time.perf_counter()
                logger.info(f"Total query processing time: {query_end - query_start:.3f}s")
                #写入日志文件
                with open("./test_yzx/time_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"Total query processing time: {query_end - query_start:.3f}s" + "\n")

                logger.info(f"{response}")
                entry["response"] = response
                entry["messages"] = messages
                all_results.append(entry)

            except Exception:
                error_queries.add(query)
                logger.error(traceback.format_exc())
    finally:
        await client.cleanup()
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
