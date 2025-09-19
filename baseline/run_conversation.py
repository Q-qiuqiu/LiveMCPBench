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
from draw import draw_from_log
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

def build_system_prompt(tools_file: str, 
                        answer_tools: List[str],
                        max_tools: Optional[int] = None, 
                        task_index:int=0,   
                        insert_number: int = 0) -> str:
    """
    把所有工具简介拼成系统 prompt，LLM 会通过 execute-tool 调用具体工具。
    如果 skip_tools 中包含的工具，就跳过注入，避免暴露“正确答案”。
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
    logger.info(f"Ground-truth tools: {answer_tools}")

    # 系统提示开头
    prompt_lines = [
               "You are an intelligent assistant designed to help users accomplish tasks using a set of MCP tools.\n",
        "Important rules:\n"
        "1. You have access to a single execution tool called 'execute-tool'. You must always use this tool to invoke any of the available MCP tools.\n"
        "2. Never call MCP tools directly. Always select the appropriate tool and call it via 'execute-tool'.\n"
        "3. Provide accurate and complete responses to the user.\n"
        "4. For each tool call, specify:\n"
        "   - server_name: the MCP server hosting the target tool\n"
        "   - tool_name: the name of the target tool\n"
        "   - params: a dictionary of input parameters for the tool\n\n"
        "Available MCP tools (choose from these, but always call via 'execute-tool'):\n"
    ]
    total_count = 0
    answers_number = len(answer_tools)
    inserted = False
    max_other_tools = None
    if max_tools is not None:
        max_other_tools = max_tools - answers_number
        if max_other_tools < 0:
            raise ValueError(
                f"max_tools={max_tools} 太小，不够容纳标准答案 {answers_number} 个"
            )
        
    for tool_entry in tools_data:
        for server_id, server in tool_entry.get("tools", {}).items():
            for tool in server.get("tools", []):
                tool_name = tool["name"]
                if tool_name in answer_tools:  # 过滤掉“正确答案”
                    logger.info(f"Skipping tool {tool_name} (marked as correct answer)")
                    continue

                # 插入标准答案工具
                if not inserted and total_count >= insert_number:
                    for i, ans_tool_name in enumerate(answer_tools):
                        if ans_tool_name in tools_map:
                            t = tools_map[ans_tool_name]
                            t_props = t.get("inputSchema", {}).get("properties", {})#防止格式异常，有些工具的inputSchema选项为空
                            prompt_lines.append(
                                f"- {ans_tool_name} (server: {t['server_id']}): {t['description']}. Input: {json.dumps(t_props)}"
                            )
                            #logger.info(f"insert tool {ans_tool_name} ")
                        else:
                            # 万一找不到对应工具，用占位
                            prompt_lines.append(f"- {ans_tool_name} (server: unknown): description missing. Input: {{}}")
                    inserted = True
                if max_other_tools is not None and total_count >= max_other_tools:
                    # 如果还没插入过答案，则现在必须插
                    if not inserted:
                        for ans_tool_name in answer_tools:
                            if ans_tool_name in tools_map:
                                t = tools_map[ans_tool_name]
                                prompt_lines.append(
                                    f"- {ans_tool_name} (server: {t['server_id']}): "
                                    f"{t['description']}. Input: {json.dumps(t['inputSchema']['properties'])}"
                                )
                                #logger.info(f"insert tool {ans_tool_name} ")
                            else:
                                prompt_lines.append(
                                    f"- {ans_tool_name} (server: unknown): description missing. Input: {{}}"
                                )
                        inserted = True
                    return "\n".join(prompt_lines)
            
                #插入备选工具
                input_props = tool.get("inputSchema", {}).get("properties", {})
                line = f"- {tool['name']} (server: {server_id}): {tool['description']}. Input: {json.dumps(input_props)}"

                prompt_lines.append(line)
                total_count += 1
                #logger.info(f"insert tool {tool_name} ")
                

                if max_tools is not None and total_count >= max_tools:
                    logger.info(f"Loaded {total_count} tools from {tools_file}")
                    return "\n".join(prompt_lines)
                
     # 如果正确答案未插入，警报
    if not inserted and answer_tools:
        logger.error(f"Please attention the answer position and the tools number!")

    logger.info(f"Loaded {total_count} tools from {tools_file}")
    return "\n".join(prompt_lines)
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
                    "content":build_system_prompt(
                        tools_file=tools_file,
                        answer_tools=answer_tools,
                        max_tools=max_tools,
                        insert_number=insert_number,
                        task_index=task_index
                    ),
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
                if tool.name != "execute-tool":
                    continue  # 跳过 route 或其他工具
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
            if idx<82:
                continue
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
                #绘图
                draw_from_log("./test_yzx/time_log.txt", args.max_tools, args.insert_number, idx)

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
