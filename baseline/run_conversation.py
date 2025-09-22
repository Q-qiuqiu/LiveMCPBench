import asyncio
import json
import logging
import os
import pathlib
import traceback
from typing import List, Optional, Tuple
import pickle
import pathlib
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
    parser.add_argument(
        "--top_tools",
        type=int,
        default=5,
        help="Maximum number of rag",
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


def extract_tools(response) -> list:
    """
    从 response 中解析 matched_tools，提取工具名并返回列表
    """
    # 获取 text 内容
    if hasattr(response, "content") and len(response.content) > 0:
        text = response.content[0].text
    else:
        raise ValueError("CallToolResult 对象中没有 content 或为空") 
    #提取 matched_tools 部分
    if "matched_tools:" not in text:
        raise ValueError("response.text 中没有找到 'matched_tools:' 段落")
        
    matched_tools_yaml = "\n".join(text.split("matched_tools:")[1].splitlines())

    # 加载为 Python 对象
    tools_list = yaml.safe_load(matched_tools_yaml)

    # 提取 tool_name
    rag_tools = [tool["tool_name"] for tool in tools_list if "tool_name" in tool]
    return rag_tools

def log_event(logs: list,event_type: str, **kwargs):
    """
    添加一条日志事件到 all_logs
    """
    log_entry = {"type": event_type, **kwargs}
    logs.append(log_entry)

        
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
        top_tools: Optional[int] = None,
        insert_number: int = 0,
        task_index: int = 0,
        history: Optional[list] = None,
        max_tool_tokens: int = 10000,
        task_logs: Optional[list] = None,  
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
                # 写入日志文件
                #log_event(task_logs,"request_payload", payload=request_payload)

                response = self.chat_model.complete_with_retry(**request_payload)
                # 写入日志文件
                log_event(task_logs,"llm_response", response=response.model_dump())

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
                            # 写入日志文件
                            log_event(task_logs,"tool_call", tool_name=tool_name, tool_args=tool_args)

                            result = await asyncio.wait_for(
                                session.call_tool(tool_name, tool_args), timeout=300
                            )
                            if tool_name=='route':
                                rag_tools=extract_tools(result)
                                log_event(task_logs,"rag_select_tools", tools=rag_tools)      
                            else:
                                # 写入日志文件
                                log_event(task_logs,"tool_result", tool_name=tool_name, result=str(result))

                        except asyncio.TimeoutError:
                            logger.error(f"Tool call {tool_name} timed out.")
                            result = "Tool call timed out."
                            # 写入日志文件
                            log_event(task_logs,"tool_call_time_out",tool_name=tool_name)

                            await self.cleanup_server("mcp-copilot")
                            await self.connect_copilot()
                        except Exception as e:
                            logger.error(f"Error calling tool {tool_name}: {e}")
                            result = f"Error: {str(e)}"
                            # 写入日志文件
                            log_event(task_logs,"error_calling_tool",tool_name=tool_name,error=str(e))
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
            logger.error(f"Error processing query '{query}': {e}")
            final_text.append(f"Error: {str(e)} ")
            messages.append({"role": "assistant", "content": str(e)})
            # 写入日志文件
            log_event(task_logs,"error_processing_query", error=str(e))
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
    all_logs = []   # 存放所有任务的日志
    try:
        for idx, entry in tqdm(enumerate(data), total=len(data)):
            task_id = entry["task_id"]
            if task_id in exist_ids:
                continue
            query = entry["Question"]
            logger.info(f"{query}")
            # 从 entry 解析出答案工具
            answer_tools = parse_answer_tools(entry)
            # 为当前任务单独建一个日志列表
            task_logs = []
            # 写入日志文件
            log_event(task_logs,"query", idx=idx, query=query, ground_truth=answer_tools)

            try:
                response, messages = await client.process_query(query=query,
                                                                 answer_tools=answer_tools,
                                                                 tools_file=TOOLS_FILE,
                                                                 max_tools=args.max_tools,
                                                                 top_tools=args.top_tools,
                                                                 insert_number=args.insert_number,
                                                                 task_index=idx,
                                                                 task_logs=task_logs)
                logger.info(f"{response}")
                entry["response"] = response
                entry["messages"] = messages
                all_results.append(entry)
                # 写入日志文件
                log_event(task_logs, "final_response", response=response)

            except Exception:
                error_queries.add(query)
                logger.error(traceback.format_exc())
            # 当前任务日志打包
            all_logs.append({
                "task_number": idx,
                "logs": task_logs
            })
    finally:
        await client.cleanup()
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        # 保存日志（带格式缩进）
        with open("./test_yzx/rag_gt.json", "w", encoding="utf-8") as f:
            json.dump(all_logs, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))

