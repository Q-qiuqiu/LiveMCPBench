import asyncio
import json
import logging
import os
import pathlib
import traceback
from typing import List, Optional, Tuple

import dotenv
from mcp import ClientSession
from tqdm import tqdm
import argparse
import uuid

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
        "--max_tools",
        type=int,
        default=None,
        help="Maximum number of tools to load from tools.json. If None, load all.",
)
    return parser.parse_args()

def build_system_prompt(tools_file: str, max_tools: Optional[int] = None) -> str:
    """把所有工具简介拼成系统 prompt，LLM 会通过 execute 调用具体工具"""
    with open(tools_file, "r", encoding="utf-8") as f:
        tools_data = json.load(f)

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
    total_count = 0
    for tool_entry in tools_data:
        for server_id, server in tool_entry.get("tools", {}).items():
            for tool in server.get("tools", []):

                line = f"- {tool['name']} (server: {server_id}): {tool['description']}. Input: {json.dumps(tool['inputSchema']['properties'])}"
                prompt_lines.append(line)
                total_count += 1
                if max_tools is not None and total_count >= max_tools:
                    #prompt_lines.append(f"...and {len(tools_data) - total_count} more tools truncated")
                    logger.info(f"Loaded {total_count} tools from {tools_file}")
                    return "\n".join(prompt_lines)
    logger.info(f"Loaded {total_count} tools from {tools_file}")
    return "\n".join(prompt_lines)


class LoggingMCPClient(MCPClient):
    def __init__(self,tools_file: str = TOOLS_FILE,max_tools: Optional[int] = None):
        super().__init__(timeout=180, max_sessions=9999)
        self.chat_model = ChatModel(
            model_name=os.getenv("MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model_url=os.getenv("BASE_URL"),
        )
        self.system_prompt  = build_system_prompt(tools_file, max_tools)

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
        history: Optional[list] = None,
        max_tool_tokens: int = 10000,
    ) -> Tuple[str, List[dict]]:
        if history is None:
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
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
            while True:
                request_payload = {
                    "messages": messages,
                    "tools": available_tools,
                }


                response = self.chat_model.complete_with_retry(**request_payload)
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
                            # timeout

                            result = await asyncio.wait_for(
                                session.call_tool(tool_name, tool_args), timeout=300 #调用的是谁的call_tool
                            )


                        except asyncio.TimeoutError:
                            logger.error(f"Tool call {tool_name} timed out.")
                            result = "Tool call timed out."
                            await self.cleanup_server("mcp-copilot")
                            await self.connect_copilot()
                        except Exception as e:
                            logger.error(f"Error calling tool {tool_name}: {e}")
                            result = f"Error: {str(e)}"
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
            final_text.append(f"Error: {str(e)}")
            messages.append({"role": "assistant", "content": str(e)})
        self.history = messages
        return "\n".join(final_text), messages


async def main(args):
    if not pathlib.Path(args.input_path).exists():
        logger.error(f"Input queries file {args.input_path} does not exist.")
        return
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"len(queries): {len(data)}")
    client = LoggingMCPClient(args.tools_path,args.max_tools)
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
        for entry in tqdm(data):
            task_id = entry["task_id"]
            if task_id in exist_ids:
                continue
            query = entry["Question"]
            logger.info(f"{query}")
            try:
                response, messages = await client.process_query(query, None)
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
