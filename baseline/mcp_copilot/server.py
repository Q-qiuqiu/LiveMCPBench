from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
import asyncio
import mcp.types as types
from mcp.server.fastmcp import Context, FastMCP
from baseline.mcp_copilot.router import Router, dump_to_yaml
from baseline.mcp_copilot.arg_generation import run_generation


def serve(config: dict[str, Any] | Path = Router._default_config_path) -> None:
    """Run the copilot MCP server.

    Args:
        config: MCP Server config for Router
    """
    print("Indexing MCP servers and tools...")
    asyncio.run(run_generation())

    @asynccontextmanager
    async def copilot_lifespan(server: FastMCP) -> AsyncIterator[dict]:
        """Lifespan context manager for the Copilot server."""
        async with Router(config) as router:
            yield {"router": router}

    print("Starting MCP Copilot server...")
    server = FastMCP("mcp-copilot", lifespan=copilot_lifespan)

    @server.tool(
        name="route",
        description=(
            """
    Tool Discovery Tool — used to find tools that can fulfill user needs.You can call this tool to 'search for the right tool'. Its purpose is to help you enter the 'Discovery Phase' before executing a task, by describing the desired functionality to retrieve and match the most suitable tools.",
    """
        ),
    )
    async def route(
        query: str,
        ctx: Context,
    ) -> types.CallToolResult:
        """Route user query to appropriate servers and tools."""
        router: Router = ctx.request_context.lifespan_context["router"]
        result = await router.route(query)
        return dump_to_yaml(result)

    @server.tool(
        name="execute-tool",
        description="""A tool for executing a specific tool on a specific server.Select tools only from the results obtained from the previous route each time.

When to use this tool:
    - When using the route tool to route to a specific MCP server and tool
    - When the 'execute-tool' fails to execute (up to 3 repetitions).
    - When the user's needs and previous needs require the same tool.

Parameters explained:
    -server_name: string, required. The name of the server where the target tool is located.

    -tool_name: string, required. The name of the target tool to be executed.

    -params: dictionary or None, optional. A dictionary containing all parameters that need to be passed to the target tool. This can be omitted if the target tool does not require parameters.
""",
    )
    async def execute_tool(
        server_name: str,
        tool_name: str,
        params: dict[str, Any] | None,
        ctx: Context,
    ) -> types.CallToolResult:
        """Execute the specific tool based on routed servers or tools."""
        router = ctx.request_context.lifespan_context["router"]
        result = await router.call_tool(server_name, tool_name, params)

        return result

    server.run(transport="stdio")
