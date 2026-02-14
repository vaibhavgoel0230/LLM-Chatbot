from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun(region="us-en")
except Exception:
    # Fallback if duckduckgo-search not installed: no-op search tool
    @tool
    def search_tool(query: str) -> str:
        """Search the web. Install duckduckgo-search and langchain-community for real search."""
        return "Search unavailable. Install: pip install duckduckgo-search langchain-community"

@tool
def calculator_tool(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Args:
        first_num: The first number.
        second_num: The second number.
        operation: The operation to perform. Can be "add", "subtract", "multiply", or "divide".
    Returns:
        A dictionary with the result of the operation.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "subtract":
            result = first_num - second_num
        elif operation == "multiply":
            result = first_num * second_num
        elif operation == "divide":
            if second_num == 0:
                return {"error": "Error: Division by zero"}
            result = first_num / second_num
        else:
            return {"error": f"unsupported operation: {operation}"}

        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": f"Error: {e}"}


client = MultiServerMCPClient(
    {
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)