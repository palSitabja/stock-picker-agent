from langchain_mcp_adapters.client import MultiServerMCPClient

class McpClient(MultiServerMCPClient):
    def __init__(self):
        super().__init__(
            {
                "alphavantage": {
                    "url": "https://mcp.alphavantage.co/mcp?apikey=API_KEY",
                    "transport": "streamable_http",
                }
            }
        )