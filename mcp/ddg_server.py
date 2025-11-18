from mcp.server.fastmcp import FastMCP
from ddgs import DDGS
import json
import os

# Initialize FastMCP server
mcp = FastMCP("ddg-server")

@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.
    """
    results = []
    try:
        with DDGS() as ddgs:
            search_gen = ddgs.text(query, max_results=max_results)
            if search_gen:
                for r in search_gen:
                    results.append(r)
    except Exception as e:
        return f"Error performing search: {str(e)}"
            
    if not results:
        return "No results found."
        
    return json.dumps(results, indent=2)

@mcp.tool()
def save_report(filename: str, content: str) -> str:
    """
    Saves text content to a file in the current directory.
    """
    try:
        # Get absolute path to ensure we know where it saves
        cwd = os.getcwd()
        file_path = os.path.join(cwd, os.path.basename(filename))
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"SUCCESS: Saved {len(content)} chars to: {file_path}"
    except Exception as e:
        return f"ERROR saving file: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport='stdio')