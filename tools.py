# tools.py

def calculator_tool(expression: str) -> str:
    """
    Evaluate a math expression using eval.
    WARNING: eval() can be dangerous with untrusted input.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


# Web Search Tool using googlesearch-python.
# Install it via: pip install googlesearch-python
try:
    from googlesearch import search
except ImportError:
    search = None

def search_tool(query: str) -> str:
    """
    Performs a web search and returns the top 3 result URLs.
    """
    if search is None:
        return "Web search tool is not available. Please install googlesearch-python."
    try:
        # Get top 3 results
        results = list(search(query, num_results=3))
        if not results:
            return "No results found."
        else:
            return "\n".join(results)
    except Exception as e:
        return f"Error during web search: {e}"
