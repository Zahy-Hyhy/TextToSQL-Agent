# utils.py

def is_math_query(user_input: str) -> bool:
    """
    Returns True if the input is a math query (starts with "calc:").
    """
    return user_input.strip().lower().startswith("calc:")

def extract_expression(user_input: str) -> str:
    """
    Extracts the math expression from a query starting with "calc:".
    """
    return user_input.split("calc:", 1)[1].strip()

def is_search_query(user_input: str) -> bool:
    """
    Returns True if the input is a search query (starts with "search:").
    """
    return user_input.strip().lower().startswith("search:")

def extract_search_query(user_input: str) -> str:
    """
    Extracts the search query from input starting with "search:".
    """
    return user_input.split("search:", 1)[1].strip()
