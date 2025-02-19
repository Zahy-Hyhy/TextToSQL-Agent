# main.py

from model import get_chat_pipeline
from tools import calculator_tool, search_tool
from utils import is_math_query, extract_expression, is_search_query, extract_search_query

def main():
    # Load the Deepseek pipeline
    chat_pipeline = get_chat_pipeline()
    
    print("Welcome to the Deepseek Chatbot!")
    print("Type 'exit' or 'quit' to end.")
    print("For math queries, type: calc: <expression>")
    print("For web search queries, type: search: <query>")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break
        
        if is_math_query(user_input):
            expression = extract_expression(user_input)
            result = calculator_tool(expression)
            print("Bot (Calculator):", result)
        elif is_search_query(user_input):
            query = extract_search_query(user_input)
            result = search_tool(query)
            print("Bot (Web Search):", result)
        else:
            response = chat_pipeline(user_input, max_length=150, do_sample=True)
            bot_reply = response[0]["generated_text"]
            print("Bot (Deepseek):", bot_reply)

if __name__ == "__main__":
    main()
