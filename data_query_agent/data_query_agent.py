import json
import traceback
import sys
from collections import defaultdict
from datetime import datetime
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
from typing_extensions import Annotated, TypedDict
import psycopg2
from psycopg2.extras import RealDictCursor

# ------------------------------
# Database Configuration
# ------------------------------
DB_CONFIG = {
    'dbname': 'text2sql',
    'user': 'postgres',
    'password': 'admin',
    'host': 'localhost',
    'port': '5432'
}

# ------------------------------
# Ollama Model Configuration
# ------------------------------
OLLAMA_CONFIG = {
    'model': 'gemma3:12b',
    'endpoint': 'http://localhost:11434/'
}

# ------------------------------
# Data Fields Meaning
# ------------------------------
DATA_FIELDS_MEANING = {
"finance_economics_dataset": {
    "data_type": "PostgreSQL",
    "fields": {
        "date": "The timestamp of the stock",
        "stock_index": "The name of the stock market index being tracked.",
        "open_price": "The opening price of the stock or asset.",
        "close_price": "The closing price of the stock or asset.",
        "daily_high": "The highest price reached by the stock or asset.",
        "daily_low": "The lowest price reached by the stock or asset.",
        "trading_volume": "The total number of shares or contracts traded.",
        "gdp_growth": "The percentage growth of GDP.",
        "inflation_rate": "The rate of inflation.",
        "unemployment_rate": "The percentage of the labor force unemployed.",
        "interest_rate": "The central bank’s policy interest rate.",
        "consumer_confidence_index": "An indicator of consumer optimism.",
        "government_debt": "The total government debt as a percentage of GDP.",
        "corporate_profits": "Profits earned by corporations after taxes.",
        "forex_USD/EUR": "The exchange rate between USD and EUR.",
        "forex_USD/JPY": "The exchange rate between USD and JPY.",
        "crude_oil_price": "The price per barrel of crude oil.",
        "gold_price": "The price per ounce of gold.",
        "real_estate_index": "An index of real estate prices.",
        "retail_sales": "The total retail sales value.",
        "bankruptcy_rate": "The rate of bankruptcies.",
        "mergers_acquisitions_deals": "The number of M&A deals.",
        "venture_capital_funding": "The amount of VC funding.",
        "consumer_spending": "Total household expenditure."
    }
}}

# ------------------------------
# SQL Query Generation Prompt
# ------------------------------
SQL_QUERY_GEN_PROMPT = f"""
You are a PostgreSQL expert with a strong attention to detail.
You can define PostgreSQL queries, analyze queries results and interpret query results to respond with an answer.
NEVER query for all the columns from a specific table, only ask for the relevant columns given the question.
Today is {datetime.today()}
"""

# ------------------------------
# State Type Definition
# ------------------------------
class State(TypedDict):
    question: str
    steps: List[Dict[str, str]]
    relevant_data: List[Dict]

# ------------------------------
# SQL Output Type
# ------------------------------
class SQLOutput(TypedDict):
    query: str = Field(description="SQL Query")

# ------------------------------
# Database Helper Functions
# ------------------------------
def run_query(query: str) -> Dict:
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.commit()
        cursor.close()
        conn.close()
        return {'data': results, 'columns': columns, 'error': None}
    except Exception as e:
        return {'data': None, 'columns': [], 'error': str(e)}

def test_database_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute('SELECT 1;')
        cursor.fetchone()
        cursor.close()
        conn.close()
        print("Database connection successful.")
        return True
    except Exception as e:
        print(f"Database connection failed: {str(e)}")
        return False

def convert_to_markdown_table(result: Dict) -> str:
    if result.get('error'):
        return f"Error: {result['error']}"
    if not result.get('data'):
        return "Query ran successfully, no record found"

    data = result['data']
    columns = result['columns']
    if not columns:
        return "No columns returned"

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row.get(col, '')) for col in columns) + " |" for row in data]

    return "\n".join([header, separator] + rows)

# ------------------------------
# LLM Helper
# ------------------------------
def get_llm_model():
    llm = OllamaLLM(model=OLLAMA_CONFIG['model'], base_url=OLLAMA_CONFIG['endpoint'])
    return llm

# ------------------------------
# SQL Agent
# ------------------------------
def sql_gen_agent(state: State):
    print('Starting sql gen')
    relevant_data = []
    llm = get_llm_model()
    structured_llm = llm.with_structured_output(SQLOutput)

    for step in state['steps']:
        if step['data_name'] != "finance_economics_dataset":
            continue

        field_info = DATA_FIELDS_MEANING.get(step['data_name'])
        if not field_info:
            print(f"Warning: No schema defined for {step['data_name']}")
            continue

        system = f"""{SQL_QUERY_GEN_PROMPT}
Only query from table {step['data_name']} with these fields:
{json.dumps(field_info['fields'], indent=2)}. Limit your query to 50 rows.
""".replace('}', '}}').replace('{', '{{')

        human = f"Given the question: {state['question']}"

        sql_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}")
        ])

        sql_gen_model = sql_gen_prompt | structured_llm

        try:
            result = sql_gen_model.invoke({"question": human})
        except Exception as e:
            print("Structured output failed, fallback to raw:")
            traceback.print_exc()
            try:
                raw_sql = llm.invoke(sql_gen_prompt.format_messages({"question": human}))
                print("Raw SQL:", raw_sql)
                result = {"query": raw_sql}
            except Exception as raw_e:
                print("Raw fallback failed:")
                traceback.print_exc()
                result = None

        print("SQL Query:", result)

        if result and 'query' in result:
            query = result.get("query")
            raw_result = run_query(query)
            query_result = convert_to_markdown_table(raw_result)
            query_error = bool(raw_result.get('error'))
        else:
            query = ""
            query_result = "No query returned."
            query_error = True

        relevant_data.append({
            'query_result': query_result,
            'query_error': query_error,
            'query': query
        })

    return {'relevant_data': relevant_data}

# ------------------------------
# Terminal Interface
# ------------------------------
def main():
    if not test_database_connection():
        print("Please check your database settings. Exiting...")
        sys.exit(1)

    while True:
        print("\n=== Text-to-SQL Agent ===")
        print("Type 'exit' to quit")
        user_query = input("Query: ").strip()

        if user_query.lower() == 'exit':
            print("Exiting...")
            break

        if not user_query:
            print("Please enter a valid query.")
            continue

        state = {
            'question': user_query,
            'steps': [{'data_name': 'finance_economics_dataset'}],
            'relevant_data': []
        }

        try:
            result = sql_gen_agent(state)
            print("\n=== Results ===")
            for data in result['relevant_data']:
                print(f"\nQuery: {data['query']}")
                print("Result:")
                print(data['query_result'])
                if data['query_error']:
                    print("Error occurred during query execution.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
