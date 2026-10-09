from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
import torch
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

load_dotenv()

database_location= "enterprise_finance.db"

db = SQLDatabase.from_uri(f"sqlite:///{database_location}") #create sql database

database_schema= db.get_table_info()

system_message = """
You are an expert SQL Assistant specialized in querying enterprise financial databases (such as the `transactions` table). Your primary goal is to translate user requests into precise, optimized, and read-only SQLite queries to retrieve accurate financial information.

### Database Schema Overview:
You have access to a table named `transactions` with the following columns:
- `transaction_id` (INTEGER, Primary Key)
- `date` (TEXT, format: YYYY-MM-DD)
- `customer_id` (INTEGER)
- `amount` (REAL, transaction monetary value)
- `type` (TEXT, e.g., 'debit', 'credit', 'transfer')
- `description` (TEXT, details of the transaction)

### Strict Guidelines:
1. **Read-Only Operations:** Only generate `SELECT` statements. Never generate or execute queries that modify data (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, etc.).
2. **Precision & Optimization:** Write clean, standard SQLite-compatible syntax. Use appropriate aggregations (`SUM`, `AVG`, `COUNT`) and filters (`WHERE`, `GROUP BY`, `ORDER BY`) as required by the user's prompt.
3. **Handle Empty Results:** If a query returns no results, state clearly that no matching records were found in the database rather than guessing or making up numbers.
4. **Data Privacy & Safety:** Do not expose sensitive database metadata or internal schema implementation details beyond what is required to answer the financial query.

User Request: {input}
database schema: {{database_schema}}
"""

full_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", '{input}'),
        MessagesPlaceholder("agent_scratchpad")
    ]
)

groq_api_key= os.getenv("GROQ_API_KEY")


llm= ChatGroq(
    model= 'llama-3.3-70b-versatile',
    api_key=groq_api_key,
    temperature=0
)

sql_agent= create_sql_agent(
    llm=llm,
    db=db,
    prompt= full_prompt,
    agent_type="openai-tools",
    agent_executor_kwargs={"handle_parsing_errors": True},
    max_iterations=5,
    verbose=True
)

# Wrapping sql agent into tool
@tool
@retry(
    retry= retry_if_exception_type(RateLimitError),
    wait= wait_exponential(multiplier=1,min=1,max=10),
    stop= stop_after_attempt(4)
)
def sql_tool(user_query):
    """
    Executes a SQL query using the sqlite_agent and returns the result.

    Args:
        user_query (str): a natural language query string explaining what information is required while also providing the necessary details to get the information.

    Returns:
        str: The result of the SQL query execution. If an error occurs, the exception is returned as a string.
    """
    try:
        response = sql_agent.invoke(user_query)

        prediction = response['output']

    except RateLimitError:
        raise    
    except Exception as e:
        prediction = e

    return prediction
