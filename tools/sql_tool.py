from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from openai import RateLimitError
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

load_dotenv()

database_location= "\0"

db = SQLDatabase.from_uri(f"sqlite:///{database_location}") #create sql database

database_schema= db.get_table_info()

system_message= f"""{database_schema}"""

full_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("human", '{input}'),
        MessagesPlaceholder("agent_scratchpad")
    ]
)

openai_api_key= os.getenv("OPENAI_API_KEY")


llm= ChatOpenAI(
    model="gpt-4",
    api_key= openai_api_key,
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
