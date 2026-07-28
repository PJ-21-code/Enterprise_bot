from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from tools.sql_tool import sql_tool
from tools.rag_tool import rag_tool
from tools.fraud_detector_tools import calculate_variance, benford_finance_fraud_detector, detect_split_transactions
from langchain_groq import ChatGroq
import os
from detoxify import Detoxify
from dotenv import load_dotenv

load_dotenv()

groq_api_key= os.getenv("GROQ_API_KEY")
system_message= """"""  #to be filled

prompt= ChatPromptTemplate.from_messages([
    ("system" ,system_message),
    ("human","{input}"),
    ("placeholder","{agent_scratchpad}"),
])

llm= ChatGroq(
    model= 'llama-3.3-70b-versatile',
    api_key=groq_api_key,
    temperature=0
)

tools= [rag_tool, sql_tool,calculate_variance,benford_finance_fraud_detector,detect_split_transactions]
agent= create_tool_calling_agent(llm=llm,prompt=prompt,tools=tools)
agent_execution= AgentExecutor(agent=agent,tools=tools,verbose=True)

store={}

def session_history(session_id: str):
    if session_id not in store:
        store[session_id]= InMemoryChatMessageHistory()
    return store[session_id]

agent_with_chat_history= RunnableWithMessageHistory(
    agent_execution,
    session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

def chat_with_enterprise_agent():

    print("Chat with our agent! Type 'exit' to end conversation")

    print("Hi! How can I help you?")

    model= Detoxify('original')

    session_config = {"configurable": {"session_id": "session_1"}}

    while True:

        user_query= input("You: ")

        score= model.predict(user_query)
        if score["toxicity"]> 0.5:
            print("Sorry, I cannot help you with this kind of behaviour")
            break

        if user_query.lower()== 'exit':
            print("Agent: Thanks for chatting, Goodbye!")
            break

        

        response = agent_with_chat_history.invoke(
        {"input": user_query}, config=session_config
        )

        agent_response= response['output']
        print(f"Agent: {agent_response}")


chat_with_enterprise_agent()        