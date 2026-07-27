from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from tools.sql_tool import sql_tool
from tools.rag_tool import rag_tool
from tools.fraud_detector_tools import calculate_variance, benford_finance_fraud_detector, second_fraud_detector
from langchain_groq import ChatGroq
import os
import pypdf
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

tools= [rag_tool]
agent= create_tool_calling_agent(llm=llm,prompt=prompt,tools=tools)
agent_execution= AgentExecutor(agent=agent,tools=tools,verbose=True)

def chat_with_enterprise_agent():

    print("Chat with our agent! Type 'exit' to end conversation")

    conversation_history=[{"role":"assistant", "content":"Hey! How can i help you?"}]

    print("Hi! How can I help you?")

    while True:

        user_query= input("You: ")
        if user_query.lower()== 'exit':
            print("Agent: Thanks for chatting, Goodbye!")
            break

        conversation_history.append({"role":"user","content": user_query})

        conversation_input = "\n".join(
            [f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation_history]
        )

        response= agent_execution.invoke({"input": conversation_input})
        agent_response= response['output']

        conversation_history.append({"role": "assistant", "content": agent_response})

        print(f"Agent: {agent_response}")


chat_with_enterprise_agent()        