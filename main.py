import numpy as np
import pandas as pd
import json
import chainlit as cl
from agent.master_agent import chat_with_enterprise_agent

@cl.on_chat_start
async def start_chat():
  # Assign a unique session identifier for Chainlit memory tracking
  cl.user_session.set("session_id", "enterprise_user_session_1")

  await cl.Message(
      content="Hi! I am your Enterprise Forensic Assistant. How can I help you"
      " today?"
  ).send()


@cl.on_message
async def main(message: cl.message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the agent, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """

    user_query= message.content
    session_id= cl.user_session.get("session_id")

    msg= cl.Message(content="")
    await msg.send()

    try:
       agent_response = chat_with_enterprise_agent(user_query, session_id)

       msg.content = agent_response
       await msg.update()

    except ValueError as ve:
    
       msg.content = f"**Security Block:** {str(ve)}"
       await msg.update()

    except Exception as e:
       msg.content = f"An error occurred: {str(e)}"
       await msg.update()




