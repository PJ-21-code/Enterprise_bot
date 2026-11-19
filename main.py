import numpy as np
import pandas as pd
import json
import chainlit as cl
from agent.master_agent import chat_with_enterprise_agent


def extract_response_text(response):
    """Extract clean user-facing text from an agent response.

    Handles the various shapes that AgentExecutor / RunnableWithMessageHistory
    can place in response['output']:
      1. str                        → returned as-is (normal RAG / SQL answers)
      2. dict with type='text'      → returns the 'text' value, dropping metadata
                                      like 'extras', 'signature', 'index'
      3. list of mixed str / dicts  → extracts text from each item and joins them
      4. dict without a 'text' key  → formatted as readable indented JSON
      5. any other type             → converted with str()
    """
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        if response.get("type") == "text" and "text" in response:
            return response["text"]
        return json.dumps(response, indent=2, default=str)

    if isinstance(response, list):
        parts = [extract_response_text(item) for item in response]
        return "\n\n".join(parts)

    return str(response)

@cl.on_chat_start
async def start_chat():
  # Assign a unique session identifier for Chainlit memory tracking
  cl.user_session.set("session_id", "enterprise_user_session_1")

  await cl.Message(
      content="Hi! I am your Enterprise Assistant. How can I help you"
      " today?"
  ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the agent, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        Response to user's query
    """

    user_query= message.content

    if user_query.lower() in ["exit", "quit", "bye"]:
       await cl.Message(
          content=(
            "Agent: Thanks for chatting, Goodbye! Feel free to start a new chat"
            " whenever you're ready."
        )
       ).send()
       return

    session_id= cl.user_session.get("session_id")

    msg= cl.Message(content="")
    await msg.send()

    try:
       agent_response = chat_with_enterprise_agent(user_query, session_id)

       msg.content = extract_response_text(agent_response)
       await msg.update()

    except ValueError as ve:
    
       msg.content = f"**Security Block:** {str(ve)}"
       await msg.update()

    except Exception as e:
       msg.content = f"An error occurred: {str(e)}"
       await msg.update()




