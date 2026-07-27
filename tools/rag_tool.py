import json
import pandas as pd
import tiktoken
import gdown
import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_chroma import Chroma
from groq import Groq, RateLimitError, log_groq_retry
from dotenv import load_dotenv
from langchain_core.tools import tool
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

load_dotenv()

groq_api_key= os.getenv("GROQ_API_KEY")

client= Groq()

#loading documents
pdf_loader= PyPDFLoader("iso27001.pdf") #used for rag tool testing
pages= pdf_loader.load()
full_text= " ".join([page.page_content for page in pages])


embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter= SemanticChunker(embeddings=embeddings)

semantic_chunks= text_splitter.create_documents([full_text])

collection= 'Enterprise_info'
vectorstore_persisted = Chroma(
    collection_name=collection,
    persist_directory='./enterprise_db',
    embedding_function=embeddings
)

vectorstore_persisted.add_documents(semantic_chunks)

retriever = vectorstore_persisted.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

system_message= """""" #to be filled

user_message_template= """ 
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Question
{question}

"""


@tool
@retry(
    retry= retry_if_exception_type(RateLimitError),
    wait= wait_exponential(multiplier=1,min=1,max=10),
    stop= stop_after_attempt(4)
)
def rag_tool(user_query: str):

    """
      Generates a response based on the user's input by retrieving relevant documents and using them as context for a llama-3.3-70b-versaltile model.

      Args:
          user_query (str): The input question or query from the user.

      Returns:
          response (str): Return the generated response or an error message if an exception occurs.

    """

    try:
      relevant_document_chunks= retriever.invoke(user_query)
      context_list= [data.page_content for data in relevant_document_chunks]
      context_for_query= ". ".join(context_list)

      prompt= [
         {"role":"system","content": system_message},
         {"role":"user","content":user_message_template.format(
           context= context_for_query,
           question= user_query
           )}
      ]
    
      response= client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=prompt,
            temperature=0
        )

      predictions= response.choices[0].message.content
      return predictions
    except RateLimitError:
       raise
    except Exception as e:
        predictions= f"Sorry I encounter the following error: {str(e)}"

        return predictions
    

