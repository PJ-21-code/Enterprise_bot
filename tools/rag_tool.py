import json
import pandas as pd
import tiktoken
import gdown
import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from groq import Groq
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

groq_api_key= os.getenv("GROQ_API_KEY")

client= Groq()

#loading documents
pdf_loader= PyPDFLoader("")
pages= pdf_loader.load()
full_text= " ".join([page.page_content for page in pages])


embeddings= HuggingFaceEmbeddings(model_name="all-Mini-L6-v2")
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
def rag_tool(user_query: str):

    """
      Generates a response based on the user's input by retrieving relevant documents and using them as context for a GPT-4o model.

      Args:
          user_input (str): The input question or query from the user.

      Returns:
          response (str): Return the generated response or an error message if an exception occurs.

    """
    relevant_document_chunks= retriever._get_relevant_documents(user_query)
    context_list= [data.page_content for data in relevant_document_chunks]
    context_for_query= ". ".join(context_list)

    prompt= [
       {"role":"system","content": system_message},
       {"role":"user","content":user_message_template.format(
           context= context_for_query,
           question= user_query
         )}
    ]
    try:
        response= client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=prompt,
            temperature=0
        )

        predictions= response.choices[0].message.content
    except Exception as e:
        predictions= f"Sorry I encounter the following error: {str(e)}"

    return predictions
    

