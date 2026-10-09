import json
import pandas as pd
import tiktoken
import gdown
import os
import torch
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_chroma import Chroma
from groq import Groq, RateLimitError
from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv
from langchain_core.tools import tool
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt

load_dotenv()

gemini_api_key= os.getenv("GEMINI_API_KEY")

client= genai.Client(api_key=gemini_api_key)

#loading documents
pdf_loader= PyPDFLoader("ey-worldwide-corporate-tax-guide.pdf") 
pages= pdf_loader.load()
full_text= " ".join([page.page_content for page in pages])

device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs= {"device":device})
text_splitter= SemanticChunker(embeddings=embeddings)

semantic_chunks= text_splitter.create_documents([full_text])

collection= 'Enterprise_info'
vectorstore_persisted = Chroma(
    collection_name=collection,
    persist_directory='./enterprise_vector_db',
    embedding_function=embeddings
)

vectorstore_persisted.add_documents(semantic_chunks)

retriever = vectorstore_persisted.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

system_message = """
You are an expert Tax Research Assistant. Your sole purpose is to answer corporate tax and compliance questions using *only* the context provided from the EY Corporate Tax Guide.

### Strict Guidelines:
1. **Rely Exclusively on Context:** Base your entire response strictly on the retrieved excerpts from the EY Corporate Tax Guide provided below. Do not use outside knowledge, assumptions, or generalized web data.
2. **Handle Missing Information:** If the answer cannot be found within the provided context, state clearly and professionally: *"Based on the provided EY Corporate Tax Guide, I could not find specific information regarding this query."* Do not attempt to guess or hallucinate tax rules.
3. **Cite Sources/Sections:** Whenever possible, reference the specific sections, chapters, or topics mentioned in the retrieved context to support your answer.
4. **Professional Tone:** Maintain a precise, objective, and professional tone appropriate for corporate tax advisors and auditors.

Context from EY Corporate Tax Guide:
{context}

User Question: {input}
"""

user_message_template= """ 
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Question
{question}

"""
def is_rate_limit_error(exception):
  if isinstance(exception, errors.APIError) and exception.code == 429:
    return True
  return False

@tool
@retry(
    retry= retry_if_exception_type(errors.APIError) & retry_if_exception_type(is_rate_limit_error),
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

      user_prompt = user_message_template.format(
        context=context_for_query, question=user_query
        )
    
      response= client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
               system_instruction=system_message,
               temperature=0,
            ),
        )

      predictions= response.text
      return predictions
    except errors.APIError as e:

        if e.code == 429:
          raise  
        else:
          predictions = (
          f"Sorry, I encountered an API error (Code {e.code}): {e.message}"
        )
          return predictions

    except Exception as e:
         predictions = f"Sorry, I encountered the following error: {str(e)}"
         return predictions
    

