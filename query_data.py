# query_data.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

# load env
load_dotenv()

CHROMA_PATH = "chroma"

def setup_qa_chain():
    # Set up the embedding function
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the Chroma database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # retriever will return the top 2 results that most relevant to the query.
    retriever = db.as_retriever(search_kwargs={"k": 2})

    # Set up the Hugging Face model
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        # control the randomness of output
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )

    prompt_template = """Use the following retrieved context information and your general knowledge to answer the question. If the question cannot be answered solely based on the context information, use your knowledge to reason and explain.
    If the context information is irrelevant or insufficient to answer the question, primarily rely on your knowledge to provide a comprehensive and accurate answer.

    Context information:
    {context}

    Question: {question}

    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # combine all relevant information into one context
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain
