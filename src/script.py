import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from pydantic import SecretStr
from typing import Any
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

GOOGLE_API_KEY: SecretStr

def load_envs():
    global GOOGLE_API_KEY
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

def generate_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, api_key=GOOGLE_API_KEY)

def generate_embedding_llm():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)

def load_pdf_files(path: str) -> list[Any]:
    docs = []

    for file in Path(path).glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(file))
            docs.extend(loader.load())
            print(f"File loaded successfully: {file.name}")
        except Exception as e:
           print(f"Error loading file {file.name}: {e}")

    print(f"Total documents loaded: {len(docs)}")

    return docs

def interactions_with_llm(llm):
    while True:
        user_input = input("")
        if user_input.lower() in ["exit", "quit"]:
            break

        resp_test = llm.invoke(user_input)
        print(resp_test.content)
        print("-" * 20)

def agent_interactions(retriever, document_chain):
    while True:
        user_input = input("")
        if user_input.lower() in ["exit", "quit"]:
            break

        context = retriever.invoke(user_input)
        answer = document_chain.invoke({"input": user_input, "context": context})
        print(answer)
        print("-" * 100)

def create_chucks_from_docs(docs) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    return splitter.split_documents(docs)

def create_document_chain(llm):
    prompt_rag = ChatPromptTemplate.from_messages([
        ("system",
         "You are an Internal Policies Assistant (HR/IT) at Carraro Desenvolvimento. "
         "Respond ONLY based on the provided context. "
         "If there is not enough information, just reply 'I don't know'."),
        ("human", "Ask: {input}\n\nContext:\n{context}")
    ])
    document_chain = create_stuff_documents_chain(llm, prompt_rag)
    return document_chain

def create_retriever(chunks, embedding_llm):
    vectorstore = FAISS.from_documents(chunks, embedding_llm)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4}
    )
    return retriever


def main():
    print("Starting IA agent...")
    print("")

    load_envs()

    llm = generate_llm()
    embedding_llm = generate_embedding_llm()

    docs = load_pdf_files("context-files/")
    chunks = create_chucks_from_docs(docs)
    retriever = create_retriever(chunks, embedding_llm)
    document_chain = create_document_chain(llm)

    print("")
    print("Chat started. Type 'exit' or 'quit' to stop.")
    print("Ask your questions:")

    agent_interactions(retriever, document_chain)

main()