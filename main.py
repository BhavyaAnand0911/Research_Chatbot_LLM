import os
import streamlit as st
import pickle
import time
from langchain_community.llms.openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

directory = os.path.dirname(file_path)
if directory and not os.path.exists(directory):
    os.makedirs(directory)

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    try:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()
    except Exception as e:
        st.error(f"Error loading data from URLs: {e}")
        data = []

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    try:
        with open(file_path, "w") as f:
            pickle.dump(vectorstore_openai, f)
    except Exception as e:
        st.error(f"Error saving FAISS index: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                vectorstore = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
            vectorstore = None
    else:
        vectorstore = None

    if vectorstore:
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            # Split the sources by newline
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
