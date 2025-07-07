

import os
import streamlit as st
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

st.title("🔍 Research Assistant (LangChain + OpenAI + FAISS)")

# Input URLs
urls = st.text_area("Enter URLs (one per line):").strip().split("\n")

# Path to save/load FAISS index
vectorstore_path = "faiss_index"

# Button to process URLs and create vectorstore
if st.button("Process URLs & Create VectorStore"):
    with st.spinner("Loading and indexing data..."):
        try:
            # Load documents
            loader = UnstructuredURLLoader(urls=urls)
            documents = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = text_splitter.split_documents(documents)

            # Create embeddings
            embeddings = OpenAIEmbeddings()

            # Create vectorstore
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Save vectorstore
            vectorstore.save_local(vectorstore_path)

            st.success("Vector store created and saved successfully.")
        except Exception as e:
            st.error(f"Error: {e}")

# Query section
query = st.text_input("Ask your research question:")

if st.button("Get Answer"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving answer..."):
            try:
                embeddings = OpenAIEmbeddings()

                # Load existing FAISS vectorstore
                vectorstore = FAISS.load_local(
                    vectorstore_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )

                # Create retrieval chain
                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=ChatOpenAI(temperature=0),
                    retriever=vectorstore.as_retriever()
                )

                # Run query
                result = chain({"question": query}, return_only_outputs=True)

                # Show result
                st.subheader("📌 Answer:")
                st.write(result["answer"])

                st.subheader("📎 Sources:")
                st.write(result["sources"])
            except Exception as e:
                st.error(f"Error: {e}")
