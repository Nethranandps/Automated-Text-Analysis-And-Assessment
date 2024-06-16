import os
import streamlit as st
import pickle
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

load_dotenv()  # Load environment variables from .env file

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
faiss_file_path = "faiss_store_openai.pkl"
rag_model_name = "facebook/rag-token-nq"
rag_tokenizer = RagTokenizer.from_pretrained(rag_model_name)
rag_retriever = RagRetriever.from_pretrained(rag_model_name)
rag_model = RagTokenForGeneration.from_pretrained(rag_model_name)

main_placeholder = st.empty()

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(faiss_file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(faiss_file_path):
        with open(faiss_file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever_input_ids = rag_retriever.vectorize(text=query)
            generated = rag_model.generate(input_ids=retriever_input_ids, return_dict_in_generate=True)
            generated_summary = rag_tokenizer.decode(generated["output"][0], skip_special_tokens=True)
            st.header("Summary")
            st.write(generated_summary)
            
            # Chatbot functionality
            if st.button("Ask Summer-Riser"):
                answer = chatbot(generated_summary, query)
                if answer:
                    st.text(f"Summer-Riser: {answer}")
                else:
                    st.text("Summer-Riser: I'm sorry, I couldn't find an answer to your question.")
