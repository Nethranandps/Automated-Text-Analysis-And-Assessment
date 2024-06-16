import os
import streamlit as st
import json
import time
import google.generativeai as genai

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially Gemini api key)
genai.configure(api_key="AIzaSyC43HmzcSp8AASUkqQBeOcJP9fJpSigSTM")

def process_urls(urls):
    generated_content = {}
    
    if urls:
        # load data
        loader = UnstructuredURLLoader(urls=urls)
        st.text("Data Loading...Started...✅✅✅")
        data = loader.load()
        # split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        st.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        
        # Generate content using Gemini API
        for i, doc in enumerate(docs):
            response = genai.GenerativeModel('gemini-pro').generate_content(f"Please summarise this document: {doc}")
            # Replace doc.get_text() with the correct attribute or method to access the text content
            content = get_text_content(doc)  # Placeholder method
            generated_content[f"Document {i+1}"] = {"content": content, "summary": response.text}

        # Save the generated content to a JSON file
        file_path = "gemini_generated_content.json"
        with open(file_path, "w") as json_file:
            json.dump(generated_content, json_file, default=str)  # Convert non-serializable types to strings
    
        st.text("Generated content saved to JSON file. ✅✅✅")
    else:
        st.warning("No URLs provided.")
    
    return generated_content




def chatbot(generated_content, question):
    if question:
        model = genai.GenerativeModel('gemini-pro')
        result = model.generate_content(question)
        return result.text

def main():
    st.title("RockyBot: News Research Tool 📈")
    st.sidebar.title("News Article URLs")

    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    process_url_clicked = st.sidebar.button("Process URLs")

    question = st.text_input("Ask a question based on the content: ")
    
    if process_url_clicked:
        generated_content = process_urls(urls)
    
    if generated_content:
        answer = chatbot(generated_content, question)
        if answer:
            st.text(f"Answer: {answer}")
        else:
            st.warning("No answer found for the question. Please try a different question.")

if __name__ == "__main__":
    main()
