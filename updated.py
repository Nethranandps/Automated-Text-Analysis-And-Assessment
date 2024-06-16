import os
import streamlit as st
import json
import time
# from google.generativeai import genai  # Assuming this import is not necessary for the modified chatbot function

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Assuming this line is not necessary for the modified chatbot function

def process_urls(urls):
    generated_content = {}
    
    if urls:
        # Save the URLs to a JSON file
        urls_data = {"urls": urls}
        urls_file_path = "urls_info.json"
        with open(urls_file_path, "w") as urls_file:
            json.dump(urls_data, urls_file, indent=4)

        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        st.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        st.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)
        
        # Generate content using Gemini API
        for i, doc in enumerate(docs):
            # response = genai.GenerativeModel('gemini-pro').generate_content(f"Please summarise this document: {doc}")  # Assuming this line is not necessary for the modified chatbot function
            # Placeholder method to access text content
            try:
                content = doc.get_text()  # Assuming 'get_text()' method exists
            except AttributeError:
                content = str(doc)  # Fallback: Convert doc to string
            generated_content[f"Document {i+1}"] = {"content": content, "summary": ""}

        # Save the generated content to a JSON file
        content_file_path = "gemini_generated_content.json"
        with open(content_file_path, "w") as json_file:
            json.dump(generated_content, json_file, default=str)  # Convert non-serializable types to strings
    
        st.text("Generated content saved to JSON file. ✅✅✅")
    else:
        st.warning("No URLs provided.")
    
    return generated_content

def chatbot(generated_content, question):
    if question:
        if generated_content:
            for key, value in generated_content.items():
                if "content" in value and value["content"]:
                    content = value["content"]
                    if question.lower() in content.lower():
                        return value["summary"]
        else:
            st.warning("No generated content available. Please process URLs first.")
    else:
        st.warning("Please input a question.")

    return None

def main():
    generated_content = {}  # Initialize generated_content here
    
    st.title("Summer-Riser 📈")
    st.sidebar.title("News Article URLs")

    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    process_url_clicked = st.sidebar.button("Process URLs")

    question = st.text_input("Ask a question based on the content: ")
    
    if process_url_clicked:
        generated_content = process_urls(urls)
    
    if generated_content:  # Now it's safe to reference generated_content
        answer = chatbot(generated_content, question)
        if answer:
            st.text(f"Answer: {answer}")
        else:
            st.warning("No answer found for the question. Please try a different question.")

if __name__ == "__main__":
    main()
