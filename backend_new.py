import os
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import json
import pickle

load_dotenv()

VECTORDB_PATH = "faiss_index"

def fetch_content_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        content = soup.get_text(separator=' ', strip=True)
        content = ' '.join(content.split())
        return content, {"source": url}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return None, None

def ingest_to_vectordb(urls):
    embeddings = OpenAIEmbeddings()

    os.makedirs(VECTORDB_PATH, exist_ok=True)

    all_texts = []
    all_metadatas = []

    for url in urls:
        content, metadata = fetch_content_from_url(url)
        if content:
            words = content.split()
            chunk_size = 500
            overlap = 50
            chunks = []

            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if len(chunk.split()) >= 50:
                    chunks.append(chunk)

            all_texts.extend(chunks)
            all_metadatas.extend([{"source": url}] * len(chunks))

            print(f"Processed URL: {url} - Added {len(chunks)} chunks")

    if all_texts:
        vectordb = FAISS.from_texts(all_texts, embeddings)
        
        # Save index and metadata separately
        vectordb.save_local(VECTORDB_PATH)
        print(f"Total documents in vector store: {len(all_texts)}")
        print("Vector store saved successfully")
    else:
        print("No content to add to vectorstore")

def load_vector_store():
    embeddings = OpenAIEmbeddings()
    
    if not os.path.exists(VECTORDB_PATH):
        raise FileNotFoundError(f"Vector store not found at {VECTORDB_PATH}")
    
    try:
        # Adjusted load method
        vector_store = FAISS.load_local(
            VECTORDB_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

if __name__ == "__main__":
    urls = [
        "https://help.ea.com/en/help/ea-sports-fc/transfer-and-convert-fc-points/",
        "https://help.ea.com/en/help/ea-sports-fc/rush/",
        "https://help.ea.com/en/help/ea-sports-fc/early-web-and-companion-app-start/"
    ]
    ingest_to_vectordb(urls)