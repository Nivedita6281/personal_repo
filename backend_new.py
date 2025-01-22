import requests
import os
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

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
    try:
        vectordb = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Loaded existing vector store")
    except FileNotFoundError:
        vectordb = FAISS.from_texts([], embeddings)
        print("Created new vector store")

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
            for chunk in chunks:
                all_texts.append(chunk)
                all_metadatas.append({"source": url})
            print(f"Processed URL: {url} - Added {len(chunks)} chunks")
    if all_texts:
        vectordb.add_texts(all_texts, metadatas=all_metadatas)
        vectordb.save_local(VECTORDB_PATH)
        print(f"Total documents in vector store: {len(all_texts)}")
        print("Vector store saved successfully")
    else:
        print("No content to add to vectorstore")
    return vectordb

def load_vector_store():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(VECTORDB_PATH):
        vector_store = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully!")
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever
    else:
        raise FileNotFoundError("Vector store not found.")

if __name__ == "__main__":
    urls = [
        "https://help.ea.com/en/help/ea-sports-fc/transfer-and-convert-fc-points/",
        "https://help.ea.com/en/help/ea-sports-fc/rush/",
        "https://help.ea.com/en/help/ea-sports-fc/early-web-and-companion-app-start/"
    ]
    ingest_to_vectordb(urls)