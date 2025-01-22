from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from backend_new import load_vector_store
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import random
import os
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract

app = FastAPI()

follow_up_messages = [
    "Feel free to challenge me with your next question!",
    "What else can I assist you with today?",
    "Does this help you enjoy Rush even more?",
    "Looking for ways to level up your Rush skills?",
    "I'm here to help what's next on your mind?",
    "Would you like to explore more about Rush?",
    "Curious about how to play with friends or unlock special features?"
]

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff", verbose=False)

class QuestionRequest(BaseModel):
    question: str

try:
    retriever = load_vector_store()
except FileNotFoundError:
    print("Vector store not found. Please run the script with `if __name__ == '__main__':` first to create it.")
    retriever = None

@app.post("/qa")
def post_answer(request: QuestionRequest):
    if retriever is None:
        return {"error": "Vector store not initialized."}

    question = request.question
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return {
            "question": question,
            "answer": "I'm sorry, I couldn't find relevant information.",
            "sources": [],
            "follow_up": random.choice(follow_up_messages)
        }

    source_urls = []
    if docs:
        source = docs[0].metadata.get("source")
        if source and source.startswith("http"):
            source_urls.append(source)

    answer = qa_chain.run(input_documents=docs, question=question)

    follow_up = random.choice(follow_up_messages)
    return {
        "question": question,
        "answer": answer,
        "sources": source_urls,
        "follow_up": follow_up
    }

@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        content = ""
        if file.content_type == "application/pdf":
            # Process PDF
            pdf_reader = PdfReader(file.file)
            for page in pdf_reader.pages:
                content += page.extract_text()
        elif file.content_type.startswith("image/"):
            # Process Image
            image = Image.open(file.file)
            content = pytesseract.image_to_string(image)
        else:
            return {"error": "Unsupported file type. Please upload PDFs or images."}

        # Add content to FAISS vector store
        if content.strip():
            embeddings = OpenAIEmbeddings()
            try:
                vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            except FileNotFoundError:
                vectordb = FAISS.from_texts([], embeddings)

            vectordb.add_texts([content], metadatas=[{"source": file.filename}])
            vectordb.save_local("faiss_index")

            return {"message": f"File '{file.filename}' uploaded and processed successfully!"}
        else:
            return {"error": "No text content found in the file."}
    except Exception as e:
        return {"error": f"An error occurred while processing the file: {e}"}

@app.get("/")
def get_welcome_message():
    return {"message": "Welcome to the chatbot! You can now upload PDFs and images for processing."}
