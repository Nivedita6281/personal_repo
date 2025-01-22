from fastapi import FastAPI
from pydantic import BaseModel
from backend_new import load_vector_store
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import random

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

    # Select the most relevant document (e.g., the first one or based on metadata score)
    source_urls = []
    if docs:
        # Assuming the first document is the most relevant, you can use any other criteria if available
        source = docs[0].metadata.get("source")
        if source and source.startswith("http"):
            source_urls.append(source)

    # Run the QA chain
    answer = qa_chain.run(input_documents=docs, question=question)

    follow_up = random.choice(follow_up_messages)
    return {
        "question": question,
        "answer": answer,
        "sources": source_urls,  # Only one source now
        "follow_up": follow_up
    }
@app.get("/")
def get_welcome_message():
    return {"message": "Welcome to the chatbot! How can I assist you today?"}