from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend_new import load_vector_store
from langchain.chains import RetrievalQA
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

# Load the retriever
retriever = load_vector_store()

# Initialize QA chain if retriever is not None
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=False
) if retriever else None

class QuestionRequest(BaseModel):
    question: str

@app.post("/qa")
def post_answer(request: QuestionRequest):
    if retriever is None:
        raise HTTPException(status_code=404, detail="Vector store not initialized.")
    
    question = request.question

    try:
        # Run the QA chain
        answer = qa_chain.run(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running QA chain: {str(e)}")

    follow_up = random.choice(follow_up_messages)
    return {
        "question": question,
        "answer": answer,
        "follow_up": follow_up
    }

@app.get("/")
def get_welcome_message():
    return {"message": "Welcome to the chatbot! How can I assist you today?"}
