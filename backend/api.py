import os
import joblib
import re
import tempfile
import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ====================== DATABASE SETUP ======================
def init_db():
    print("🗄️ Initializing SQLite Database...")
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/support_logs.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_email TEXT,
            review_text TEXT,
            sentiment TEXT,
            escalation_required BOOLEAN,
            generated_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ====================== API KEYS ======================
os.environ["GROQ_API_KEY"] = "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"
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ====================== ML & RAG SETUP ======================
ml_model = None
vectorizer = None
try:
    ml_model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
except Exception: pass

retriever = None
try:
    loader = PyPDFLoader("data/ecommerce_rag_document.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
except Exception: pass

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

class AgentState(TypedDict):
    review_text: str
    cleaned_text: str
    sentiment: str
    requires_escalation: bool
    retrieved_context: str
    final_email: str
    customer_email: str = None

def clean_text(text):
    return re.sub(r'[^a-z\s]', '', str(text).lower())

def analyzer_agent(state: AgentState):
    cleaned_lower = str(state['review_text']).lower()
    negative_keywords = ["bad", "refund", "broken", "issue", "late", "delay", "frustrating", "smell", "ozone", "ticking", "angry", "terrible", "worst", "nahi aaya", "gussa", "wapas", "masla", "kharab", "paisa", "bekar", "bakwas", "fati", "toota", "kya hai yeh", "fraud", "maslay"]
    
    sentiment = "Positive"
    escalation = False
    
    if any(w in cleaned_lower for w in negative_keywords):
        sentiment = "Negative"
        escalation = True
    elif ml_model and vectorizer:
        try:
            vec = vectorizer.transform([clean_text(cleaned_lower)])
            pred = ml_model.predict(vec)[0]
            if int(pred) == 0:
                sentiment = "Negative"
                escalation = True
        except: pass
            
    context = "Standard corporate policy applies."
    if retriever:
        try:
            docs = retriever.invoke(state['review_text'])
            context = "\n".join([d.page_content for d in docs])
        except: pass
    
    return {"cleaned_text": cleaned_lower, "sentiment": sentiment, "requires_escalation": escalation, "retrieved_context": context}

def apology_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_template("You are a strict but empathetic Customer Support Agent representing the company.\nNEVER impersonate the customer. Do not thank the customer for a delay or damage.\nWrite a professional apology email (4-6 sentences max) addressing their issue.\nUse the Policy Summary below to provide the correct solution (e.g., refund windows, safety escalations).\n\nPolicy Summary: {context}\n\nCustomer Review to address: \"{review_text}\"\nSign off as \"Customer Support Team\".")
    response = (prompt | llm).invoke({"context": state.get("retrieved_context", ""), "review_text": state['review_text']})
    return {"final_email": response.content.strip()}

def gratitude_agent(state: AgentState):
    prompt = ChatPromptTemplate.from_template("You are a warm Customer Support Agent representing the company. \nNEVER impersonate the customer.\nWrite a short, professional thank you email (3-4 sentences) appreciating their positive feedback.\n\nCustomer Review: \"{review_text}\"\nSign off as \"Customer Support Team\".")
    response = (prompt | llm).invoke({"review_text": state['review_text']})
    return {"final_email": response.content.strip()}

def route_review(state: AgentState):
    return "apology" if state["requires_escalation"] else "gratitude"

workflow = StateGraph(AgentState)
workflow.add_node("analyzer", analyzer_agent)
workflow.add_node("apology", apology_agent)
workflow.add_node("gratitude", gratitude_agent)
workflow.set_entry_point("analyzer")
workflow.add_conditional_edges("analyzer", route_review)
workflow.add_edge("apology", END)
workflow.add_edge("gratitude", END)
app_graph = workflow.compile()

app = FastAPI(title="Hybrid ML + RAG Enterprise API")

class ReviewRequest(BaseModel):
    text: str
    customer_email: str = None

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name
        with open(temp_audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(file=(temp_audio_path, audio_file.read()), model="whisper-large-v3", response_format="json")
        os.remove(temp_audio_path)
        return {"status": "success", "text": transcription.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze_review(request: ReviewRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty review text provided.")
    
    state = {"review_text": request.text, "customer_email": request.customer_email}
    result = app_graph.invoke(state)
    
    # 🔴 NEW: Save to Database
    try:
        conn = sqlite3.connect("data/support_logs.db")
        c = conn.cursor()
        c.execute('''
            INSERT INTO tickets (customer_email, review_text, sentiment, escalation_required, generated_response)
            VALUES (?, ?, ?, ?, ?)
        ''', (request.customer_email, request.text, result["sentiment"], bool(result["requires_escalation"]), result["final_email"]))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

    return {
        "status": "success",
        "detected_sentiment": result["sentiment"],
        "escalation_required": bool(result["requires_escalation"]),
        "generated_response": result["final_email"]
    }

# 🔴 NEW: Analytics Endpoint for Dashboard
@app.get("/analytics")
def get_analytics():
    try:
        conn = sqlite3.connect("data/support_logs.db")
        c = conn.cursor()
        
        c.execute("SELECT sentiment, COUNT(*) FROM tickets GROUP BY sentiment")
        sentiment_counts = dict(c.fetchall())
        
        c.execute("SELECT escalation_required, COUNT(*) FROM tickets GROUP BY escalation_required")
        escalation_counts = dict(c.fetchall())
        
        c.execute("SELECT customer_email, sentiment, timestamp FROM tickets ORDER BY timestamp DESC LIMIT 5")
        recent_tickets = [{"email": row[0], "sentiment": row[1], "time": row[2]} for row in c.fetchall()]
        
        conn.close()
        
        return {
            "total_tickets": sum(sentiment_counts.values()),
            "sentiment_distribution": sentiment_counts,
            "escalations": escalation_counts.get(1, 0),
            "recent_tickets": recent_tickets
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)