import os
import schedule
import time
from datetime import datetime
from typing import TypedDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

# ====================== CONFIG ======================
os.environ["GROQ_API_KEY"] = "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"

# ====================== RAG SETUP ======================
print("📚 Loading RAG Policy Data...")
PDF_PATH = "attachments/ecommerce_rag_document.pdf"

retriever = None
try:
    if os.path.exists(PDF_PATH):
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        print("✅ RAG Knowledge Base Synced!")
    else:
        print(f"⚠️ PDF document missing at {PDF_PATH}. Operating in non-RAG mode.")
except Exception as e:
    print(f"RAG Engine Exception: {e}")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

class State(TypedDict):
    review_text: str
    customer_email: str
    sentiment: str
    requires_escalation: bool
    policy_context: str
    final_response: str

# ====================== AGENTS ======================
def analyzer(state: State):
    print("🤖 Automation Engine: Analyzing Ticket...")
    text_lower = state['review_text'].lower()
    
    # Robust trigger keywords for background processing
    escalation_triggers = ["bad", "terrible", "refund", "broken", "late", "delay", "issue", "smell", "spark"]
    
    if any(word in text_lower for word in escalation_triggers):
        sentiment = "Negative"
        escalation = True
    else:
        sentiment = "Positive"
        escalation = False
        
    return {"sentiment": sentiment, "requires_escalation": escalation}

def rag_agent(state: State):
    if retriever is None:
        return {"policy_context": "Standard policies apply."}
    
    print("📚 Document Engine: Searching vectors...")
    docs = retriever.invoke(state['review_text'])
    context = "\n\n".join([d.page_content for d in docs])
    return {"policy_context": context}

def response_agent(state: State):
    print("✍️ Generative Engine: Compiling Draft...")
    if state["requires_escalation"]:
        prompt = ChatPromptTemplate.from_template("""
        You are an official Customer Support Agent. NEVER impersonate the customer.
        Use this policy context to solve the customer's complaint:\n{context}\n\n
        Write a professional apology + solution email for this review:\n"{review}"
        """)
    else:
        prompt = ChatPromptTemplate.from_template("""
        You are an official Customer Support Agent.
        Write a warm, professional thank you email for this positive review:\n"{review}"
        """)
    
    response = (prompt | llm).invoke({
        "context": state.get("policy_context", ""),
        "review": state['review_text']
    })
    return {"final_response": response.content.strip()}

# ====================== GRAPH COMPILATION ======================
workflow = StateGraph(State)
workflow.add_node("analyzer", analyzer)
workflow.add_node("rag", rag_agent)
workflow.add_node("response", response_agent)

workflow.set_entry_point("analyzer")
workflow.add_edge("analyzer", "rag")
workflow.add_edge("rag", "response")
workflow.add_edge("response", END)

app = workflow.compile()

# ====================== DAEMON LOOP ======================
def process_review(review_text: str, customer_email=None):
    result = app.invoke({"review_text": review_text, "customer_email": customer_email})
    print(f"\n✅ Processing Complete: {result['sentiment']} | Routed to Support: {result['requires_escalation']}")
    print(f"✉️ Output Generated:\n{result['final_response'][:400]}...\n")
    return result

if __name__ == "__main__":
    print("🚀 Background Automation Daemon Activated.\n")
    
    # Health Check Run
    test_review = "I received the product 15 days ago and now it stopped working. I want a refund."
    process_review(test_review, "customer@example.com")
    
    # Scheduler Logic
    def job():
        print(f"[{datetime.now()}] Background sync complete.")
        
    schedule.every(30).minutes.do(job)
    print("Listening for scheduled tasks. Press Ctrl+C to exit.")
    
    while True:
        schedule.run_pending()
        time.sleep(10)