import os
import joblib
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

# 1. API Key Setup
os.environ["GROQ_API_KEY"] = "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"

# 2. Load Custom ML Model
print("Loading Legacy Machine Learning Models...")
try:
    ml_model = joblib.load('models/sentiment_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    print("✅ Models loaded successfully.")
except FileNotFoundError:
    print("Error: Models not found in 'models/' folder.")
    exit()

# 3. Define Graph State
class AgentState(TypedDict):
    review_text: str
    sentiment: str
    requires_escalation: bool
    final_email: str

# 4. Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --- NODES ---

def analyzer_agent(state: AgentState):
    print("🤖 Base Analyzer (Custom ML): Processing data matrix...")
    cleaned_review = clean_text(state['review_text'])
    vectorized_review = vectorizer.transform([cleaned_review])
    
    prediction = ml_model.predict(vectorized_review)[0]
    
    sentiment_label = "Positive" if prediction == 1 else "Negative"
    requires_escalation = True if prediction == 0 else False
    
    return {
        "sentiment": sentiment_label,
        "requires_escalation": requires_escalation
    }

def apology_agent(state: AgentState):
    print("🚨 Routing: Drafting escalation email...")
    prompt = f"""
    You are an authorized Support Agent. NEVER impersonate the customer.
    Write a short, professional 2-sentence apology email to a customer who left this negative review:
    "{state['review_text']}"
    Sign off as "Customer Support Team".
    """
    response = llm.invoke(prompt)
    return {"final_email": response.content.strip()}

def gratitude_agent(state: AgentState):
    print("🌟 Routing: Drafting thank you email...")
    prompt = f"""
    You are an authorized Support Agent. NEVER impersonate the customer.
    Write a short, professional 2-sentence thank you email to a customer who left this positive review:
    "{state['review_text']}"
    Sign off as "Customer Support Team".
    """
    response = llm.invoke(prompt)
    return {"final_email": response.content.strip()}

def route_review(state: AgentState):
    if state["requires_escalation"]:
        return "apology"
    return "gratitude"

# --- GRAPH COMPILATION ---
workflow = StateGraph(AgentState)
workflow.add_node("analyzer", analyzer_agent)
workflow.add_node("apology", apology_agent)
workflow.add_node("gratitude", gratitude_agent)

workflow.set_entry_point("analyzer")
workflow.add_conditional_edges(
    "analyzer", 
    route_review, 
    {"apology": "apology", "gratitude": "gratitude"}
)
workflow.add_edge("apology", END)
workflow.add_edge("gratitude", END)

app = workflow.compile()

# --- EXECUTION ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST: EXECUTING HYBRID GRAPH PIPELINE")
    test_review = {"review_text": "Terrible product. It broke after two days of use. Do not buy this."}
    
    result = app.invoke(test_review)
    
    print("\n📊 Graph Engine Output:")
    print(f"ML Detected Sentiment: {result.get('sentiment')}")
    print(f"LLM Generated Email:\n{result.get('final_email')}")
    print("="*60 + "\n")