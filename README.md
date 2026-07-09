# 🛡️ Enterprise AI Support Intelligence Center

An advanced, decoupled Customer Support Automation Engine powered by Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Agentic Workflows. This system automatically classifies incoming tickets, retrieves corporate policies, drafts context-aware responses, and dispatches real emails to customers.

## 🚀 Key Features
* **Hybrid Sentiment Analysis:** Combines Custom ML Models (Logistic Regression/TF-IDF) with LLM fallbacks for hyper-accurate ticket routing.
* **Agentic Orchestration:** Utilizes **LangGraph** to route tickets intelligently between 'Apology/Escalation' and 'Gratitude' agents.
* **RAG Policy Engine:** Integrated **FAISS Vector Database** and HuggingFace Embeddings to strictly enforce corporate SLAs and technical safety guidelines from PDF documents.
* **Real-time Email Dispatch:** Built-in SMTP module to automatically send verified AI-generated drafts directly to customer inboxes.
* **Microservices Architecture:** Fully decoupled FastAPI backend and Streamlit frontend.

## 🛠️ Technology Stack
* **Backend:** FastAPI, Python
* **Frontend:** Streamlit
* **AI/Orchestration:** LangChain, LangGraph, Groq API (Llama-3.3-70b-versatile)
* **Vector DB & Embeddings:** FAISS, sentence-transformers (all-MiniLM-L6-v2)
* **Machine Learning:** Scikit-learn, Joblib

## 📂 Project Structure
```text
├── ai_core/           # LangGraph workflows and background daemons
├── backend/           # FastAPI application and routing logic
├── data/              # Policy PDFs and static RAG knowledge base
├── frontend/          # Streamlit UI dashboard
├── models/            # Pre-trained ML classifiers
├── services/          # External integrations (SMTP email sender)
├── requirements.txt   # Dependency list
└── README.md          # Project documentation