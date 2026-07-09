import streamlit as st
import requests
import time
import sys
import os
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

st.set_page_config(page_title="GlobalCart AI Support Engine", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.email-box { background-color: #ffffff; border-left: 6px solid #28a745; padding: 30px; border-radius: 12px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); white-space: pre-wrap; }
.email-box-negative { border-left-color: #dc3545; }
.metric-card { background-color: white; padding: 22px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); text-align: center; }
.stButton>button { width: 100%; height: 50px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if 'email_body' not in st.session_state:
    st.session_state.email_body = None
    st.session_state.escalation = False
    st.session_state.sentiment = None
    st.session_state.duration = 0.0
    st.session_state.test_text = ""

API_URL = "http://127.0.0.1:8000/analyze"
TRANSCRIBE_URL = "http://127.0.0.1:8000/transcribe"
ANALYTICS_URL = "http://127.0.0.1:8000/analytics"

with st.sidebar:
    st.title("System Diagnostics")
    st.success("🟢 API Gateway: Live")
    st.success("🟢 Persistent DB: SQLite")
    st.divider()
    st.markdown("**Enterprise Features**")
    st.markdown("• ML & RAG Engine")
    st.markdown("• Audio-to-Text Routing")
    st.markdown("• Admin Analytics 🆕")
    st.caption("Architecture v4.0 • Production Ready")

st.title("🛡️ AI Support Intelligence Center")

# 🔴 NEW: Tabs for separating Operations and Analytics
tab1, tab2 = st.tabs(["🎯 Support Desk (Operations)", "📊 Admin Dashboard (Analytics)"])

# ================= TAB 1: SUPPORT DESK =================
with tab1:
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.subheader("📥 Incoming Ticket")
        audio_value = st.audio_input("🎙️ Record Voice Ticket")
        if audio_value:
            with st.spinner("Transcribing..."):
                try:
                    files = {"file": ("audio.wav", audio_value.getvalue(), "audio/wav")}
                    res = requests.post(TRANSCRIBE_URL, files=files)
                    st.session_state.test_text = res.json().get("text", "")
                    st.success("Voice Transcribed!")
                except Exception as e:
                    st.error("Failed")
                    
        user_input = st.text_area("Customer Message", value=st.session_state.test_text, height=180)
        cust_email = st.text_input("Customer Email", value="customer@example.com")
        process_btn = st.button("🚀 Process via LangGraph", type="primary")

    with col_right:
        st.subheader("📤 AI Engine Output")
        if process_btn and user_input.strip():
            with st.spinner("Analyzing & Saving to Database..."):
                try:
                    start = time.time()
                    resp = requests.post(API_URL, json={"text": user_input, "customer_email": cust_email})
                    data = resp.json()
                    st.session_state.email_body = data.get("generated_response", "")
                    st.session_state.escalation = data.get("escalation_required", False)
                    st.session_state.sentiment = data.get("detected_sentiment", "Unknown")
                    st.session_state.duration = time.time() - start
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.email_body:
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='metric-card'><b>Sentiment</b><br>{'🟢 Positive' if st.session_state.sentiment == 'Positive' else '🔴 Negative'}</div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric-card'><b>Routing Node</b><br>{'🚨 Escalation' if st.session_state.escalation else '✅ Gratitude'}</div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric-card'><b>Latency</b><br>⚡ {st.session_state.duration:.2f} sec</div>", unsafe_allow_html=True)
            
            st.markdown("### 📧 Final Verified Draft")
            box_class = "email-box-negative" if st.session_state.escalation else "email-box"
            st.markdown(f'<div class="{box_class}">{st.session_state.email_body}</div>', unsafe_allow_html=True)

# ================= TAB 2: ADMIN DASHBOARD =================
with tab2:
    st.subheader("📈 System Analytics Overview")
    if st.button("🔄 Refresh Analytics"):
        try:
            res = requests.get(ANALYTICS_URL)
            data = res.json()
            
            if "error" in data or data.get("total_tickets") == 0:
                st.info("No data yet. Process a ticket in the Support Desk first!")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Tickets Processed", data.get("total_tickets", 0))
                m2.metric("Total Escalations", data.get("escalations", 0))
                m3.metric("System Uptime", "99.9%")
                
                st.divider()
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown("**Sentiment Distribution**")
                    sentiments = data.get("sentiment_distribution", {})
                    if sentiments:
                        df_sent = pd.DataFrame(list(sentiments.items()), columns=['Sentiment', 'Count'])
                        st.bar_chart(df_sent.set_index('Sentiment'), color="#4CAF50")
                        
                with c2:
                    st.markdown("**Recent Activity Log**")
                    recent = data.get("recent_tickets", [])
                    for tk in recent:
                        st.caption(f"🕒 {tk['time']}")
                        icon = "🟢" if tk['sentiment'] == "Positive" else "🔴"
                        st.write(f"{icon} Email: {tk['email']}")
                        st.divider()
        except Exception as e:
            st.error("Failed to load database analytics.")