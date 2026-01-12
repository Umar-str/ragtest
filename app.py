# app.py
import streamlit as st
import brain
import os

st.set_page_config(page_title="Perk Expert Agent", layout="centered")

# --- UI SKIN ---
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .card { 
        background: #161b22; 
        padding: 25px; 
        border-radius: 12px; 
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# API Key handling
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.warning("Please configure GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

if "agent" not in st.session_state:
    st.session_state.agent = brain.PerkAgent(api_key)

agent = st.session_state.agent

# --- SIDEBAR ---
with st.sidebar:
    st.success("Gemini 3 API Connected")
    st.title("Settings")
    file = st.file_uploader("Upload Knowledge Base (.txt)", type="txt")
    if file and st.button("Index Now"):
        text = file.getvalue().decode("utf-8")
        with st.spinner("Embedding..."):
            count = agent.add_documents(text)
            st.info(f"Successfully indexed {count} chunks.")

# --- MAIN ---
st.title("🤖 Perk Expert Agent")
query = st.text_input("How can I help you today?")

if st.button("Consult AI", type="primary"):
    if query:
        with st.spinner("Consulting knowledge base..."):
            ans, thoughts = agent.ask(query)
            
            # Use our custom HTML card
            st.markdown(f'<div class="card"><b>Expert Response:</b><br><br>{ans}</div>', unsafe_allow_html=True)
            
            # PDF Download
            pdf = brain.generate_pdf_report(query, ans, thoughts)
            st.download_button("📥 Download Official Report", pdf, "Perk_Report.pdf", "application/pdf")