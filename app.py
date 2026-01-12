# app.py
import streamlit as st
import brain
import os

st.set_page_config(page_title="Perk AI Agent", layout="centered")

# --- UI STYLE ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .card {
        background: #161b22;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .thought { color: #8b949e; font-style: italic; font-size: 0.9em; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

# --- AGENT INITIALIZATION ---
if "agent" not in st.session_state:
    # Use your actual key or streamlit secrets
    api_key = st.secrets.get("GEMINI_API_KEY", "YOUR_KEY_HERE")
    st.session_state.agent = brain.PerkAgent(api_key)

agent = st.session_state.agent

# --- SIDEBAR (Uploads) ---
with st.sidebar:
    st.header("🏢 Knowledge Base")
    file = st.file_uploader("Upload Policy (.txt)", type="txt")
    if file and st.button("Index Data"):
        text = file.getvalue().decode("utf-8")
        count = agent.add_documents(text)
        st.success(f"Indexed {count} sections.")

# --- MAIN UI ---
st.title("⚽ Perk Expert")
query = st.text_input("Ask a policy question:", placeholder="e.g. What is the WFH policy?")

if st.button("Consult Agent", type="primary"):
    if query:
        with st.spinner("Analyzing knowledge base..."):
            ans, thoughts = agent.ask(query)
            
            # Displaying in our custom HTML skin
            st.markdown(f"""
            <div class="card">
                <div style="color:#58a6ff; font-weight:bold; margin-bottom:10px;">EXPERT GUIDANCE</div>
                <div>{ans}</div>
                <div class="thought"><b>Agent Trace:</b> {thoughts[:300]}...</div>
            </div>
            """, unsafe_allow_html=True)
            
            # PDF Download
            pdf_bytes = brain.generate_pdf_report(query, ans, thoughts)
            st.download_button(
                label="📥 Download Report",
                data=pdf_bytes,
                file_name="perk_report.pdf",
                mime="application/pdf"
            )