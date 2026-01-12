__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import brain

st.set_page_config(page_title="Classic AI Assistant", layout="wide")

# Initialize the RAG Agent
if "agent" not in st.session_state:
    st.session_state.agent = brain.PerkAgent(st.secrets["GEMINI_API_KEY"])

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("📂 Knowledge Base")
    file = st.file_uploader("Upload Policy (.txt)", type="txt")
    
    if file and st.button("Index Documents", use_container_width=True):
        with st.status("Reading & Embedding...", expanded=True) as status:
            count = st.session_state.agent.add_documents(file.getvalue().decode("utf-8"))
            status.update(label=f"Successfully indexed {count} chunks!", state="complete")

    st.divider()
    st.metric("DB Size", f"{st.session_state.agent.collection.count()} Chunks")
    
    if st.button("Reset Session", type="secondary"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT ---
st.title("Your Custom AI")

# 1. Display historical messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Accept new input
if prompt := st.chat_input("Ask a question about the policy..."):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.status("Searching Vector DB...") as status:
            response = st.session_state.agent.ask(prompt)
            status.update(label="Found Answer", state="complete")
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})