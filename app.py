import streamlit as st
import brain

st.set_page_config(page_title="Perk AI | RAG Tester", layout="wide")

# Custom Dark Theme for a "Product" feel
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .stChatMessage { border-bottom: 1px solid #21262d; padding: 20px; }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
if "agent" not in st.session_state:
    st.session_state.agent = brain.PerkAgent(st.secrets["GEMINI_API_KEY"])
if "messages" not in st.session_state:
    st.session_state.messages = []

agent = st.session_state.agent

# --- SIDEBAR: DB VERIFICATION & UPLOAD ---
with st.sidebar:
    st.title("📂 Knowledge Base")
    file = st.file_uploader("Upload Policy Document", type="txt")
    
    if file and st.button("Index into Vector DB", use_container_width=True):
        with st.spinner("Embedding..."):
            count = agent.add_documents(file.getvalue().decode("utf-8"))
            st.success(f"Added {count} new chunks!")
    
    st.divider()
    
    # DB Status (Live Verification)
    db_size = agent.collection.count()
    st.metric(label="Total Chunks in DB", value=db_size)
    
    if st.button("Clear All Data", type="secondary", use_container_width=True):
        agent.clear_db()
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("Custom AI Tester")

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Bottom Input Chat
if query := st.chat_input("Verify a policy..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching DB..."):
            response = agent.ask(query)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})