import chromadb
from google import genai
from google.genai import types

MODEL_NAME = "gemini-3-flash-preview"
EMBED_MODEL = "text-embedding-004"

class PerkAgent:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self.chroma_client.get_or_create_collection(name="perk_temp_db")

    def clear_db(self):
        """Wipes the database for clean testing."""
        self.chroma_client.delete_collection("perk_temp_db")
        self.collection = self.chroma_client.get_or_create_collection("perk_temp_db")

    def add_documents(self, text):
        """Splits and indexes text into the vector store."""
        chunks = [c.strip() for c in text.split('\n\n') if len(c.strip()) > 10]
        for i, chunk in enumerate(chunks):
            res = self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=chunk,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            self.collection.add(
                ids=[f"id_{i}_{hash(chunk)}"],
                embeddings=[res.embeddings[0].values],
                documents=[chunk]
            )
        return len(chunks)

    def ask(self, query):
        """RAG logic: Retrieval -> Generation."""
        # 1. Embed query
        q_res = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        
        # 2. Search DB
        results = self.collection.query(query_embeddings=[q_res.embeddings[0].values], n_results=3)
        context = " ".join(results['documents'][0]) if results['documents'] else "No relevant context found."
        
        # 3. Answer
        resp = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=f"CONTEXT: {context}\n\nQUESTION: {query}",
            config=types.GenerateContentConfig(
                system_instruction="You are a Perk HRMS assistant. Answer ONLY using the context. Be direct."
            )
        )
        return resp.text