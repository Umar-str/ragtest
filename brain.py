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

    def add_documents(self, text):
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
        q_res = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        results = self.collection.query(query_embeddings=[q_res.embeddings[0].values], n_results=3)
        context = " ".join(results['documents'][0]) if results['documents'] else ""
        
        resp = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=f"CONTEXT: {context}\n\nQUESTION: {query}",
            config=types.GenerateContentConfig(
                system_instruction="Be a footballer. Only use context provided and some generic coach related data."
            )
        )
        return resp.text