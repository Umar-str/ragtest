# brain.py
import chromadb
from google import genai
from google.genai import types
from fpdf import FPDF
import io

# Constants for 2026 Models
MODEL_NAME = "gemini-3-flash-preview"
EMBED_MODEL = "gemini-embedding-001"

class PerkAgent:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        # RAM-based DB for speed and session persistence
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self.chroma_client.get_or_create_collection(name="perk_temp_db")

    def add_documents(self, text_content):
        """Chunks and embeds text into the RAM cache."""
        # Simple chunking: split by paragraphs to keep context
        chunks = [c.strip() for c in text_content.split('\n\n') if len(c.strip()) > 10]
        
        for i, chunk in enumerate(chunks):
            response = self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=chunk,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            vector = response.embeddings[0].values
            self.collection.add(
                ids=[f"id_{hash(chunk)}"],
                embeddings=[vector],
                documents=[chunk]
            )
        return self.collection.count()

    def ask(self, query):
        """Retrieves context and reasons through the answer."""
        # 1. Retrieval
        query_resp = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        results = self.collection.query(
            query_embeddings=[query_resp.embeddings[0].values], 
            n_results=3
        )
        context = " ".join(results['documents'][0]) if results['documents'] else "No context found."

        # 2. Generation with Thinking
        system_msg = "You are a Perk HRMS expert. Answer ONLY using the provided context."
        prompt = f"CONTEXT: {context}\n\nQUESTION: {query}"

        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_msg,
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
        )
        
        answer = response.text
        thoughts = response.candidates[0].content.parts[0].text if response.candidates else "N/A"
        return answer, thoughts

def generate_pdf_report(query, answer, thoughts):
    """Creates a professional PDF byte-stream for download."""
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Perk AI: Official HRMS Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, f"Query: {query}", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 8, f"Response:\n{answer}")
    pdf.ln(10)
    
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 6, f"Internal Reasoning Trace:\n{thoughts}")
    
    # Return as bytes
    return pdf.output()