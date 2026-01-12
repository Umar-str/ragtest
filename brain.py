# brain.py
import chromadb
from google import genai
from google.genai import types
from fpdf import FPDF
import io

# 2026 Native Gemini 3 Models
MODEL_NAME = "gemini-3-flash-preview"
EMBED_MODEL = "text-embedding-004" 

class PerkAgent:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.chroma_client = chromadb.EphemeralClient()
        self.collection = self.chroma_client.get_or_create_collection(name="perk_temp_db")

    def add_documents(self, text_content):
        # Chunking text into readable blocks
        chunks = [c.strip() for c in text_content.split('\n\n') if len(c.strip()) > 10]
        
        for i, chunk in enumerate(chunks):
            # The SDK requires this exact structure for Gemini 3 embeddings
            response = self.client.models.embed_content(
                model=EMBED_MODEL,
                contents=chunk,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            vector = response.embeddings[0].values
            
            self.collection.add(
                ids=[f"id_{i}_{hash(chunk)}"],
                embeddings=[vector],
                documents=[chunk]
            )
        return self.collection.count()

    def ask(self, query):
        # 1. Retrieve using the Query task type
        query_resp = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        
        results = self.collection.query(
            query_embeddings=[query_resp.embeddings[0].values], 
            n_results=3
        )
        
        context = " ".join(results['documents'][0]) if results['documents'] else "No relevant info found."

        # 2. Generate Answer with Thinking Enabled
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
        # Extract thoughts from the Gemini 3 response parts
        thoughts = response.candidates[0].content.parts[0].text if response.candidates else "Process completed."
        
        return answer, thoughts

def generate_pdf_report(query, answer, thoughts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Perk AI Official Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, f"Query: {query}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 7, f"Expert Response:\n{answer}")
    return pdf.output()