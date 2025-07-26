import os
import json
import fitz  # PyMuPDF
import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_persona(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_text_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            chunks.append({
                "document": os.path.basename(pdf_path),
                "page": page_num,
                "text": text
            })
    return chunks

def rank_chunks(chunks, query, model):
    query_embedding = model.encode([query])
    texts = [chunk['text'] for chunk in chunks]
    text_embeddings = model.encode(texts)

    scores = cosine_similarity(query_embedding, text_embeddings)[0]
    scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    return scored_chunks[:5]  # top 5 relevant sections

def build_output(scored_chunks, persona_data):
    timestamp = datetime.datetime.now().isoformat()

    metadata = {
        "documents": list(set([c['document'] for c, _ in scored_chunks])),
        "persona": persona_data['persona'],
        "job": persona_data['job'],
        "timestamp": timestamp
    }

    extracted_sections = []
    subsection_analysis = []

    for rank, (chunk, score) in enumerate(scored_chunks, start=1):
        extracted_sections.append({
            "document": chunk["document"],
            "page": chunk["page"],
            "section_title": chunk["text"].split('\n')[0][:100],
            "importance_rank": rank
        })

        subsection_analysis.append({
            "document": chunk["document"],
            "page": chunk["page"],
            "refined_text": chunk["text"][:1000]  # trim long text
        })

    return {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

def main():
    input_dir = "/app/input"
    output_dir = "/app/output"

    persona_file = os.path.join(input_dir, "persona.json")
    persona_data = load_persona(persona_file)
    query = persona_data["persona"] + " - " + persona_data["job"]

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

    all_chunks = []
    for file in os.listdir(input_dir):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, file)
            all_chunks.extend(extract_text_chunks(pdf_path))

    top_chunks = rank_chunks(all_chunks, query, model)
    output = build_output(top_chunks, persona_data)

    with open(os.path.join(output_dir, "output.json"), "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
