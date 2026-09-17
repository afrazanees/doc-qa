import io
import os
import numpy as np
from pypdf import PdfReader
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI(title="Document QA Assistant")

DOC_CHUNKS: list[str] = []
DOC_VECTORS: list[list[float]] = []

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"

# 700-word chunks with 100-word overlap preserve context across boundaries
# while fitting multiple retrieved excerpts within prompt limits
CHUNK_SIZE, CHUNK_OVERLAP = 700, 100


def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured. Add it to .env or system environment variables.",
        )
    return genai.Client(api_key=api_key)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        start += size - overlap
    return chunks


def top_chunks(query_vec: list[float], chunk_vecs: list[list[float]], chunks: list[str], k: int = 4) -> list[str]:
    q, m = np.array(query_vec), np.array(chunk_vecs)
    sims = m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-10)
    return [chunks[i] for i in np.argsort(sims)[-k:][::-1]]


class QuestionRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global DOC_CHUNKS, DOC_VECTORS

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    reader = PdfReader(io.BytesIO(content))
    extracted = [page.extract_text() for page in reader.pages if page.extract_text()]
    full_text = " ".join(extracted).strip()

    if not full_text:
        raise HTTPException(status_code=400, detail="No readable text found in the PDF.")

    chunks = chunk_text(full_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not extract text chunks from document.")

    client = get_client()
    vectors = []
    for chunk in chunks:
        res = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk,
        )
        vectors.append(res.embeddings[0].values)

    DOC_CHUNKS = chunks
    DOC_VECTORS = vectors

    return {
        "filename": file.filename,
        "chunks_count": len(chunks),
        "message": f"Successfully processed {len(chunks)} chunks from {file.filename}.",
    }


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    if not DOC_CHUNKS or not DOC_VECTORS:
        raise HTTPException(status_code=400, detail="Please upload a PDF document before asking questions.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    client = get_client()

    query_res = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
    )
    query_vec = query_res.embeddings[0].values
    relevant_chunks = top_chunks(query_vec, DOC_VECTORS, DOC_CHUNKS, k=4)

    context = "\n\n---\n\n".join(relevant_chunks)
    prompt = f"Context from document:\n{context}\n\nQuestion: {question}"

    # Strict grounding constraint to ensure answers rely only on document text
    system_prompt = 'Answer only from the provided context. Say "I don\'t know" if it isn\'t there.'

    response = None
    last_err = None
    for model_name in [GENERATION_MODEL, "gemini-2.5-flash-lite"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"system_instruction": system_prompt},
            )
            break
        except Exception as err:
            last_err = err

    if response is None:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {last_err}")

    return {
        "answer": response.text,
        "sources": relevant_chunks,
    }


app.mount("/", StaticFiles(directory="static", html=True), name="static")
