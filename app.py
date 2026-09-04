import math
import re

import requests
import streamlit as st


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "embeddinggemma"
CHAT_URL = "http://localhost:11434/api/chat"
CHAT_MODEL = "llama3.2"


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def create_embeddings(chunks: list[str]) -> list[list[float]]:
    """Ask local Ollama to turn every chunk into a numeric vector."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": EMBEDDING_MODEL, "input": chunks},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Measure how similar two embedding vectors are, from -1 to 1."""
    dot_product = sum(a * b for a, b in zip(first, second))
    first_length = math.sqrt(sum(value * value for value in first))
    second_length = math.sqrt(sum(value * value for value in second))
    return dot_product / (first_length * second_length)

def keyword_overlap(question: str, chunk: str)->float:
    """Return the proportion of meaningful question words found in a chunk."""
    ignored_words = {"a", "an", "are", "for", "is", "my", "of", "the", "to", "what"}
    question_words = set(re.findall(r"\b[a-z]+\b", question.lower())) - ignored_words
    chunk_words = set(re.findall(r"\b[a-z]+\b", chunk.lower()))
    if not question_words:
        return 0.0
    return len(question_words & chunk_words) / len(question_words)


def find_relevant_chunks(
    question: str,
    chunks: list[str],
    document_embeddings: list[list[float]],
    top_k: int = 5,
) -> list[tuple[str, float,float]]:
    """Embed a question, score every chunk, and return the best matches."""
    question_embedding = create_embeddings([question])[0]

    scored_chunks = [(chunk, cosine_similarity(question_embedding, embedding),keyword_overlap(question, chunk),)
        for chunk, embedding in zip(chunks, document_embeddings)
    ]
    return sorted(scored_chunks, key=lambda item: (0.85 * item[1]) + (0.15 * item[2]), reverse=True)[:top_k]


def generate_answer(question: str, relevant_chunks: list[tuple[str, float]]) -> str:
    """Generate an answer using only the retrieved document passages."""
    context = "\n\n".join(
        f"Passage {number}:\n{chunk}"
        for number, (chunk, _, _) in enumerate(relevant_chunks, start=1)
    )
    response = requests.post(
        CHAT_URL,
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise document question-answering assistant. "
                        "Read every supplied passage before answering. "
                        "Answer only from facts explicitly written in those passages. "
                        "When a passage explicitly states the requested fact, give that fact directly. "
                        "Only say 'I cannot find that in the document.' when no supplied passage "
                        "contains the requested fact. Do not use outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Document passages:\n{context}\n\nQuestion: {question}",
                },
            ],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


st.set_page_config(page_title="My RAG Learning App", page_icon="📚")

st.title("📚 My RAG Learning App")
st.write("We will build this application step by step.")

uploaded_file = st.file_uploader(
    "Upload a text document",
    type=["txt"],
)

if uploaded_file:
    document_text = uploaded_file.getvalue().decode("utf-8")
    chunks = split_text(document_text)

    st.success(f"Loaded: {uploaded_file.name}")
    st.caption(f"Document length: {len(document_text):,} characters")
    st.caption(f"Created {len(chunks)} searchable chunks")
    st.caption("Embeddings run locally with Ollama — no API key required.")
    with st.expander("Preview document text"):
        st.text(document_text)

    with st.expander("Preview chunks"):
        for number, chunk in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {number}** — {len(chunk)} characters")
            st.code(chunk, language=None)

    if st.button("Create local embeddings"):
        try:
            with st.spinner("Creating vectors locally with Ollama..."):
                embeddings = create_embeddings(chunks)

            st.session_state.embeddings = embeddings
            st.success(f"Created {len(embeddings)} local embeddings.")
            st.caption(f"Each embedding has {len(embeddings[0]):,} numeric dimensions.")
        except requests.RequestException as error:
            st.error(f"Ollama could not create embeddings: {error}")

    st.subheader("Retrieve relevant chunks")
    question = st.text_input("Ask a question about this document")

    if "embeddings" not in st.session_state:
        st.info("First, click 'Create local embeddings' for this document.")
    elif st.button("Find relevant chunks", disabled=not question):
        try:
            with st.spinner("Searching document chunks..."):
                st.session_state.relevant_chunks = find_relevant_chunks(
                    question,
                    chunks,
                    st.session_state.embeddings,
                )
                st.session_state.retrieval_question = question
        except requests.RequestException as error:
            st.error(f"Ollama could not search the document: {error}")

    if st.session_state.get("retrieval_question") == question:
        relevant_chunks = st.session_state.relevant_chunks
        st.write("These passages were retrieved from your document:")
        for number, (chunk, score, keyword_score) in enumerate(relevant_chunks, start=1):
            st.markdown(f"**Result {number}** — similarity: {score:.3f}"f"keyword match: {keyword_score:.0%}")
            st.code(chunk, language=None)
        if st.button("Generate grounded answer"):
            try:
                with st.spinner("Generating an answer from retrieved passages..."):
                    answer = generate_answer(question, relevant_chunks)

                st.subheader("Answer")
                st.write(answer)
            except requests.RequestException as error:
                st.error(f"Ollama could not generate an answer: {error}")
else:
    st.info("Upload a .txt file to begin.")
