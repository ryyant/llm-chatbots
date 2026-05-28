import io
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from config import RAG_EMBEDDING_MODEL


@dataclass
class Chunk:
    filename: str
    text: str
    embedding: list[float]


SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from an uploaded file. Raises ValueError for unsupported types."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if lower.endswith(".txt") or lower.endswith(".md"):
        return data.decode("utf-8", errors="replace")
    raise ValueError(
        f"Unsupported file type: {filename}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


_PARA_RE = re.compile(r"\n\s*\n+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        return [text] if text else []
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def chunk(text: str, chunk_chars: int = 2000, overlap_chars: int = 200) -> list[str]:
    """Split text into ~chunk_chars windows with overlap, preferring paragraph then sentence boundaries."""
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]

    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_chars:
            pieces.append(para)
            continue
        sentences = [s.strip() for s in _SENT_RE.split(para) if s.strip()]
        buf = ""
        for sent in sentences:
            if len(sent) > chunk_chars:
                if buf:
                    pieces.append(buf)
                    buf = ""
                pieces.extend(_hard_split(sent, chunk_chars, overlap_chars))
                continue
            candidate = (buf + " " + sent).strip() if buf else sent
            if len(candidate) <= chunk_chars:
                buf = candidate
            else:
                pieces.append(buf)
                buf = sent
        if buf:
            pieces.append(buf)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = (current + "\n\n" + piece).strip() if current else piece
        if len(candidate) <= chunk_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
                tail = current[-overlap_chars:] if overlap_chars > 0 else ""
                current = (tail + "\n\n" + piece).strip() if tail else piece
            else:
                current = piece
    if current:
        chunks.append(current)

    return chunks


EmbedFn = Callable[[list[str]], list[list[float]]]


def _default_embed(texts: list[str], api_key: str) -> list[list[float]]:
    from google import genai
    client = genai.Client(api_key=api_key)
    result = client.models.embed_content(
        model=RAG_EMBEDDING_MODEL,
        contents=texts,
    )
    return [list(e.values) for e in result.embeddings]


def embed(texts: list[str], api_key: str, embed_fn: Optional[EmbedFn] = None) -> list[list[float]]:
    """Embed a batch of texts. embed_fn is injectable for testing."""
    if not texts:
        return []
    if embed_fn is not None:
        return embed_fn(texts)
    return _default_embed(texts, api_key)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norms = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return b_norms @ a_norm


def retrieve(
    query: str,
    index: list[Chunk],
    api_key: str,
    top_k: int = 4,
    embed_fn: Optional[EmbedFn] = None,
) -> list[Chunk]:
    """Return top_k chunks most similar to query by cosine similarity."""
    if not index:
        return []
    query_vec = embed([query], api_key=api_key, embed_fn=embed_fn)[0]
    q = np.array(query_vec, dtype=np.float32)
    embeddings = np.array([c.embedding for c in index], dtype=np.float32)
    scores = _cosine(q, embeddings)
    order = np.argsort(-scores)[:top_k]
    return [index[i] for i in order]


def build_context_prompt(chunks: list[Chunk], user_message: str) -> str:
    """Format retrieved chunks + user question into a single message for the chat model."""
    if not chunks:
        return user_message
    lines = [
        "You have access to the following document excerpts. "
        "Use them to answer the user's question when relevant.",
        "",
    ]
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{c.filename} — excerpt {i}]")
        lines.append(c.text)
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"User question: {user_message}")
    return "\n".join(lines)
