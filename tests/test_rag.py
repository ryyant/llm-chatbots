import io
import math

import pytest

from rag import (
    Chunk,
    build_context_prompt,
    chunk,
    extract_text,
    retrieve,
)


def test_extract_text_txt():
    out = extract_text("notes.txt", b"hello world")
    assert out == "hello world"


def test_extract_text_md():
    out = extract_text("readme.md", b"# Title\n\nbody")
    assert "# Title" in out
    assert "body" in out


def test_extract_text_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text("data.docx", b"...")


def test_extract_text_pdf():
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    out = extract_text("blank.pdf", buf.getvalue())
    assert isinstance(out, str)


def test_chunk_empty_text_returns_empty():
    assert chunk("") == []
    assert chunk("   \n  ") == []


def test_chunk_short_text_returns_one_chunk():
    out = chunk("short text", chunk_chars=2000)
    assert out == ["short text"]


def test_chunk_long_text_splits_into_multiple():
    paragraphs = ["para " + str(i) + " " + "x" * 600 for i in range(5)]
    text = "\n\n".join(paragraphs)
    out = chunk(text, chunk_chars=1000, overlap_chars=100)
    assert len(out) > 1
    assert all(len(c) <= 1100 for c in out)


def test_chunk_preserves_paragraph_boundaries_when_possible():
    text = "alpha\n\nbeta\n\ngamma"
    out = chunk(text, chunk_chars=2000)
    assert len(out) == 1
    assert "alpha" in out[0]
    assert "gamma" in out[0]


def test_chunk_hard_splits_single_huge_paragraph():
    text = "x" * 5000
    out = chunk(text, chunk_chars=1000, overlap_chars=100)
    assert len(out) > 1


def _fake_embed_factory(mapping: dict[str, list[float]]):
    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [mapping[t] for t in texts]
    return fake_embed


def test_retrieve_empty_index_returns_empty():
    out = retrieve("query", [], api_key="k", embed_fn=lambda ts: [[1.0]])
    assert out == []


def test_retrieve_orders_by_cosine_similarity():
    chunks = [
        Chunk(filename="a.txt", text="apples", embedding=[1.0, 0.0, 0.0]),
        Chunk(filename="b.txt", text="bananas", embedding=[0.0, 1.0, 0.0]),
        Chunk(filename="c.txt", text="oranges", embedding=[0.0, 0.0, 1.0]),
    ]
    fake = _fake_embed_factory({"fruit?": [0.9, 0.1, 0.0]})
    out = retrieve("fruit?", chunks, api_key="k", top_k=2, embed_fn=fake)
    assert [c.filename for c in out] == ["a.txt", "b.txt"]


def test_retrieve_top_k_caps_results():
    chunks = [
        Chunk(filename=f"{i}.txt", text=str(i), embedding=[float(i), 0.0])
        for i in range(1, 6)
    ]
    fake = _fake_embed_factory({"q": [1.0, 0.0]})
    out = retrieve("q", chunks, api_key="k", top_k=3, embed_fn=fake)
    assert len(out) == 3


def test_build_context_prompt_with_chunks_includes_filename_and_question():
    chunks = [
        Chunk(filename="report.pdf", text="revenue grew 12%", embedding=[]),
    ]
    out = build_context_prompt(chunks, "what happened to revenue?")
    assert "report.pdf" in out
    assert "revenue grew 12%" in out
    assert "what happened to revenue?" in out
    assert "User question:" in out


def test_build_context_prompt_no_chunks_returns_message_as_is():
    out = build_context_prompt([], "hi")
    assert out == "hi"
