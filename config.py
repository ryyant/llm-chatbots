VALID_PROVIDERS = ("gemini", "openai", "anthropic")

DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "openai": "gpt-4o",
    "anthropic": "claude-opus-4-7",
}

API_KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

RAG_CHUNK_CHARS = 2000
RAG_OVERLAP_CHARS = 200
RAG_TOP_K = 4
RAG_EMBEDDING_MODEL = "gemini-embedding-001"
SUPPORTED_UPLOAD_EXTENSIONS = (".pdf", ".txt", ".md")
