from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer


class RetrievalEmbeddingService:
    """Owns the local embedding model and tokenizer used by retrieval."""

    QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name

        # Load the tokenizer separately so chunking uses the same token vocabulary as the embedding model.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Keep the embedding model local and normalize vectors before semantic retrieval.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
            query_encode_kwargs={"normalize_embeddings": True, "prompt": self.QUERY_PROMPT},
            show_progress=False,
        )

    def count_tokens(self, text: str) -> int:
        """Counts tokens using the embedding model's tokenizer."""

        if not text:
            return 0

        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    