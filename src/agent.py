from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        retrieved = self.store.search(query=question, top_k=top_k)

        if retrieved:
            context_blocks = [
                f"[Chunk {idx + 1}] (score={item['score']:.3f})\n{item['content']}"
                for idx, item in enumerate(retrieved)
            ]
            context = "\n\n".join(context_blocks)
        else:
            context = "[No relevant context retrieved from the knowledge base.]"

        prompt = (
            "You are a helpful assistant that answers ONLY from provided context.\n"
            "If context is insufficient, say what is missing.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
