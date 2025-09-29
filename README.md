# Workflow

1.  **User input** → goes into your app (main loop).
2.  **Guardrails / Moderation**: you run a content-moderation LLM that either passes, sanitizes, or blocks input.
3.  **Routing**: a small LLM classifier decides llm vs rag.
4.  **LLM**: answer from chat-history + LLM only.
5.  **RAG**: retrieval from vector store first, then LLM uses retrieved context + query.
6.  **Retrieval**: retriever (FAISS) finds top-k similar document chunks (these Document objects include page_content and metadata like id, slug, title, source, chunk_id).
7.  **Answer generation**: LLM is given a prompt with context (the retrieved chunks) and the question. It produces an answer. You append a compact list of slug/title as sources for provenance.
8.  **State persistence**: the StateGraph with MongoDBSaver writes the updated state (messages list, router_decision, retrieved_docs if you return them) to MongoDB under the thread_id. On the next turn the state is rehydrated so the agent sees past messages.
9.  **Return**: the final answer (and optionally explicit source list) is printed back to the user.

