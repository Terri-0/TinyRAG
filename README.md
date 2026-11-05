# Tiny RAG: Retail Payments Q&A

A small retrieval-augmented generation (RAG) prototype in Python. It answers questions about Canada's *Retail Payment Activities Act* (RPAA) and the Bank of Canada's retail payments supervision pages, using a **local** LLM through [Ollama](https://ollama.com). No API key is needed, and no data leaves the machine.

## How it works

```
docs/*.txt ──► chunk (overlapping) ──► embed ──► vectors
                                                   │
question ──► embed ──► cosine similarity ──► top-k chunks
                                                   │
                       prompt (context + rules) ──► LLM ──► answer with citations
```

1. **Ingest:** each document is split into word chunks (default 120 words, with a 20-word overlap so sentences on a chunk boundary stay intact in at least one chunk), and each chunk is embedded with `nomic-embed-text`.
2. **Retrieve:** the question is embedded and compared with every chunk using cosine similarity. The top-k (default 3) chunks are kept, best first.
3. **Generate:** the chunks are numbered and passed to `llama3.1:8b` with a system prompt that requires:
   - answering only from the provided context,
   - citing chunks as `[1]`, `[2]`, …,
   - replying "I don't know" when the answer isn't in the context,
   - treating the context as data, not instructions (a basic prompt-injection mitigation).

   Temperature is set to 0.1 for consistent answers.

## Setup

```powershell
# Install Ollama: https://ollama.com/download  (or: winget install Ollama.Ollama)
ollama pull nomic-embed-text
ollama pull llama3.1:8b          # or llama3.2:3b on lighter hardware
pip install ollama numpy
```

Put plain-text source documents in `docs/` (only `.txt` files), then run:

```powershell
python rag.py
```

The app prints the retrieved chunks with their similarity scores, then the answer.

## Sources used

- Bank of Canada: Retail payments supervision
- Bank of Canada: Retail payments supervisory framework
- *Retail Payment Activities Act* (S.C. 2021, c. 23, s. 177)

## Findings from testing

- **Ranking:** for "What must payment service providers submit each year?", the chunk containing the answer ranked **2nd**, behind a more general page. The answer was still correct because the top 3 chunks were passed. At scale, a re-ranker (a cross-encoder, or a late-interaction model such as ColBERT) would improve ordering.
- **Citation-only answers:** for a question with no real answer in the corpus, the model first returned only `[1]`. Fixed by requiring complete sentences before citations.
- **Unsupported citations:** the model sometimes attached a citation to a claim the cited chunk didn't support. Checking that citations are present isn't enough; groundedness has to be checked claim by claim.

## Limitations and next steps

- Brute-force cosine search over all chunks; a vector index (e.g. pgvector, FAISS) would be needed at scale.
- Fixed-size word chunking; structure-aware chunking on sections and headings would suit legal text better.
- Next: a small evaluation set (in-scope, out-of-scope, paraphrased and adversarial questions) measuring retrieval recall@k and groundedness; tool calling with least-privilege tools.

---

*This README was generated with the help of Claude (Anthropic).*
