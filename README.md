# Medical RAG Hallucination Detection

A Retrieval-Augmented Generation (RAG) system for medical question answering with built-in hallucination detection and verification.

## Project Structure

```
medical-rag-hallucination/
├── config/          # Central configuration (Pydantic Settings)
├── data/            # Datasets (corpus, PubMedQA, MedMCQA, annotations)
├── index/           # Index building (corpus, BM25, FAISS)
├── retrieval/       # Retrieval pipeline (BM25 + FAISS hybrid)
├── generation/      # LLM-based answer generation
├── verification/    # Hallucination detection & confidence scoring
├── evaluation/      # Metrics and baseline evaluation
├── app/             # Streamlit web application
├── tests/           # Pytest test suite
└── logs/            # Application logs
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd medical-rag-hallucination
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # for development
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

5. **Run tests**
   ```bash
   pytest
   ```

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Project scaffolding & config | ✅ |
| 2 | Data loading & preprocessing | ⬜ |
| 3 | Corpus construction | ⬜ |
| 4 | BM25 index building | ⬜ |
| 5 | Hybrid retrieval (BM25 + FAISS) | ⬜ |
| 6 | FAISS index building | ✅ |
| 7 | Retriever integration | ⬜ |
| 8 | LLM generation with prompts | ⬜ |
| 9 | Claim extraction | ⬜ |
| 10 | Hallucination verification | ⬜ |
| 11 | Confidence scoring | ⬜ |
| 12 | Evaluation & baselines | ⬜ |
| 13 | Streamlit app | ⬜ |

## License

MIT