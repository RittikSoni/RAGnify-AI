# 🔍 Context-Aware Chatbot

A blazing-fast, **RAG (Retrieval‑Augmented Generation)** privacy-focused chatbot built with **FastAPI**, **LangChain**, **FAISS**, and **LLMs** to answer user questions strictly from your own data—no hallucinations, no external information.

---

## 🚀 Features

- **FastAPI**-powered REST API
- **FAISS** vector store for efficient semantic search
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2`) for document representation
- **LLaMA 3** (via Ollama) with zero-temperature for deterministic responses
- **Strict instruction** prompt to avoid hallucinations
- **Typo correction** for known questions
- **Fallback** to customer support when no answer is found
- **Retrieval‑Augmented Generation** (RAG) pipeline for context‑grounded answers

---

## 📦 Requirements

- Python 3.9+
- `fastapi`
- `uvicorn`
- `langchain-community`
- `langchain-huggingface`
- `langchain-ollama`
- `faiss-cpu` or `faiss-gpu`
- `pydantic`

Install dependencies:

```bash
pip install fastapi uvicorn langchain-community langchain-huggingface langchain-ollama faiss-cpu pydantic
```

---

## 🛠️ Project Structure

```
.
├── app.py            # Main FastAPI application
├── ingest.py         # Script to build FAISS index from data files
├── data/             # Folder containing .txt files to index
├── faiss_index/      # Generated FAISS index directory
├── README.md         # This file
└── requirements.txt  # Pinned dependencies (optional)
```

---

## 🔧 Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-org/faq-chatbot.git
   cd faq-chatbot
   ```

2. **Index your data**  
   Place your `.txt` (or other supported) files in the `data/` folder, then run:

   ```bash
   python -m ingest
   ```

3. **Run the API**

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. **Expose via Ngrok (optional)**
   ```bash
   ngrok http 8000
   ```

---

## 📡 API Usage

### POST `/chat`

Request:

```json
{
  "question": "Your FAQ question here"
}
```

Response:

```json
{
  "answer": "Answer from context or fallback message",
  "sources": ["source1.txt", "source2.txt"]
}
```

---

## 📜 System Prompt

Located in `app.py` as `SYSTEM_PROMPT`, enforcing:

1. **Context-only answers**
2. **Greeting handling**
3. **Typo correction**
4. **Fallback message** on unknown queries

---

## 📝 Scripts

| Command                          | Description                         |
| -------------------------------- | ----------------------------------- |
| `python -m ingest`               | Build/update the FAISS index        |
| `uvicorn app:app --host 0.0.0.0` | Run the FastAPI application         |
| `ngrok http 8000`                | Tunnel local API to public endpoint |

---

## 📂 Environment Variables

| Variable      | Default            | Description               |
| ------------- | ------------------ | ------------------------- |
| `DATA_INDEX`  | `faiss_index`      | Directory for FAISS index |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model name      |
| `LLM_MODEL`   | `llama3.1:8b`      | Ollama LLM model version  |
| `PORT`        | `8000`             | Port for FastAPI server   |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

Please adhere to **PEP8**, write **tests**, and follow **best practices**.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
