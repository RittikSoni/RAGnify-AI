from fastapi import FastAPI
from pydantic import BaseModel
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage

app = FastAPI()

class Query(BaseModel):
    question: str

DATA_INDEX = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """
You are a helpful assistant that answers user questions using only the information provided in the context passages from our FAQ.

🔒 **Strict Instructions:**
- Answer **only** using the given context.
- Do **not** use external knowledge or make up any information.
- If the context contains a **closely related** or **similar** entry, suggest the correct topic and answer using that context.
- If the user's question is a **greeting** (e.g., hi, hello, hey, thanks), respond appropriately and politely.
- If the user's input is a **misspelling or typo** of a known question in the context, correct it and answer accordingly.

🚫 **Fallback Rule:**
If the question:
- Has **no relevant answer in the context**, and
- Is **not a greeting**, and
- Is **not a typo of a known FAQ entry**,

Then respond **only** with this exact fallback message:
"I’m sorry, I don’t know that. Please contact customer support for help."
"""

@lru_cache(maxsize=1)
def load_resources():
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    index = FAISS.load_local(
        DATA_INDEX,
        embedder,
        allow_dangerous_deserialization=True
    )
    llm = ChatOllama(model="llama3.1:8b", temperature=0.0)
    return index, llm

@app.post("/chat")
async def chat(query: Query):
    index, llm = load_resources()
    user_q = query.question.strip()

    # 1️⃣ retrieve top-K chunks
    docs = index.similarity_search(user_q, k=3)

    # 2️⃣ assemble context with metadata headers
    context = "\n\n".join(
        f"[{d.metadata['source']}]\n{d.page_content}"
        for d in docs
    )

    # 3️⃣ build LLM messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {user_q}")
    ]

    # 4️⃣ one-shot call
    resp = llm(messages)

    # 5️⃣ return answer + where it came from
    return {
        "answer": resp.content,
        "sources": [ d.metadata["source"] for d in docs ]
    }
