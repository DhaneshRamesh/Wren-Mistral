# Wren Mistral AI Service

A modular, FastAPI-based backend service for AI pipeline orchestration powered by **Mistral 7B** (via `llama-cpp-python`) and **Haystack**, designed to support RAG pipelines and job skill recommendation use cases.

---

## 🚀 Features

* ⚙️ Modular pipeline components using Hamilton
* 🧠 Integrated Mistral 7B model (Q4 GGUF) via `llama-cpp-python`
* 🔍 Semantic search with Qdrant
* 🌐 REST API using FastAPI + OpenAPI Docs
* 🧪 Health check & versioned routes (`/v1`, `/dev`)
* ↻ Hot-reload for development mode
* 📊 Langfuse support for telemetry (can be disabled)
* 🛠️ Configurable with `config.yaml` (optional)

---

## 🗂️ Project Structure

```bash
wren-ai-service/
├── src/
│   ├── core/             # Pipelines and components
│   ├── models/           # Pydantic models
│   ├── providers/        # Mistral, Vector DB, etc.
│   ├── web/              # Routes, exceptions, API handlers
│   ├── config.py         # Settings (uses pydantic-settings)
│   ├── globals.py        # Global registries and factories
│   └── __main__.py       # Entry point (FastAPI app)
├── models/               # .gguf model files (e.g., mistral-7b)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 🐍 1. Create virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 📦 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install fastapi uvicorn orjson pyyaml httptools uvloop \
            llama-cpp-python qdrant-client openai \
            hamilton pydantic-settings toml \
            farm-haystack
```

---

### 📀 3. Download the Mistral 7B model

Run:

```bash
python download_model.py
```

This downloads `mistral-7b-instruct-v0.1.Q4_0.gguf` into the `models/` directory.

---

## 🏁 Run the Service

```bash
python -m src.__main__
```

Then open:

```
http://localhost:8000/docs
```

For interactive OpenAPI docs.

---

## �힧 Health Check

```
GET /health
→ { "status": "ok" }
```

---

## 📌 Notes

* If `pydantic` schema error for `DataFrame` occurs, ensure the BaseModel includes:

```python
class Config:
    arbitrary_types_allowed = True
```

* You can create `.env.dev` or `config.yaml` to override default config.

---

## 📜 License

MIT © Dhanesh Ramesh
