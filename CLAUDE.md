# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval Augmented Generation) research project that implements RAG primitives using LangChain, ChromaDB, and OpenAI. The project focuses on question generation, ground truth creation, and RAG evaluation using scientist biographies as the knowledge base.

## Key Commands

### Environment Setup
```bash
# Create virtual environment
uv venv

# Activate virtual environment
source ./venv/bin/activate

# Install/sync dependencies
uv sync

# Add new dependencies
uv add <package_name>
```

### Running Course Modules
```bash
# Module 1: Basic RAG
uv run python "01_basic_rag/1. minimal_rag.py"
uv run python "01_basic_rag/2. minimal_rag_wtih_chunking.py"

# Module 2: Vector Stores
uv run python "02_vector_stores/1_in_memory.py"
uv run python "02_vector_stores/2_chroma_basic.py"
uv run python "02_vector_stores/3_faiss_intro.py"

# Legacy scripts (use course modules instead)
uv run python main.py
uv run python read_from_chroma_db.py
```

### Dataset Generation
```bash
# Generate questions for documents
uv run python datasets/generate_questions_for_files.py

# Create question dataset
uv run python datasets/create_question_dataset.py

# Generate ground truth answers
uv run python datasets/generate_ground_truth_answers.py

# Generate multi-hop questions
uv run python datasets/generate_multi-hop_questions.py
```

## Architecture

### Core RAG Components
- **Vector Store**: ChromaDB with LangChain integration (persisted in `./chroma_langchain_db/`)
- **Embeddings**: OpenAI embeddings via `langchain-openai`
- **LLM**: GPT-4o-mini for question answering
- **Text Splitting**: RecursiveCharacterTextSplitter (1000 char chunks, 20 overlap)
- **Retrieval**: Standard similarity search retriever

### Course Module Structure
- `01_basic_rag/`: Introduction to RAG concepts with in-memory storage
- `02_vector_stores/`: Comparison of vector stores (InMemory, ChromaDB, FAISS)
- `datasets/scientists_bios/`: Source documents (scientist biographies)
- `main.py`: Legacy ChromaDB implementation (use course modules instead)
- `read_from_chroma_db.py`: Legacy query script
- Scripts are numbered for execution order: `1_script.py`, `2_script.py`, etc.

### RAG Chain Structure
```
{context: retriever, question: RunnablePassthrough}
| prompt
| llm
| StrOutputParser
```

### Data Flow
1. **Ingestion**: `main.py` loads documents from `datasets/scientists_bios/`
2. **Chunking**: Documents split into 1000-character chunks
3. **Embedding**: Chunks embedded using OpenAI embeddings
4. **Storage**: Embedded chunks stored in ChromaDB collection "scientists_bios"
5. **Retrieval**: Similarity search retrieves relevant chunks for questions
6. **Generation**: GPT-4o-mini generates answers using retrieved context

## Package Management

**ALWAYS use `uv` for all Python operations:**

```bash
# Install dependencies
uv sync

# Add new packages
uv add package_name

# Run any Python script
uv run python script_name.py
```

Core dependencies:
- `langchain` ecosystem (langchain-core, langchain-community, langchain-openai, langchain-chroma)
- `unstructured` for document processing

## Environment Variables

Requires OpenAI API key in environment or `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Important Development Rules

- **Always use `uv run python` for script execution**
- **Live demo format**: Scripts execute immediately when run (no `def main()` blocks)
- **Numbered execution order**: Within each module, scripts are numbered `1_`, `2_`, `3_` etc.
- **Helper functions only**: Use `def` only for technical helper functions, never for main execution
- **Interactive variables**: Scripts create accessible variables for live exploration
- In all the scripts *ALWAYS* add:\
from dotenv import load_dotenv
load_dotenv(override=True)