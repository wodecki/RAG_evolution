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

### Running the Application
```bash
# Main RAG application (creates and populates ChromaDB)
python main.py

# Query existing ChromaDB
python read_from_chroma_db.py
```

### Dataset Generation
```bash
# Generate questions for documents
python datasets/generate_questions_for_files.py

# Create question dataset
python datasets/create_question_dataset.py

# Generate ground truth answers
python datasets/generate_ground_truth_answers.py

# Generate multi-hop questions
python datasets/generate_multi-hop_questions.py
```

## Architecture

### Core RAG Components
- **Vector Store**: ChromaDB with LangChain integration (persisted in `./chroma_langchain_db/`)
- **Embeddings**: OpenAI embeddings via `langchain-openai`
- **LLM**: GPT-4o-mini for question answering
- **Text Splitting**: RecursiveCharacterTextSplitter (1000 char chunks, 20 overlap)
- **Retrieval**: Standard similarity search retriever

### Key Files
- `main.py`: Creates ChromaDB from scientist biographies, runs sample questions
- `read_from_chroma_db.py`: Queries existing ChromaDB without recreation
- `datasets/`: Question generation and ground truth creation scripts
- `datasets/scientists_bios/`: Source documents (scientist biographies)
- `chroma_langchain_db/`: Persisted ChromaDB vector store
- `datasets/questions_with_answers.csv`: Generated question-answer pairs

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

## Dependencies

Uses `uv` for dependency management. Core dependencies:
- `langchain` ecosystem (langchain-core, langchain-community, langchain-openai, langchain-chroma)
- `unstructured` for document processing

## Environment Variables

Requires OpenAI API key in environment or `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```