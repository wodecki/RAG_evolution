# RAG Course Restructuring Plan

## Educational Principles Applied

**1. Progressive Complexity:**
- Start with basic RAG → advanced patterns
- Each folder builds on previous concepts
- Atomic scripts focus on one concept each

**2. Hands-on Learning:**
- Each script is immediately runnable
- Clear input/output examples
- Minimal dependencies per script

**3. Professional Patterns:**
- Mirror real-world RAG development progression
- Include production considerations early
- Show alternative approaches side-by-side

## Proposed Folder Structure

```
RAG_Course/
├── 01_basic_rag/
│   ├── minimal_rag.py           # Simplest RAG implementation
│   ├── with_chunking.py         # Add text splitting
│   └── README.md
├── 02_vector_stores/
│   ├── in_memory.py             # InMemoryVectorStore
│   ├── chroma_basic.py          # ChromaDB basics
│   ├── faiss_intro.py           # FAISS alternative
│   └── README.md
├── 03_document_loading/
│   ├── text_files.py            # Basic text loading
│   ├── pdf_loading.py           # PDF documents
│   ├── web_sources.py           # Web scraping
│   ├── multiple_files.py        # Directory loading
│   └── README.md
├── 04_advanced_retrieval/
│   ├── similarity_search.py     # Different search methods
│   ├── metadata_filtering.py    # Filtering with metadata
│   ├── hybrid_search.py         # Combining methods
│   └── README.md
├── 05_evaluation/
│   ├── generate_questions.py    # Question generation
│   ├── ground_truth.py          # Creating ground truth
│   ├── basic_metrics.py         # Simple evaluation
│   ├── advanced_eval.py         # Complex evaluation
│   └── README.md
├── 06_observability/
│   ├── basic_logging.py         # Simple logging
│   ├── phoenix_tracing.py       # Arize Phoenix
│   ├── performance_monitoring.py
│   └── README.md
├── 07_production/
│   ├── error_handling.py        # Robust error handling
│   ├── caching.py              # Response caching
│   ├── streaming.py            # Streaming responses
│   └── README.md
├── data/
│   └── scientists_bios/        # Shared dataset
└── README.md                   # Main course overview
```

## Current Branch Mapping to New Structure

**Branch → Target Folder:**
- main/chroma → 01_basic_rag, 02_vector_stores
- faiss → 02_vector_stores
- loaders/file_types → 03_document_loading
- loaders/sources → 03_document_loading
- multiple_files → 03_document_loading
- evaluation → 05_evaluation
- Arize_Phoenix → 06_observability
- indexing → 04_advanced_retrieval

## File Structure Per Folder

**Each folder contains:**
- **2-4 focused Python scripts** (max 100 lines each)
- **README.md** with:
  - Learning objectives
  - Prerequisites from previous modules
  - Concepts introduced
  - Code explanations
  - Next steps

**Script naming convention:**
- Use descriptive, action-oriented names
- Number only when order matters within a folder
- Keep filenames under 20 characters

## Sample README Template

```markdown
# Module X: [Topic]

## Learning Objectives
- Understand [concept 1]
- Implement [skill 1]
- Compare [alternatives]

## Prerequisites
- Completed Module [X-1]
- Understanding of [prerequisite concept]

## Scripts in This Module
1. `script1.py` - [Brief description]
2. `script2.py` - [Brief description]

## Key Concepts
- **[Concept]**: Definition and importance
- **[Pattern]**: When and why to use

## Running the Code
```bash
python script1.py
```

## What's Next
Module [X+1] will introduce...
```

## Migration Strategy

1. **Extract 01_basic_rag** from main branch
2. **Split 02_vector_stores** from chroma/faiss branches
3. **Combine loaders** into 03_document_loading
4. **Build 04_advanced_retrieval** from indexing branch
5. **Build 05_evaluation** from evaluation branch
6. **Extract 06_observability** from Arize_Phoenix branch
7. **Create 07_production** from production patterns across branches

## Notes

- Start with 01_basic_rag (foundations covered in previous course)
- Focus on atomic, runnable scripts
- Each module builds on previous concepts
- Mirror professional RAG development workflow