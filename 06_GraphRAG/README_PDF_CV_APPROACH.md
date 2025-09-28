# PDF CV to Knowledge Graph - GraphRAG Implementation

## Overview

This implementation demonstrates the conversion of PDF CVs to a knowledge graph using LangChain's LLMGraphTransformer and Neo4j, showcasing real-world document-to-knowledge-graph capabilities for GraphRAG.

## What's Different from Traditional Approach

Instead of using structured JSON data, this approach:

1. **Generates realistic PDF CVs** using reportlab
2. **Extracts unstructured text** from PDFs using the `unstructured` library
3. **Converts text to knowledge graph** using LangChain's LLMGraphTransformer
4. **Stores in Neo4j** for GraphRAG queries

This is much closer to real-world scenarios where you have unstructured documents that need to be converted to structured knowledge.

## Architecture

```
PDF CVs → Text Extraction → LLM Graph Transformer → Neo4j → GraphRAG Queries
```

## Key Components

### 1. PDF CV Generator (`1_generate_pdfs.py`)
- Creates realistic CV PDFs with professional formatting
- Uses Faker for synthetic but realistic data
- Includes skills, experience, education, certifications

### 2. Knowledge Graph Builder (`2_cvs_to_knowledge_graph.py`)
- Extracts text from PDFs using `unstructured`
- Defines CV-specific ontology (Person, Company, Skill, etc.)
- Uses LLMGraphTransformer for entity and relationship extraction
- Stores structured graph in Neo4j

### 3. GraphRAG Query System (`3_query_knowledge_graph.py`)
- Natural language querying using GraphCypherQAChain
- Demonstrates multi-hop reasoning capabilities
- Shows advantages over traditional RAG

## CV Ontology

### Node Types
- **Person**: Individual from CV
- **Company**: Organizations worked at
- **University**: Educational institutions
- **Skill**: Technical and soft skills
- **Technology**: Specific technologies used
- **Project**: Projects worked on
- **Certification**: Professional certifications
- **Location**: Geographic locations
- **JobTitle**: Positions held
- **Industry**: Industry sectors

### Relationship Types
- `Person -[WORKED_AT]-> Company`
- `Person -[HAS_SKILL]-> Skill`
- `Person -[STUDIED_AT]-> University`
- `Person -[EARNED]-> Certification`
- `Project -[USED_TECHNOLOGY]-> Technology`
- And more...

## Quick Start

### 1. Prerequisites
```bash
# Install dependencies
uv sync

# Start Neo4j (using Docker)
docker-compose up -d

# Set environment variables
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and Neo4j credentials
```

### 2. Generate PDF CVs
```bash
uv run python 1_generate_pdfs.py
```
This creates 10 realistic PDF CVs in `data/cvs_pdf/`

### 3. Build Knowledge Graph
```bash
uv run python 2_cvs_to_knowledge_graph.py
```
This extracts entities/relationships from PDFs and populates Neo4j

### 4. Query the Graph
```bash
uv run python 3_query_knowledge_graph.py
```
This starts the interactive GraphRAG query system

## Example Queries

### Basic Information
- "How many people are in the knowledge graph?"
- "What companies appear in the CVs?"
- "List all programming languages mentioned."

### Skill-based Queries
- "Who has Python skills?"
- "Find people with both AWS and Docker experience."
- "List people with more than 5 years of JavaScript experience."

### Multi-hop Reasoning
- "Find colleagues who worked at the same companies."
- "Who are potential team members for a Python + AWS project?"
- "What technologies are commonly used together?"

### Complex Analytics
- "What is the most common skill combination?"
- "Which companies hire the most Python developers?"
- "Find the career progression patterns."

## Advantages Over Traditional RAG

1. **Structured Queries**: Can ask "How many Python developers?" instead of hoping embeddings capture this
2. **Multi-hop Reasoning**: "Find people who worked at companies that use React"
3. **Filtering & Aggregation**: "Average years of experience for AWS developers"
4. **Relationship Discovery**: "Who are potential colleagues based on shared companies?"
5. **Temporal Queries**: "Show career progression over time"

## File Structure

```
06_GraphRAG/
├── 1_generate_pdfs.py              # PDF CV generator
├── 2_cvs_to_knowledge_graph.py     # Graph builder
├── 3_query_knowledge_graph.py      # Query system
├── data/
│   └── cvs_pdf/                    # Generated PDF CVs
├── docker-compose.yml              # Neo4j setup
├── README_PDF_CV_APPROACH.md       # This file
└── utils/                          # Utility modules
```

## Technical Details

### LLMGraphTransformer Configuration
- **Model**: GPT-4o-mini (cost-efficient for education)
- **Mode**: Tool-based (structured output)
- **Schema**: Strict mode enabled for consistency
- **Properties**: Extracts dates, experience levels, descriptions

### PDF Processing
- **Library**: `unstructured` (better than PyPDF for complex layouts)
- **Strategy**: Process entire document as single context (avoids chunking issues)
- **Text Extraction**: Combines all PDF elements into coherent text

### Neo4j Integration
- **Storage**: Uses `add_graph_documents` with base entity labels
- **Indexing**: Creates performance indexes automatically
- **Schema**: Flexible graph schema with proper relationship directions

## Troubleshooting

### Common Issues

**1. "No PDF files found"**
- Run `1_generate_pdfs.py` first
- Check `data/cvs_pdf/` directory exists

**2. "Failed to connect to Neo4j"**
- Ensure Neo4j is running: `docker-compose up -d`
- Check credentials in `.env` file

**3. "No text extracted from PDF"**
- Install tesseract for OCR: `brew install tesseract` (macOS)
- Check PDF file integrity

**4. "Graph is empty"**
- Run `2_cvs_to_knowledge_graph.py` first
- Check OpenAI API key in `.env`

## Educational Value

This implementation is perfect for teaching:

1. **Document Processing**: Real PDF handling vs. structured data
2. **LLM-based Extraction**: Modern NLP for information extraction
3. **Graph Databases**: Neo4j and graph thinking
4. **GraphRAG**: Advantages over vector-based RAG
5. **System Integration**: End-to-end AI pipeline

## Next Steps

- Add more complex CV formats (multi-page, different layouts)
- Implement entity resolution (same company, different names)
- Add temporal analysis capabilities
- Create comparison with traditional RAG system
- Add evaluation metrics for extraction quality

## Research Background

This implementation is based on:
- Tomaz Bratanic's "Building Knowledge Graphs with LLM Graph Transformer" (Medium, 2024)
- LangChain's official documentation on graph construction
- Neo4j's GraphRAG best practices

The approach demonstrates state-of-the-art (2025) methods for document-to-knowledge-graph conversion in educational settings.