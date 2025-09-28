# GraphRAG Setup Guide

This guide helps you set up and use the GraphRAG (Graph Retrieval Augmented Generation) educational system for programmer staffing scenarios.

## 🎯 Overview

The GraphRAG system demonstrates how graph databases can enhance RAG (Retrieval Augmented Generation) for complex, relationship-based queries that traditional vector search cannot handle effectively.

### What You'll Learn
- Setting up Neo4j with Docker for persistence
- Converting unstructured PDFs to knowledge graphs using LLMs
- Querying graph databases with natural language
- Comparing GraphRAG vs traditional RAG approaches

## 🛠️ Prerequisites

### Required Software
- **Python 3.11+** with `uv` package manager
- **Docker Desktop** (for Neo4j database)
- **OpenAI API Key** (for LLM processing)

### Required Python Packages
The system uses `uv` for dependency management. Key packages:
- `langchain-openai` - OpenAI integration
- `langchain-neo4j` - Neo4j graph database
- `langchain-experimental` - LLM graph transformers
- `neo4j` - Neo4j Python driver
- `unstructured` - PDF processing

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone or navigate to the project
cd 06_GraphRAG

# Install dependencies
uv sync

# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### 2. Run Interactive Setup
```bash
# Start the intelligent setup system
uv run python 0_setup.py
```

The setup will:
- Check prerequisites
- Start Neo4j with Docker if needed
- Detect existing data
- Guide you through initialization

### 3. Alternative Setup Commands
```bash
# Fresh installation (clears existing data)
uv run python 0_setup.py --fresh

# Continue with existing data
uv run python 0_setup.py --continue

# Check system status only
uv run python 0_setup.py --check

# Educational mode with explanations
uv run python 0_setup.py --learning
```

## 📊 System Status

### Quick Status Check
```bash
# Basic status
uv run python check_status.py

# Detailed analysis
uv run python check_status.py --detailed

# Show sample queries
uv run python check_status.py --samples
```

### What to Expect
- **Green indicators**: System ready
- **Yellow warnings**: System working but with issues
- **Red errors**: Action required

## 🔧 Docker Configuration

### Persistent Data
The system uses Docker volumes for data persistence:

```yaml
# docker-compose.yml creates these volumes:
volumes:
  neo4j-data:      # Database files
  neo4j-logs:      # Log files
  neo4j-import:    # Data import directory
  neo4j-plugins:   # Plugin storage
  neo4j-conf:      # Configuration files
```

### Docker Commands
```bash
# Start Neo4j
docker-compose up -d

# Stop Neo4j (data persists)
docker-compose down

# View logs
docker-compose logs neo4j

# Reset everything (destroys data)
docker-compose down -v
docker volume prune
```

## 📁 Project Structure

```
06_GraphRAG/
├── 0_setup.py                    # Intelligent setup system
├── check_status.py              # Quick status checker
├── docker-compose.yml           # Neo4j configuration
├── config.toml                  # System configuration
├── 1_generate_data.py           # Generate sample data
├── 2_data_to_knowledge_graph.py # Build graph from PDFs
├── 3_query_knowledge_graph.py  # Query the graph
├── 4_graph_rag_system.py       # Full GraphRAG implementation
├── 5_compare_systems.py        # Compare RAG approaches
├── utils/
│   ├── neo4j_utils.py          # Neo4j utilities
│   ├── graph_schema.py         # Graph schema management
│   └── models.py               # Data models
├── data/
│   ├── programmers/            # Generated CVs (PDF + JSON)
│   ├── projects/               # Project data
│   └── RFP/                    # RFP documents
└── results/                    # Query results and comparisons
```

## 🎓 Educational Workflow

### Phase 1: Data Generation
```bash
# Generate synthetic programmer CVs and project data
uv run python 1_generate_data.py
```
**Learning objective**: Understand how to create realistic test data for graph systems.

### Phase 2: Knowledge Graph Construction
```bash
# Convert PDFs to knowledge graph using LLM
uv run python 2_data_to_knowledge_graph.py
```
**Learning objective**: See how LLMs can extract structured relationships from unstructured text.

### Phase 3: Graph Querying
```bash
# Query the graph with natural language
uv run python 3_query_knowledge_graph.py
```
**Learning objective**: Experience how graph databases enable complex relationship queries.

### Phase 4: System Comparison
```bash
# Compare GraphRAG vs traditional RAG
uv run python 5_compare_systems.py
```
**Learning objective**: Understand when GraphRAG provides advantages over vector-based RAG.

## 🔍 Example Queries

The system can answer complex questions that traditional RAG cannot:

### Counting Queries
- "How many Python developers do we have?"
- "How many developers have AWS certifications?"

### Aggregation Queries
- "What's the average hourly rate for React developers?"
- "Which skills are most common among our programmers?"

### Relationship Queries
- "Find developers who worked with AWS-certified colleagues"
- "Show me Python developers who have fintech experience"

### Temporal Queries
- "Who is available for immediate start?"
- "Which developers will be free next month?"

## 🌐 Neo4j Browser Access

Once Neo4j is running, access the browser interface:

- **URL**: http://localhost:7474
- **Username**: `neo4j`
- **Password**: `password123`

### Useful Browser Queries
```cypher
// Overview of data
MATCH (n) RETURN labels(n)[0] as type, count(n) as count

// Python developers
MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill {name: 'Python'})
RETURN p.name, p.location, p.hourly_rate

// Collaboration networks
MATCH (p1:Programmer)-[w:WORKED_WITH]-(p2:Programmer)
RETURN p1.name, p2.name, w.collaboration_count
```

## 🐛 Troubleshooting

### Common Issues

#### Neo4j Connection Failed
```bash
# Check Docker status
docker ps

# Restart Neo4j
docker-compose restart neo4j

# Check logs
docker-compose logs neo4j
```

#### No Data in Database
```bash
# Check if data generation completed
ls -la data/programmers/

# Regenerate data
uv run python 0_setup.py --fresh
```

#### LLM Extraction Errors
- Verify `OPENAI_API_KEY` in `.env` file
- Check API key has sufficient credits
- Review rate limits if getting 429 errors

#### Memory Issues
- Adjust Neo4j memory settings in `docker-compose.yml`
- Reduce data generation parameters in `config.toml`

### Getting Help

1. **Check Status**: `uv run python check_status.py --detailed`
2. **Review Logs**: `docker-compose logs neo4j`
3. **Reset System**: `uv run python 0_setup.py --fresh`
4. **View Setup Help**: `uv run python 0_setup.py --help`

## 📚 Learning Resources

### Graph Database Concepts
- **Nodes**: Entities (Programmers, Skills, Projects)
- **Relationships**: Connections (HAS_SKILL, WORKED_ON)
- **Properties**: Attributes (proficiency, years_experience)

### Cypher Query Language
- `MATCH`: Find patterns in the graph
- `WHERE`: Filter results
- `RETURN`: Specify output
- `CREATE`: Add new data
- `MERGE`: Find or create patterns

### GraphRAG Advantages
- **Relationship traversal**: Multi-hop queries
- **Structural reasoning**: Team formation, skill gaps
- **Temporal queries**: Availability, project timelines
- **Aggregation**: Counting, averaging across relationships

## 🔄 Data Persistence

### What Persists
- ✅ Knowledge graph data (nodes, relationships)
- ✅ Neo4j configuration and indexes
- ✅ Generated PDFs and JSON files
- ✅ Docker volumes and containers

### What Doesn't Persist
- ❌ Query results and comparisons
- ❌ Temporary processing files
- ❌ Application logs

### Backup Strategy
```bash
# Backup volumes (automated backup coming soon)
docker run --rm \
  -v neo4j-data:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/neo4j-backup-$(date +%Y%m%d).tar.gz /data
```

## 🚀 Next Steps

Once you have the system running:

1. **Experiment with queries** in Neo4j Browser
2. **Modify data generation** parameters in `config.toml`
3. **Try different LLM models** in the configuration
4. **Add your own PDFs** to the data generation process
5. **Extend the schema** for new entity types

## 💡 Tips for Educators

- Use `--learning` mode for explanations
- Start with small datasets (3-5 programmers)
- Focus on one query type at a time
- Compare results with traditional search
- Encourage exploration in Neo4j Browser

---

**Need help?** Run `uv run python 0_setup.py` for interactive guidance or `uv run python check_status.py` for system status.