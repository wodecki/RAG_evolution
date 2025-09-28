# GraphRAG Implementation: Programmer Staffing System

A comprehensive demonstration of **GraphRAG vs Naive RAG** using a realistic programmer staffing scenario. This project showcases how knowledge graphs enable structured queries that are impossible with traditional vector-based RAG systems.

## 🎯 Problem Addressed

Traditional RAG systems struggle with structured queries requiring:
1. **Counting**: "How many Python developers do we have?"
2. **Filtering**: "Find developers with AWS certifications available in Q2"
3. **Aggregation**: "What's the average hourly rate for React developers?"
4. **Sorting**: "List developers by project experience"
5. **Multi-hop reasoning**: "Find Python devs who worked with AWS-certified colleagues"

## 🏗️ Architecture

### Knowledge Graph Schema (Neo4j)
```
Nodes:
├── Programmer (id, name, location, hourly_rate, availability)
├── Skill (name, category)
├── Certification (name, provider)
├── Project (id, name, client, dates, status)
└── RFP (id, title, requirements)

Relationships:
├── (Programmer)-[HAS_SKILL {proficiency, years}]->(Skill)
├── (Programmer)-[HAS_CERTIFICATION {dates}]->(Certification)
├── (Programmer)-[WORKED_ON {role, dates}]->(Project)
├── (Programmer)-[WORKED_WITH {projects}]->(Programmer)
├── (Project)-[REQUIRES_SKILL {min_level}]->(Skill)
└── (RFP)-[RFP_REQUIRES_SKILL]->(Skill)
```

### System Components
- **Neo4j Database**: Knowledge graph storage with Docker setup
- **GraphRAG System**: Natural language → Cypher → Results → LLM
- **Naive RAG Baseline**: ChromaDB vector search for comparison
- **Query Translator**: Converts questions to Cypher queries
- **Evaluation Framework**: Side-by-side performance comparison

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Start Neo4j database
docker-compose up -d

# Run environment validation
uv run python 0_setup.py
```

### 2. Generate Data & Build Graph
```bash
# Generate 50 synthetic programmer profiles
uv run python 1_generate_data.py

# Populate Neo4j knowledge graph
uv run python 2_build_knowledge_graph.py
```

### 3. Run Comparisons
```bash
# Test Naive RAG baseline
uv run python 3_naive_rag_baseline.py

# Test GraphRAG system
uv run python 4_graph_rag_system.py

# Compare both systems
uv run python 5_compare_systems.py
```

## 📊 Key Demonstrations

### Exact Counting
**Query**: "How many Python developers are available?"

**Naive RAG**: Searches text chunks, estimates from context
**GraphRAG**:
```cypher
MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill {name: 'Python'})
WHERE p.availability_start <= date() OR p.availability_start IS NULL
RETURN count(p) as python_developers
```
*Result: Exact count with perfect accuracy*

### Multi-Criteria Filtering
**Query**: "Find senior Python developers with AWS certifications"

**Naive RAG**: Relies on semantic similarity across text chunks
**GraphRAG**:
```cypher
MATCH (p:Programmer)-[hs:HAS_SKILL]->(s:Skill {name: 'Python'})
MATCH (p)-[:HAS_CERTIFICATION]->(c:Certification)
WHERE hs.proficiency >= 4 AND c.name CONTAINS 'AWS'
RETURN p.name, p.location, hs.proficiency, c.name
ORDER BY hs.proficiency DESC
```
*Result: Precise filtering with relationship properties*

### Multi-Hop Reasoning
**Query**: "Find Python developers who worked with AWS-certified colleagues"

**Naive RAG**: ❌ Impossible - cannot traverse relationships
**GraphRAG**:
```cypher
MATCH (p1:Programmer)-[:HAS_SKILL]->(s:Skill {name: 'Python'})
MATCH (p1)-[:WORKED_WITH]->(p2:Programmer)
MATCH (p2)-[:HAS_CERTIFICATION]->(c:Certification)
WHERE c.name CONTAINS 'AWS'
RETURN p1.name, p2.name, c.name
```
*Result: Complex relationship reasoning*

## 📈 Performance Comparison

| Query Type | Naive RAG | GraphRAG | Advantage |
|------------|-----------|----------|-----------|
| **Counting** | Estimates from text | Exact graph traversal | ✅ GraphRAG |
| **Filtering** | Semantic similarity | Relationship properties | ✅ GraphRAG |
| **Aggregation** | Text-based approximation | Direct calculation | ✅ GraphRAG |
| **Sorting** | Limited by chunk ranking | Property-based sorting | ✅ GraphRAG |
| **Multi-hop** | ❌ Impossible | ✅ Native support | ✅ GraphRAG |
| **Transparency** | Black box similarity | Interpretable Cypher | ✅ GraphRAG |

## 🔍 Example Queries

### Business Intelligence Queries
```bash
# Resource planning
"How many developers will be available next quarter?"

# Skills gap analysis
"What skills are we missing for the new fintech RFP?"

# Team optimization
"Find the most collaborative developers for team leads"

# Market analysis
"What's the average rate for full-stack developers?"
```

### Complex Analytical Queries
```cypher
-- Skill proficiency distribution
MATCH (p:Programmer)-[hs:HAS_SKILL]->(s:Skill {name: 'Python'})
RETURN hs.proficiency, count(p) as developer_count

-- Collaboration network analysis
MATCH (p1:Programmer)-[w:WORKED_WITH]->(p2:Programmer)
RETURN p1.name, p2.name, w.collaboration_count
ORDER BY w.collaboration_count DESC

-- Temporal availability analysis
MATCH (p:Programmer)
WHERE p.availability_start > date()
RETURN p.availability_start.month as month, count(p) as available
ORDER BY month
```

## 📁 Project Structure

```
06_GraphRAG/
├── 0_setup.py                 # Environment validation
├── 1_generate_data.py          # Synthetic data generation
├── 2_build_knowledge_graph.py  # Graph population
├── 3_naive_rag_baseline.py     # Traditional RAG system
├── 4_graph_rag_system.py       # GraphRAG implementation
├── 5_compare_systems.py        # Performance comparison
├── docker-compose.yml          # Neo4j setup
├── test_queries.json          # Evaluation queries
├── utils/
│   ├── models.py              # Pydantic data models
│   ├── graph_schema.py        # Neo4j schema management
│   └── query_translator.py    # NL → Cypher translation
├── data/                      # Generated synthetic data
│   ├── programmers/           # 50 programmer profiles
│   ├── projects/              # 20 project descriptions
│   └── rfps/                  # 3 RFP documents
└── results/                   # Comparison results
    ├── comparison_report.md   # Detailed analysis
    ├── naive_rag_results.json
    ├── graph_rag_results.json
    └── detailed_comparison_analysis.json
```

## 🛠️ Technical Stack

- **Graph Database**: Neo4j 5.x with Docker
- **Vector Store**: ChromaDB (for baseline)
- **LLM**: OpenAI GPT-4o-mini
- **Frameworks**: LangChain, Pydantic
- **Language**: Python 3.10+
- **Package Manager**: uv

## 🎓 Educational Value

This implementation demonstrates:

1. **When to use GraphRAG**: Structured data with clear relationships
2. **Query translation**: Natural language → Cypher → Results
3. **Performance trade-offs**: Accuracy vs simplicity
4. **Real-world applications**: Staffing, inventory, recommendation systems
5. **System design**: Modular architecture with clear separation of concerns

## 🚀 Next Steps

1. **Explore Neo4j Browser**: http://localhost:7474 (neo4j/password123)
2. **Review Results**: Check `results/comparison_report.md`
3. **Experiment**: Try custom queries in both systems
4. **Extend**: Add new node types or relationships
5. **Scale**: Test with larger datasets

## 🤝 Use Cases Beyond Staffing

The patterns demonstrated here apply to:

- **E-commerce**: Product recommendations with complex filters
- **Healthcare**: Patient care coordination with medical relationships
- **Finance**: Risk analysis with entity relationships
- **Supply Chain**: Multi-hop dependency tracking
- **Social Networks**: Influence and connection analysis

## 📚 Key Learnings

1. **GraphRAG excels** at structured queries requiring precise relationships
2. **Naive RAG works better** for semantic text search and fuzzy matching
3. **Hybrid approaches** can combine both strengths
4. **Query complexity** determines the appropriate RAG architecture
5. **Transparency** in GraphRAG enables debugging and validation

---

**🎉 Congratulations!** You've successfully implemented and compared both RAG architectures. The knowledge graph approach demonstrates clear advantages for structured business queries while maintaining the flexibility to handle natural language inputs.