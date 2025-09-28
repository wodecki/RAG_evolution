# Rationale
Naive RAG can't be used when we need:
1. (metadata) filtering: "Show me all employment contracts singed in California after 2020"
2. Counting: "How many python programmers do we have?", "How many ongoing patent disputes involve Apple?"
3. Sorting: "List the most recent Python projects we realised?", "List the most recent Supreme Court decisions about antitrus law?"
4. Aggregation: "What's the avarege duration of a Python-dominated project?", "Whats the average duration of a trademark ligitation case?"

For that type of questions, we need GraphRAG

# The concept
Create a demo system that showcases the GraphRAG advantages over problems highlighted in # Rationale
We will need:
1. A source docs dataset, with data allowing as to ask questions like in # Rationale
2. RAG implementations:
   2.1. Naive RAG
   2.2. GraphRAG (LangChain + Neo4j)
3. A set of test questions with ground-truths generated separately by LLM like GPT-5
4. A simple test (no RAGAS) comparing the answers of Naive RAG and GraphRAG, displaying the advantage of GraphRAG over Naive RAG

# Potential use case
Imagine the programmers body leasing company: it leases programmers to customers IT projects.

inputs: 
1. a set of synthetically generated 50 programmers CVs, displaying their declarations of skills, appropriate certificates, and participation in projects (e.g. project title, customer, project goal, start, end, a role in a project).
2. a database of currently running and plannego projects, with programmers assignment
3. a set of 3 example new RFPs from customers
4. problem: answer the questions similar to those in # Rationale, plus a business case: which programmers can we assign to a new project, taking into account their skills and assignments to existing or planned projects? Whom do we miss and should look outside?

# Your task
1. Analyze critically this concept. Think deep: suggest improvements, or approve
2. Create a simple PRD for this concept
3. Suggest a very simple architecture and tech stack, with LangChain and Neo4j (Neo4j from Docker)
4. Design and implement the code

# Architecture Decision: Dynamic RFP Matching (2024-09-28)

## Problem Statement
Initial design proposed pre-computing match scores between programmers and RFPs during graph construction. This approach had limitations:
- Scores become stale when data changes
- Need to rebuild graph when new RFPs arrive
- Inflexible scoring algorithms

## Solution: Two-Phase System

### Phase 1: Static Knowledge Graph (CVs)
Build a stable foundation from programmer CVs:
- **Nodes**: Programmer, Skill, Certification, HistoricalProject
- **Relationships**: HAS_SKILL, HAS_CERTIFICATION, WORKED_ON, WORKED_WITH
- **Characteristics**:
  - Built once from PDF CVs using LLM extraction
  - Rarely changes (only when new CVs added)
  - Contains skills, certifications, historical experience

### Phase 2: Dynamic RFP Processing
Handle RFPs and project assignments dynamically:
- **Input**: RFP document + projects.yaml (current assignments)
- **Process**:
  1. Parse RFP requirements
  2. Load current project assignments from YAML
  3. Calculate real-time availability
  4. Compute match scores dynamically
  5. Return ranked candidates
- **Benefits**:
  - Always current availability
  - Flexible scoring algorithms
  - Support for what-if scenarios

## Implementation Flow
```python
# 1. Build static KG from CVs (one-time)
python 2_cvs_to_knowledge_graph.py

# 2. When new RFP arrives (dynamic)
rfp_matcher.process(
    rfp_file="new_rfp.pdf",
    projects_file="current_projects.yaml"
)
```

## Key Design Principles
1. **Separation of Concerns**: Static data (CVs) vs dynamic data (RFPs/assignments)
2. **Real-time Matching**: Compute scores at query time, not build time
3. **Flexibility**: Can adjust matching algorithms without rebuilding graph
4. **Auditability**: Track why certain matches were made at specific times

## Schema Extensions for Dynamic Matching

### Temporary Nodes (created from projects.yaml)
```cypher
(Assignment:Dynamic {
    programmer_id: string,
    project_id: string,
    start_date: date,
    end_date: date,
    allocation_percentage: integer
})
```

### Query Patterns
```cypher
// Find available programmers
MATCH (p:Programmer)
OPTIONAL MATCH (a:Assignment:Dynamic {programmer_id: p.id})
WHERE a.start_date <= $rfp_start AND a.end_date >= $rfp_start
WITH p, sum(a.allocation_percentage) as allocated
WHERE (100 - coalesce(allocated, 0)) >= $required_allocation
RETURN p
```

## Advantages Over Static Scoring
- **No stale data**: Availability always reflects current state
- **Flexible matching**: Can experiment with different scoring algorithms
- **What-if analysis**: Can simulate different assignment scenarios
- **Scalability**: Process thousands of CVs once, handle RFPs on-demand
