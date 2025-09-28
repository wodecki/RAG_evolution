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
