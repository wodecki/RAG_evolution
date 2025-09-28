"""
GraphRAG System Implementation
=============================

Advanced RAG system using Neo4j knowledge graph and Cypher queries
for structured data retrieval and reasoning.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import json
import time
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
import logging

from utils.models import QueryResult
from utils.query_translator import CypherQueryTranslator
from utils.graph_schema import GraphSchema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGSystem:
    """Advanced RAG system using Neo4j knowledge graph."""

    def __init__(self, uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "password123"):
        """Initialize the GraphRAG system."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.query_translator = CypherQueryTranslator()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()

    def execute_cypher_query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = []
                for record in result:
                    records.append(dict(record))
                return records
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            logger.error(f"Query: {cypher_query}")
            return []

    def format_graph_results(self, results: List[Dict[str, Any]], query_type: str = "unknown") -> str:
        """Format graph query results into readable text."""
        if not results:
            return "No results found."

        # Determine formatting based on result structure
        if len(results) == 1 and len(results[0]) == 1:
            # Single value result (count, average, etc.)
            key, value = list(results[0].items())[0]
            if isinstance(value, float):
                return f"Result: {value:.2f}"
            else:
                return f"Result: {value}"

        # Multiple rows with multiple columns
        formatted_lines = []
        for i, record in enumerate(results):
            if i == 0:
                # Add header
                headers = list(record.keys())
                formatted_lines.append(" | ".join(headers))
                formatted_lines.append("-" * (len(" | ".join(headers))))

            # Format values
            values = []
            for key, value in record.items():
                if isinstance(value, float):
                    values.append(f"{value:.2f}")
                elif isinstance(value, list):
                    values.append(", ".join(map(str, value)))
                else:
                    values.append(str(value))

            formatted_lines.append(" | ".join(values))

            # Limit output for readability
            if i >= 19:  # Show first 20 results
                remaining = len(results) - 20
                if remaining > 0:
                    formatted_lines.append(f"... and {remaining} more results")
                break

        return "\n".join(formatted_lines)

    def generate_contextual_answer(self, question: str, graph_results: str, cypher_query: str) -> str:
        """Generate a natural language answer using graph results."""
        prompt = f"""
You are an AI assistant helping with programmer staffing queries. You have access to a knowledge graph database and have executed a query to retrieve specific information.

Original Question: {question}

Cypher Query Executed:
{cypher_query}

Query Results:
{graph_results}

Instructions:
1. Provide a direct, comprehensive answer to the question using the query results
2. Include specific numbers, names, and details from the results
3. If the results show tables or lists, summarize the key findings
4. Maintain a professional, helpful tone
5. If the results are empty or incomplete, acknowledge this clearly
6. For counting questions, state the exact count
7. For listing questions, provide the list in a clear format
8. For comparison questions, highlight the key differences or rankings

Answer:"""

        response = self.llm.invoke(prompt)
        return response.content

    def query(self, question: str) -> QueryResult:
        """Process a query through the GraphRAG system."""
        start_time = time.time()

        try:
            # Translate natural language to Cypher
            cypher_query = self.query_translator.translate_to_cypher(question)

            # Validate the query
            if not self.query_translator.validate_cypher_query(cypher_query):
                raise Exception("Generated Cypher query failed validation")

            # Execute the query
            graph_results = self.execute_cypher_query(cypher_query)

            # Format results
            formatted_results = self.format_graph_results(graph_results)

            # Generate natural language answer
            answer = self.generate_contextual_answer(question, formatted_results, cypher_query)

            execution_time = time.time() - start_time

            # Create result
            result = QueryResult(
                query=question,
                answer=answer,
                source_type="graph_rag",
                context=[formatted_results],
                cypher_query=cypher_query,
                execution_time=execution_time,
                confidence_score=None
            )

            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")

            # Fallback response
            execution_time = time.time() - start_time
            return QueryResult(
                query=question,
                answer=f"Sorry, I encountered an error processing your query: {str(e)}",
                source_type="graph_rag",
                context=[],
                cypher_query=cypher_query if 'cypher_query' in locals() else None,
                execution_time=execution_time,
                confidence_score=None
            )

    def get_database_overview(self) -> Dict[str, Any]:
        """Get an overview of the database contents."""
        overview_queries = {
            "total_programmers": "MATCH (p:Programmer) RETURN count(p) as count",
            "total_skills": "MATCH (s:Skill) RETURN count(s) as count",
            "total_projects": "MATCH (pr:Project) RETURN count(pr) as count",
            "total_certifications": "MATCH (c:Certification) RETURN count(c) as count",
            "skill_categories": """
                MATCH (s:Skill)
                RETURN s.category as category, count(s) as skill_count
                ORDER BY skill_count DESC
            """,
            "top_skills": """
                MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill)
                RETURN s.name as skill, count(p) as programmer_count
                ORDER BY programmer_count DESC
                LIMIT 10
            """,
            "location_distribution": """
                MATCH (p:Programmer)
                RETURN p.location as location, count(p) as programmer_count
                ORDER BY programmer_count DESC
                LIMIT 10
            """,
            "certification_providers": """
                MATCH (c:Certification)
                RETURN c.provider as provider, count(c) as cert_count
                ORDER BY cert_count DESC
            """
        }

        overview = {}
        with self.driver.session() as session:
            for key, query in overview_queries.items():
                try:
                    result = session.run(query)
                    records = [dict(record) for record in result]
                    overview[key] = records
                except Exception as e:
                    logger.error(f"Error executing overview query {key}: {e}")
                    overview[key] = []

        return overview

    def demonstrate_graph_advantages(self) -> List[Dict[str, Any]]:
        """Demonstrate queries that show GraphRAG advantages over naive RAG."""
        demonstration_queries = [
            {
                "title": "Complex Filtering Query",
                "description": "Find Python developers with 4+ proficiency, available now, with AWS certification",
                "query": """
                MATCH (p:Programmer)-[hs:HAS_SKILL]->(s:Skill {name: 'Python'})
                MATCH (p)-[:HAS_CERTIFICATION]->(c:Certification)
                WHERE hs.proficiency >= 4
                  AND c.name CONTAINS 'AWS'
                  AND (p.availability_start <= date() OR p.availability_start IS NULL)
                RETURN p.name, p.location, p.hourly_rate, hs.proficiency, c.name as certification
                ORDER BY hs.proficiency DESC, p.hourly_rate ASC
                """,
                "advantage": "Precise multi-criteria filtering with exact proficiency levels"
            },
            {
                "title": "Aggregation Query",
                "description": "Average project count and hourly rate by skill category",
                "query": """
                MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill)
                MATCH (p)-[:WORKED_ON]->(pr:Project)
                WITH s.category as category, p, count(pr) as project_count
                RETURN category,
                       avg(project_count) as avg_projects,
                       avg(p.hourly_rate) as avg_hourly_rate,
                       count(DISTINCT p) as programmer_count
                ORDER BY avg_hourly_rate DESC
                """,
                "advantage": "Complex aggregations across multiple node types"
            },
            {
                "title": "Multi-hop Relationship Query",
                "description": "Find Python developers who worked with AWS-certified team members",
                "query": """
                MATCH (p1:Programmer)-[:HAS_SKILL]->(s:Skill {name: 'Python'})
                MATCH (p1)-[:WORKED_WITH]->(p2:Programmer)
                MATCH (p2)-[:HAS_CERTIFICATION]->(c:Certification)
                WHERE c.name CONTAINS 'AWS'
                RETURN DISTINCT p1.name as python_dev,
                       p1.location,
                       p2.name as aws_certified_colleague,
                       c.name as certification
                ORDER BY p1.name
                """,
                "advantage": "Multi-hop reasoning impossible with vector search"
            },
            {
                "title": "Temporal Reasoning Query",
                "description": "Developers becoming available within 90 days with specific skills",
                "query": """
                MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill)
                WHERE p.availability_start > date()
                  AND p.availability_start <= date() + duration({days: 90})
                  AND s.name IN ['React', 'Node.js', 'MongoDB']
                RETURN p.name,
                       p.availability_start,
                       collect(s.name) as skills,
                       p.hourly_rate
                ORDER BY p.availability_start
                """,
                "advantage": "Precise temporal reasoning with date arithmetic"
            },
            {
                "title": "Ranking and Sorting Query",
                "description": "Top 5 most collaborative developers (worked with most people)",
                "query": """
                MATCH (p:Programmer)-[w:WORKED_WITH]->()
                WITH p, sum(w.collaboration_count) as total_collaborations
                RETURN p.name,
                       p.location,
                       total_collaborations,
                       p.hourly_rate
                ORDER BY total_collaborations DESC
                LIMIT 5
                """,
                "advantage": "Exact ranking based on quantitative relationship properties"
            }
        ]

        results = []
        for demo in demonstration_queries:
            try:
                query_results = self.execute_cypher_query(demo["query"])
                formatted_results = self.format_graph_results(query_results)

                results.append({
                    "title": demo["title"],
                    "description": demo["description"],
                    "cypher_query": demo["query"],
                    "results": formatted_results,
                    "advantage": demo["advantage"],
                    "result_count": len(query_results)
                })
            except Exception as e:
                logger.error(f"Error in demonstration query {demo['title']}: {e}")

        return results

def test_graph_rag_system():
    """Test the GraphRAG system with sample queries."""
    test_queries = [
        "How many Python developers are available?",
        "List developers with AWS certifications",
        "What is the average hourly rate for React developers?",
        "Find developers who have worked on fintech projects",
        "Which skills are most common among our programmers?",
        "Who are the most experienced JavaScript developers?",
        "What projects require machine learning skills?",
        "Find developers available for immediate start",
        "Show me the top 5 developers by number of projects completed",
        "What is the collaboration network around Python developers?"
    ]

    print("\n" + "="*60)
    print("TESTING GRAPHRAG SYSTEM")
    print("="*60)

    graph_rag = GraphRAGSystem()

    # Get database overview
    print("Database Overview:")
    print("-" * 30)
    overview = graph_rag.get_database_overview()

    for key, data in overview.items():
        if isinstance(data, list) and data:
            if key in ["total_programmers", "total_skills", "total_projects", "total_certifications"]:
                print(f"{key.replace('_', ' ').title()}: {data[0]['count']}")

    print(f"\nTesting {len(test_queries)} queries...")
    print("-" * 60)

    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)

        result = graph_rag.query(query)
        results.append(result)

        print(f"Cypher Query:\n{result.cypher_query}")
        print(f"\nAnswer: {result.answer}")
        print(f"Execution time: {result.execution_time:.2f}s")

    # Demonstrate graph advantages
    print("\n" + "="*60)
    print("GRAPHRAG ADVANTAGES DEMONSTRATION")
    print("="*60)

    demonstrations = graph_rag.demonstrate_graph_advantages()
    for demo in demonstrations:
        print(f"\n{demo['title']}")
        print("-" * len(demo['title']))
        print(f"Description: {demo['description']}")
        print(f"Advantage: {demo['advantage']}")
        print(f"Results ({demo['result_count']} rows):")
        print(demo['results'][:500] + "..." if len(demo['results']) > 500 else demo['results'])

    # Save results
    def json_serializer(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    results_data = [result.model_dump() for result in results]
    with open("results/graph_rag_results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=json_serializer)

    graph_rag.close()

    print("\n" + "="*60)
    print("✓ GraphRAG testing completed!")
    print("Results saved to: results/graph_rag_results.json")
    print("\nNext step: Run uv run python 5_compare_systems.py")

def main():
    """Main function for GraphRAG system."""
    print("GraphRAG System with Neo4j")
    print("=" * 30)

    # Ensure results directory exists
    import os
    os.makedirs("results", exist_ok=True)

    # Test the system
    test_graph_rag_system()

if __name__ == "__main__":
    main()