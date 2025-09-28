"""
GraphRAG Query System for CV Knowledge Graph
============================================

Demonstrates GraphRAG capabilities by querying the knowledge graph
built from PDF CVs using natural language queries.

Shows advantages of structured graph queries over traditional RAG.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
from typing import List, Dict, Any
import logging

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CVGraphRAGSystem:
    """GraphRAG system for querying CV knowledge graph."""

    def __init__(self):
        """Initialize the GraphRAG system."""
        self.setup_neo4j()
        self.setup_qa_chain()
        self.load_example_queries()

    def setup_neo4j(self):
        """Setup Neo4j connection."""
        try:
            self.graph = Neo4jGraph()
            logger.info("✓ Connected to Neo4j successfully")

            # Refresh schema for accurate query generation
            self.graph.refresh_schema()
            logger.info("✓ Graph schema refreshed")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def setup_qa_chain(self):
        """Setup the GraphCypherQA chain."""
        # Initialize LLM for query generation
        self.llm = ChatOpenAI(
            model="gpt-4o",  # Use more powerful model for query generation
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Create the GraphCypher QA chain
        self.qa_chain = GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            verbose=True,  # Show generated Cypher queries
            return_intermediate_steps=True,
            allow_dangerous_requests=True  # Allow DELETE operations for demo
        )

        logger.info("✓ GraphCypher QA chain initialized")

    def load_example_queries(self):
        """Load example queries that demonstrate GraphRAG capabilities."""
        self.example_queries = {
            "Basic Information": [
                "How many people are in the knowledge graph?",
                "What companies appear in the CVs?",
                "List all the programming languages mentioned.",
                "What certifications do people have?"
            ],

            "Skill-based Queries": [
                "Who has Python skills?",
                "Find all people with machine learning experience.",
                "Who has both AWS and Docker skills?",
                "List people with more than 5 years of JavaScript experience."
            ],

            "Company and Experience": [
                "Who worked at Google?",
                "Find people who worked at both Microsoft and Amazon.",
                "What projects used React technology?",
                "Who has experience in the finance industry?"
            ],

            "Multi-hop Reasoning": [
                "Find colleagues who worked at the same companies.",
                "Who are potential team members for a Python + AWS project?",
                "Find people who have skills in common.",
                "What technologies are commonly used together?"
            ],

            "Complex Analytics": [
                "What is the most common skill combination?",
                "Which companies hire the most Python developers?",
                "Find the career progression patterns.",
                "What skills should someone learn to work at tech companies?"
            ],

            "Recruitment Scenarios": [
                "Find candidates for a full-stack web development project.",
                "Who would be good for a machine learning team?",
                "Find senior developers with cloud experience.",
                "Match candidates to a React + Node.js + AWS project."
            ]
        }

    def query_graph(self, question: str) -> Dict[str, Any]:
        """Execute a natural language query against the graph.

        Args:
            question: Natural language question

        Returns:
            Dict containing query results and metadata
        """
        try:
            logger.info(f"Executing query: {question}")

            # Execute the query
            result = self.qa_chain.invoke({"query": question})

            # Extract components
            response = {
                "question": question,
                "answer": result.get("result", "No answer generated"),
                "cypher_query": result.get("intermediate_steps", [{}])[0].get("query", ""),
                "success": True
            }

            logger.info(f"✓ Query executed successfully")
            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "cypher_query": "",
                "success": False
            }

    def run_example_queries(self, category: str = None) -> List[Dict[str, Any]]:
        """Run example queries to demonstrate GraphRAG capabilities.

        Args:
            category: Optional category to filter queries

        Returns:
            List of query results
        """
        results = []

        categories_to_run = [category] if category else self.example_queries.keys()

        for cat in categories_to_run:
            if cat not in self.example_queries:
                logger.warning(f"Category '{cat}' not found")
                continue

            print(f"\n{'='*60}")
            print(f"Category: {cat}")
            print(f"{'='*60}")

            for question in self.example_queries[cat]:
                print(f"\n🔍 Query: {question}")
                print("-" * 40)

                result = self.query_graph(question)
                results.append(result)

                if result["success"]:
                    print(f"📊 Generated Cypher: {result['cypher_query']}")
                    print(f"💡 Answer: {result['answer']}")
                else:
                    print(f"❌ Error: {result['answer']}")

                print()

        return results

    def custom_query(self, question: str) -> None:
        """Execute a custom user query.

        Args:
            question: User's natural language question
        """
        print(f"\n🔍 Custom Query: {question}")
        print("-" * 50)

        result = self.query_graph(question)

        if result["success"]:
            print(f"📊 Generated Cypher: {result['cypher_query']}")
            print(f"💡 Answer: {result['answer']}")
        else:
            print(f"❌ Error: {result['answer']}")

    def validate_graph_content(self) -> None:
        """Validate that the graph has content for querying."""
        validation_queries = [
            ("Total nodes", "MATCH (n) RETURN count(n) as count"),
            ("Total relationships", "MATCH ()-[r]->() RETURN count(r) as count"),
            ("People count", "MATCH (p:Person) RETURN count(p) as count"),
            ("Companies count", "MATCH (c:Company) RETURN count(c) as count"),
            ("Skills count", "MATCH (s:Skill) RETURN count(s) as count")
        ]

        print("\n📊 Graph Validation")
        print("-" * 30)

        for description, query in validation_queries:
            try:
                result = self.graph.query(query)
                count = result[0]["count"] if result else 0
                print(f"{description}: {count}")
            except Exception as e:
                print(f"{description}: Error - {e}")

        # Check if we have enough data
        person_count = self.graph.query("MATCH (p:Person) RETURN count(p) as count")[0]["count"]

        if person_count == 0:
            print("\n⚠️  Warning: No Person nodes found in the graph!")
            print("Please run 2_cvs_to_knowledge_graph.py first to populate the graph.")
            return False

        return True

    def show_graph_schema(self) -> None:
        """Display the current graph schema."""
        print("\n📋 Graph Schema")
        print("-" * 20)
        print(self.graph.schema)

    def interactive_mode(self) -> None:
        """Start interactive query mode."""
        print("\n🎯 Interactive GraphRAG Query Mode")
        print("Type your questions or 'quit' to exit")
        print("-" * 40)

        while True:
            try:
                question = input("\n❓ Your question: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                self.custom_query(question)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function to demonstrate GraphRAG capabilities."""
    print("CV Knowledge Graph - GraphRAG Query System")
    print("=" * 50)

    try:
        # Initialize system
        system = CVGraphRAGSystem()

        # Validate graph content
        if not system.validate_graph_content():
            return

        # Show schema
        system.show_graph_schema()

        # Menu system
        while True:
            print("\n🎯 GraphRAG Demo Options:")
            print("1. Run basic example queries")
            print("2. Run skill-based queries")
            print("3. Run multi-hop reasoning queries")
            print("4. Run all example queries")
            print("5. Interactive query mode")
            print("6. Exit")

            choice = input("\nSelect option (1-6): ").strip()

            if choice == "1":
                system.run_example_queries("Basic Information")
            elif choice == "2":
                system.run_example_queries("Skill-based Queries")
            elif choice == "3":
                system.run_example_queries("Multi-hop Reasoning")
            elif choice == "4":
                system.run_example_queries()
            elif choice == "5":
                system.interactive_mode()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid option. Please select 1-6.")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()