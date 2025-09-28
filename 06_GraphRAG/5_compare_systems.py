"""
System Comparison and Evaluation Framework
=========================================

Compares the performance of Naive RAG vs GraphRAG systems to demonstrate
the advantages of graph-based retrieval for structured queries.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import json
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime
import statistics

from utils.models import QueryResult
from utils.graph_schema import GraphSchema

class SystemComparator:
    """Compares Naive RAG and GraphRAG systems."""

    def __init__(self):
        """Initialize the comparator."""
        self.evaluation_queries = [
            {
                "id": "count_python_devs",
                "query": "How many Python developers are available?",
                "type": "counting",
                "expected_approach": "Count nodes with specific relationships",
                "graph_advantage": "Exact counting through graph traversal"
            },
            {
                "id": "aws_certified_count",
                "query": "How many developers have AWS certifications?",
                "type": "counting",
                "expected_approach": "Count programmers with AWS certifications",
                "graph_advantage": "Precise relationship-based counting"
            },
            {
                "id": "avg_react_rate",
                "query": "What is the average hourly rate for React developers?",
                "type": "aggregation",
                "expected_approach": "Calculate average from matching programmers",
                "graph_advantage": "Direct aggregation on filtered nodes"
            },
            {
                "id": "top_developers_projects",
                "query": "List the top 5 developers by number of completed projects",
                "type": "ranking",
                "expected_approach": "Count projects per developer and rank",
                "graph_advantage": "Relationship counting with precise sorting"
            },
            {
                "id": "fintech_experience",
                "query": "Find developers who have worked on fintech projects",
                "type": "filtering",
                "expected_approach": "Filter by project domain",
                "graph_advantage": "Graph traversal to find specific relationships"
            },
            {
                "id": "skill_distribution",
                "query": "Which skills are most common among our programmers?",
                "type": "aggregation",
                "expected_approach": "Group by skill and count",
                "graph_advantage": "Aggregate across skill relationships"
            },
            {
                "id": "senior_javascript_devs",
                "query": "Who are the most experienced JavaScript developers?",
                "type": "filtering_ranking",
                "expected_approach": "Filter by skill, sort by experience",
                "graph_advantage": "Multi-criteria filtering with relationship properties"
            },
            {
                "id": "ml_projects",
                "query": "What projects require machine learning skills?",
                "type": "filtering",
                "expected_approach": "Find projects with ML requirements",
                "graph_advantage": "Direct relationship traversal"
            },
            {
                "id": "immediate_availability",
                "query": "Find developers available for immediate start",
                "type": "filtering",
                "expected_approach": "Filter by availability date",
                "graph_advantage": "Temporal property filtering"
            },
            {
                "id": "collaboration_network",
                "query": "Find Python developers who worked with AWS-certified colleagues",
                "type": "multi_hop",
                "expected_approach": "Multi-hop graph traversal",
                "graph_advantage": "Complex relationship reasoning impossible for naive RAG"
            }
        ]

    def load_existing_results(self) -> Tuple[List[QueryResult], List[QueryResult]]:
        """Load existing results from both systems."""
        try:
            with open("results/naive_rag_results.json", "r") as f:
                naive_data = json.load(f)
                naive_results = [QueryResult.model_validate(item) for item in naive_data]
        except FileNotFoundError:
            print("❌ Naive RAG results not found. Run 3_naive_rag_baseline.py first.")
            naive_results = []

        try:
            with open("results/graph_rag_results.json", "r") as f:
                graph_data = json.load(f)
                graph_results = [QueryResult.model_validate(item) for item in graph_data]
        except FileNotFoundError:
            print("❌ GraphRAG results not found. Run 4_graph_rag_system.py first.")
            graph_results = []

        return naive_results, graph_results

    def analyze_query_capabilities(self, naive_results: List[QueryResult], graph_results: List[QueryResult]) -> Dict[str, Any]:
        """Analyze the capabilities of each system for different query types."""
        analysis = {
            "query_type_performance": {},
            "accuracy_assessment": {},
            "execution_time_comparison": {},
            "response_quality": {}
        }

        # Group results by query type
        for eval_query in self.evaluation_queries:
            query_text = eval_query["query"]
            query_type = eval_query["type"]

            # Find matching results
            naive_result = next((r for r in naive_results if r.query == query_text), None)
            graph_result = next((r for r in graph_results if r.query == query_text), None)

            if naive_result and graph_result:
                comparison = {
                    "query": query_text,
                    "type": query_type,
                    "naive_rag": {
                        "answer": naive_result.answer,
                        "execution_time": naive_result.execution_time,
                        "context_chunks": len(naive_result.context)
                    },
                    "graph_rag": {
                        "answer": graph_result.answer,
                        "execution_time": graph_result.execution_time,
                        "cypher_query": graph_result.cypher_query,
                        "context_chunks": len(graph_result.context)
                    },
                    "graph_advantage": eval_query["graph_advantage"]
                }

                analysis["query_type_performance"][eval_query["id"]] = comparison

        return analysis

    def assess_answer_quality(self, naive_results: List[QueryResult], graph_results: List[QueryResult]) -> Dict[str, Any]:
        """Assess the quality and accuracy of answers."""
        quality_metrics = {
            "specificity": {},
            "quantitative_accuracy": {},
            "completeness": {},
            "interpretability": {}
        }

        for eval_query in self.evaluation_queries:
            query_text = eval_query["query"]
            query_id = eval_query["id"]

            naive_result = next((r for r in naive_results if r.query == query_text), None)
            graph_result = next((r for r in graph_results if r.query == query_text), None)

            if naive_result and graph_result:
                # Specificity: Does the answer include specific numbers/names?
                naive_has_numbers = any(char.isdigit() for char in naive_result.answer)
                graph_has_numbers = any(char.isdigit() for char in graph_result.answer)

                # Quantitative analysis for counting/aggregation queries
                quantitative_assessment = "N/A"
                if eval_query["type"] in ["counting", "aggregation"]:
                    if graph_has_numbers and not naive_has_numbers:
                        quantitative_assessment = "GraphRAG provides specific numbers, NaiveRAG does not"
                    elif both_have_numbers := (graph_has_numbers and naive_has_numbers):
                        quantitative_assessment = "Both provide quantitative answers"
                    else:
                        quantitative_assessment = "Neither provides clear quantitative answers"

                quality_metrics["specificity"][query_id] = {
                    "naive_specific": naive_has_numbers,
                    "graph_specific": graph_has_numbers,
                    "advantage": "GraphRAG" if graph_has_numbers and not naive_has_numbers else "Equal" if both_have_numbers else "Neither"
                }

                quality_metrics["quantitative_accuracy"][query_id] = quantitative_assessment

                # Completeness: Length and detail of answers
                quality_metrics["completeness"][query_id] = {
                    "naive_length": len(naive_result.answer),
                    "graph_length": len(graph_result.answer),
                    "naive_context_sources": len(naive_result.context),
                    "graph_context_sources": len(graph_result.context)
                }

                # Interpretability: Can we understand how the answer was derived?
                quality_metrics["interpretability"][query_id] = {
                    "naive_interpretable": "semantic similarity" in naive_result.answer.lower(),
                    "graph_interpretable": graph_result.cypher_query is not None,
                    "cypher_provided": graph_result.cypher_query is not None
                }

        return quality_metrics

    def calculate_performance_metrics(self, naive_results: List[QueryResult], graph_results: List[QueryResult]) -> Dict[str, Any]:
        """Calculate performance metrics for both systems."""
        naive_times = [r.execution_time for r in naive_results]
        graph_times = [r.execution_time for r in graph_results]

        metrics = {
            "execution_time": {
                "naive_rag": {
                    "mean": statistics.mean(naive_times) if naive_times else 0,
                    "median": statistics.median(naive_times) if naive_times else 0,
                    "min": min(naive_times) if naive_times else 0,
                    "max": max(naive_times) if naive_times else 0,
                    "total_queries": len(naive_times)
                },
                "graph_rag": {
                    "mean": statistics.mean(graph_times) if graph_times else 0,
                    "median": statistics.median(graph_times) if graph_times else 0,
                    "min": min(graph_times) if graph_times else 0,
                    "max": max(graph_times) if graph_times else 0,
                    "total_queries": len(graph_times)
                }
            },
            "query_success_rate": {
                "naive_rag": len([r for r in naive_results if "error" not in r.answer.lower()]) / max(len(naive_results), 1),
                "graph_rag": len([r for r in graph_results if "error" not in r.answer.lower()]) / max(len(graph_results), 1)
            },
            "average_context_length": {
                "naive_rag": statistics.mean([len(r.context) for r in naive_results]) if naive_results else 0,
                "graph_rag": statistics.mean([len(r.context) for r in graph_results]) if graph_results else 0
            }
        }

        return metrics

    def generate_comparison_report(self, analysis: Dict[str, Any], quality_metrics: Dict[str, Any], performance_metrics: Dict[str, Any]) -> str:
        """Generate a comprehensive comparison report."""
        report_lines = []

        # Header
        report_lines.extend([
            "# GraphRAG vs Naive RAG Comparison Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            "This report compares the performance of Naive RAG (vector similarity) vs GraphRAG (knowledge graph)",
            "systems for structured queries in a programmer staffing scenario.",
            ""
        ])

        # Performance Overview
        naive_perf = performance_metrics["execution_time"]["naive_rag"]
        graph_perf = performance_metrics["execution_time"]["graph_rag"]

        report_lines.extend([
            "## Performance Overview",
            "",
            f"**Naive RAG:**",
            f"- Queries processed: {naive_perf['total_queries']}",
            f"- Average execution time: {naive_perf['mean']:.3f}s",
            f"- Success rate: {performance_metrics['query_success_rate']['naive_rag']:.1%}",
            "",
            f"**GraphRAG:**",
            f"- Queries processed: {graph_perf['total_queries']}",
            f"- Average execution time: {graph_perf['mean']:.3f}s",
            f"- Success rate: {performance_metrics['query_success_rate']['graph_rag']:.1%}",
            ""
        ])

        # Query Type Analysis
        report_lines.extend([
            "## Query Type Analysis",
            ""
        ])

        query_types = {}
        for query_id, comparison in analysis["query_type_performance"].items():
            query_type = comparison["type"]
            if query_type not in query_types:
                query_types[query_type] = []
            query_types[query_type].append(comparison)

        for query_type, comparisons in query_types.items():
            report_lines.extend([
                f"### {query_type.replace('_', ' ').title()} Queries",
                ""
            ])

            for comp in comparisons:
                report_lines.extend([
                    f"**Query:** {comp['query']}",
                    f"- GraphRAG Advantage: {comp['graph_advantage']}",
                    f"- Naive RAG Answer: {comp['naive_rag']['answer'][:150]}...",
                    f"- GraphRAG Answer: {comp['graph_rag']['answer'][:150]}...",
                    f"- Execution Time: Naive {comp['naive_rag']['execution_time']:.3f}s vs Graph {comp['graph_rag']['execution_time']:.3f}s",
                    ""
                ])

        # Quality Assessment
        report_lines.extend([
            "## Answer Quality Assessment",
            ""
        ])

        specific_advantages = sum(1 for metrics in quality_metrics["specificity"].values()
                                if metrics["advantage"] == "GraphRAG")
        total_queries = len(quality_metrics["specificity"])

        report_lines.extend([
            f"**Specificity:** GraphRAG provided more specific answers in {specific_advantages}/{total_queries} queries",
            "",
            f"**Interpretability:** GraphRAG provides Cypher queries for transparency, Naive RAG relies on semantic similarity",
            ""
        ])

        # Key Findings
        report_lines.extend([
            "## Key Findings",
            "",
            "### GraphRAG Advantages:",
            "1. **Exact Counting:** Provides precise counts through graph traversal",
            "2. **Complex Filtering:** Multi-criteria filtering with relationship properties",
            "3. **Aggregations:** Direct mathematical operations on graph data",
            "4. **Multi-hop Reasoning:** Complex relationship traversals impossible for vector search",
            "5. **Temporal Logic:** Precise date-based filtering and arithmetic",
            "6. **Transparency:** Cypher queries show exact reasoning path",
            "",
            "### Naive RAG Strengths:",
            "1. **Semantic Similarity:** Good for fuzzy text matching",
            "2. **Setup Simplicity:** Easier to implement for basic use cases",
            "3. **Flexibility:** Handles diverse query types without schema knowledge",
            "",
            "### Use Case Recommendations:",
            "",
            "**Use GraphRAG for:**",
            "- Structured data with clear relationships",
            "- Counting, aggregation, and ranking queries",
            "- Multi-criteria filtering",
            "- Temporal reasoning",
            "- Complex business logic",
            "",
            "**Use Naive RAG for:**",
            "- Unstructured text search",
            "- Semantic similarity queries",
            "- Simple question-answering",
            "- Rapid prototyping",
            ""
        ])

        return "\n".join(report_lines)

    def save_detailed_analysis(self, analysis: Dict[str, Any], quality_metrics: Dict[str, Any], performance_metrics: Dict[str, Any]):
        """Save detailed analysis data."""
        detailed_analysis = {
            "generation_timestamp": datetime.now().isoformat(),
            "query_type_performance": analysis["query_type_performance"],
            "quality_metrics": quality_metrics,
            "performance_metrics": performance_metrics,
            "evaluation_queries": self.evaluation_queries
        }

        with open("results/detailed_comparison_analysis.json", "w") as f:
            json.dump(detailed_analysis, f, indent=2, default=str)

def create_summary_dashboard():
    """Create a simple text-based dashboard of results."""
    print("\n" + "="*80)
    print("GRAPHRAG vs NAIVE RAG - SUMMARY DASHBOARD")
    print("="*80)

    try:
        # Load graph database stats
        graph = GraphSchema()
        stats = graph.get_database_stats()

        print("\n📊 Knowledge Graph Statistics:")
        print(f"   Programmers: {stats['programmers']:>6} | Skills: {stats['skills']:>6} | Projects: {stats['projects']:>6}")
        print(f"   Relationships: {stats['has_skill'] + stats['worked_on'] + stats['has_certification']:>6} total")

        graph.close()

    except Exception as e:
        print(f"\n❌ Could not load graph statistics: {e}")

    # Load comparison results
    try:
        with open("results/detailed_comparison_analysis.json", "r") as f:
            analysis = json.load(f)

        naive_perf = analysis["performance_metrics"]["execution_time"]["naive_rag"]
        graph_perf = analysis["performance_metrics"]["execution_time"]["graph_rag"]

        print("\n⚡ Performance Comparison:")
        print(f"   Naive RAG:  {naive_perf['mean']:.3f}s avg | {naive_perf['total_queries']} queries")
        print(f"   GraphRAG:   {graph_perf['mean']:.3f}s avg | {graph_perf['total_queries']} queries")

        # Count advantages
        graph_advantages = 0
        total_comparisons = len(analysis["quality_metrics"]["specificity"])

        for metrics in analysis["quality_metrics"]["specificity"].values():
            if metrics["advantage"] == "GraphRAG":
                graph_advantages += 1

        print(f"\n🎯 Answer Quality:")
        print(f"   GraphRAG more specific: {graph_advantages}/{total_comparisons} queries")
        print(f"   GraphRAG transparency: ✓ (Cypher queries) | Naive RAG: ✗ (black box)")

    except Exception as e:
        print(f"\n❌ Could not load analysis results: {e}")

    print("\n" + "="*80)

def main():
    """Main comparison function."""
    print("System Comparison and Evaluation")
    print("=" * 40)

    # Ensure results directory exists
    import os
    os.makedirs("results", exist_ok=True)

    comparator = SystemComparator()

    print("Loading results from both systems...")
    naive_results, graph_results = comparator.load_existing_results()

    if not naive_results:
        print("❌ No Naive RAG results found. Please run 3_naive_rag_baseline.py first.")
        return

    if not graph_results:
        print("❌ No GraphRAG results found. Please run 4_graph_rag_system.py first.")
        return

    print(f"✓ Loaded {len(naive_results)} Naive RAG results")
    print(f"✓ Loaded {len(graph_results)} GraphRAG results")

    print("\nAnalyzing query capabilities...")
    analysis = comparator.analyze_query_capabilities(naive_results, graph_results)

    print("Assessing answer quality...")
    quality_metrics = comparator.assess_answer_quality(naive_results, graph_results)

    print("Calculating performance metrics...")
    performance_metrics = comparator.calculate_performance_metrics(naive_results, graph_results)

    print("Generating comparison report...")
    report = comparator.generate_comparison_report(analysis, quality_metrics, performance_metrics)

    # Save report
    with open("results/comparison_report.md", "w") as f:
        f.write(report)

    # Save detailed analysis
    comparator.save_detailed_analysis(analysis, quality_metrics, performance_metrics)

    print("✓ Comparison completed!")
    print("\nFiles generated:")
    print("- results/comparison_report.md")
    print("- results/detailed_comparison_analysis.json")

    # Show summary dashboard
    create_summary_dashboard()

    print("\n" + "="*80)
    print("🎉 GraphRAG Implementation Complete!")
    print("="*80)
    print("\nKey Achievements:")
    print("✓ Neo4j knowledge graph with 50 programmers, skills, and projects")
    print("✓ Naive RAG baseline with ChromaDB vector search")
    print("✓ GraphRAG system with natural language to Cypher translation")
    print("✓ Comprehensive comparison demonstrating GraphRAG advantages")
    print("\nNext Steps:")
    print("• Review results/comparison_report.md for detailed analysis")
    print("• Explore Neo4j Browser at http://localhost:7474")
    print("• Experiment with custom queries in both systems")

if __name__ == "__main__":
    main()