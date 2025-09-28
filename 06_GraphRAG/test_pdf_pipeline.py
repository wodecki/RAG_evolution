"""
Test Pipeline for PDF CV to Knowledge Graph
==========================================

Simple test script to validate the complete PDF CV to Knowledge Graph pipeline.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import asyncio
from pathlib import Path

def test_environment():
    """Test environment setup."""
    print("🔧 Testing Environment Setup")
    print("-" * 30)

    # Check required environment variables
    required_vars = ["OPENAI_API_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file")
        return False
    else:
        print("✅ All environment variables are set")
        return True

def test_dependencies():
    """Test required dependencies."""
    print("\n📦 Testing Dependencies")
    print("-" * 25)

    required_packages = [
        "reportlab",
        "unstructured",
        "langchain_experimental",
        "langchain_openai",
        "langchain_neo4j",
        "faker"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: uv sync")
        return False
    else:
        print("\n✅ All dependencies available")
        return True

def test_neo4j_connection():
    """Test Neo4j connection."""
    print("\n🗄️ Testing Neo4j Connection")
    print("-" * 30)

    try:
        from langchain_neo4j import Neo4jGraph
        graph = Neo4jGraph()

        # Simple test query
        result = graph.query("RETURN 1 as test")
        if result and result[0]["test"] == 1:
            print("✅ Neo4j connection successful")
            return True
        else:
            print("❌ Neo4j connection failed")
            return False

    except Exception as e:
        print(f"❌ Neo4j connection error: {e}")
        print("Make sure Neo4j is running: docker-compose up -d")
        return False

def test_openai_connection():
    """Test OpenAI API connection."""
    print("\n🤖 Testing OpenAI Connection")
    print("-" * 30)

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke("Say 'test successful'")

        if "test successful" in response.content.lower():
            print("✅ OpenAI API connection successful")
            return True
        else:
            print("❌ OpenAI API responded but content unexpected")
            return False

    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        print("Check your OPENAI_API_KEY in .env file")
        return False

async def test_pdf_generation():
    """Test PDF generation."""
    print("\n📄 Testing PDF Generation")
    print("-" * 25)

    try:
        # Import and run integrated data generator
        from generate_data import GraphRAGDataGenerator

        # Generate just 2 test CVs using integrated approach
        generator = GraphRAGDataGenerator()
        result = generator.generate_all_data(2)
        generated_files = result["cv_files"]

        if generated_files and len(generated_files) >= 2:
            print(f"✅ Generated {len(generated_files)} test PDFs")

            # Check if files exist
            for file_path in generated_files:
                if Path(file_path).exists():
                    file_size = Path(file_path).stat().st_size
                    print(f"  - {Path(file_path).name}: {file_size} bytes")
                else:
                    print(f"  - {Path(file_path).name}: File not found")
                    return False

            return True
        else:
            print("❌ PDF generation failed")
            return False

    except Exception as e:
        print(f"❌ PDF generation error: {e}")
        return False

async def test_knowledge_graph_extraction():
    """Test knowledge graph extraction."""
    print("\n🧠 Testing Knowledge Graph Extraction")
    print("-" * 40)

    try:
        # Import the CV knowledge graph builder
        import importlib.util
        spec = importlib.util.spec_from_file_location("cvs_to_knowledge_graph", "2_cvs_to_knowledge_graph.py")
        cvs_to_kg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cvs_to_kg_module)

        builder = cvs_to_kg_module.CVKnowledgeGraphBuilder()

        # Process the test CVs
        processed_count = await builder.process_all_cvs()

        if processed_count > 0:
            print(f"✅ Processed {processed_count} CVs successfully")

            # Validate graph content
            builder.validate_graph()
            return True
        else:
            print("❌ No CVs were processed successfully")
            return False

    except Exception as e:
        print(f"❌ Knowledge graph extraction error: {e}")
        return False

def test_graph_querying():
    """Test graph querying."""
    print("\n🔍 Testing Graph Querying")
    print("-" * 25)

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("query_knowledge_graph", "3_query_knowledge_graph.py")
        query_kg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(query_kg_module)

        system = query_kg_module.CVGraphRAGSystem()

        # Test a simple query
        test_question = "How many people are in the knowledge graph?"
        result = system.query_graph(test_question)

        if result["success"]:
            print("✅ GraphRAG query successful")
            print(f"  Question: {test_question}")
            print(f"  Answer: {result['answer']}")
            return True
        else:
            print("❌ GraphRAG query failed")
            print(f"  Error: {result['answer']}")
            return False

    except Exception as e:
        print(f"❌ Graph querying error: {e}")
        return False

async def run_full_pipeline_test():
    """Run the complete pipeline test."""
    print("🚀 PDF CV to Knowledge Graph Pipeline Test")
    print("=" * 50)

    tests = [
        ("Environment", test_environment),
        ("Dependencies", test_dependencies),
        ("Neo4j Connection", test_neo4j_connection),
        ("OpenAI Connection", test_openai_connection),
        ("PDF Generation", test_pdf_generation),
        ("Knowledge Graph Extraction", test_knowledge_graph_extraction),
        ("Graph Querying", test_graph_querying)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()

            results.append((test_name, success))

            if not success:
                print(f"\n❌ {test_name} failed. Stopping pipeline test.")
                break

        except Exception as e:
            print(f"\n❌ {test_name} error: {e}")
            results.append((test_name, False))
            break

    # Summary
    print("\n" + "=" * 50)
    print("🏁 Pipeline Test Summary")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<30} {status}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your PDF CV to Knowledge Graph pipeline is working!")
        print("\nNext steps:")
        print("1. Generate more CVs: uv run python 1_generate_pdfs.py")
        print("2. Try different queries: uv run python 3_query_knowledge_graph.py")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please fix the issues above.")

if __name__ == "__main__":
    asyncio.run(run_full_pipeline_test())