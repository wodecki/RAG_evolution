"""
Setup and Environment Validation for GraphRAG System
===================================================

This script sets up the environment and validates that all required components
are working correctly.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import sys
import subprocess
import time
from neo4j import GraphDatabase
import chromadb
from langchain_openai import ChatOpenAI

def check_docker():
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Docker is installed")
            return True
        else:
            print("✗ Docker is not installed")
            return False
    except FileNotFoundError:
        print("✗ Docker is not installed")
        return False

def start_neo4j():
    """Start Neo4j container if not running."""
    try:
        # Check if container is already running
        result = subprocess.run(['docker', 'ps', '--filter', 'name=neo4j_graphrag', '--format', '{{.Names}}'],
                              capture_output=True, text=True)

        if 'neo4j_graphrag' in result.stdout:
            print("✓ Neo4j container is already running")
            return True

        # Start the container
        print("Starting Neo4j container...")
        result = subprocess.run(['docker-compose', 'up', '-d'],
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))

        if result.returncode == 0:
            print("✓ Neo4j container started successfully")
            print("Waiting for Neo4j to be ready...")
            time.sleep(10)  # Give Neo4j time to start
            return True
        else:
            print(f"✗ Failed to start Neo4j container: {result.stderr}")
            return False

    except Exception as e:
        print(f"✗ Error starting Neo4j: {e}")
        return False

def test_neo4j_connection():
    """Test connection to Neo4j database."""
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password123"))
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful' as message")
            message = result.single()["message"]
            print("✓ Neo4j connection successful")
            driver.close()
            return True
    except Exception as e:
        print(f"✗ Neo4j connection failed: {e}")
        return False

def test_openai_api():
    """Test OpenAI API key."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("✗ OPENAI_API_KEY not found in environment")
            return False

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke("Say 'API test successful'")

        if "successful" in response.content.lower():
            print("✓ OpenAI API connection successful")
            return True
        else:
            print("✗ Unexpected OpenAI API response")
            return False

    except Exception as e:
        print(f"✗ OpenAI API test failed: {e}")
        return False

def test_chromadb():
    """Test ChromaDB functionality."""
    try:
        client = chromadb.Client()
        collection = client.create_collection("test_collection")
        collection.add(
            documents=["Test document"],
            ids=["test_id"]
        )
        results = collection.query(query_texts=["Test"], n_results=1)
        client.delete_collection("test_collection")
        print("✓ ChromaDB working correctly")
        return True
    except Exception as e:
        print(f"✗ ChromaDB test failed: {e}")
        return False

def create_project_structure():
    """Ensure all required directories exist."""
    directories = [
        "data/programmers",
        "data/projects",
        "data/rfps",
        "utils",
        "results"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✓ Project directory structure created")
    return True

def main():
    """Main setup and validation function."""
    print("GraphRAG Environment Setup and Validation")
    print("=" * 50)

    all_checks_passed = True

    # Check components
    checks = [
        ("Docker", check_docker),
        ("Project Structure", create_project_structure),
        ("Neo4j Startup", start_neo4j),
        ("Neo4j Connection", test_neo4j_connection),
        ("OpenAI API", test_openai_api),
        ("ChromaDB", test_chromadb)
    ]

    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if not check_func():
            all_checks_passed = False

    print("\n" + "=" * 50)
    if all_checks_passed:
        print("✓ All setup checks passed! Environment is ready.")
        print("\nNext steps:")
        print("1. Run: uv run python 1_generate_data.py")
        print("2. Access Neo4j browser at: http://localhost:7474")
        print("   Username: neo4j, Password: password123")
    else:
        print("✗ Some setup checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()