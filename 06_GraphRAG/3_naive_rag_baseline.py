"""
Naive RAG Baseline Implementation
================================

Traditional RAG system using ChromaDB for vector storage and semantic search.
This serves as the baseline to compare against GraphRAG performance.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import json
import time
from glob import glob
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

from utils.models import ProgrammerProfile, Project, RFP, QueryResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NaiveRAGSystem:
    """Traditional RAG system using ChromaDB and semantic search."""

    def __init__(self, collection_name: str = "programmer_documents"):
        """Initialize the naive RAG system."""
        self.collection_name = collection_name

        # Initialize components
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", ", ", " "]
        )

        # Initialize ChromaDB
        self.client = chromadb.Client()
        self._setup_collection()

    def _setup_collection(self):
        """Set up ChromaDB collection."""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except:
            # Create new collection
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"Created new collection: {self.collection_name}")

    def load_and_process_documents(self) -> List[Document]:
        """Load and process all documents into chunks."""
        documents = []

        # Load programmer profiles
        programmer_files = glob("data/programmers/*.json")
        for file_path in programmer_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    programmer = ProgrammerProfile.model_validate(data)
                    text = self._programmer_to_text(programmer)

                    doc = Document(
                        page_content=text,
                        metadata={
                            "type": "programmer",
                            "id": programmer.id,
                            "name": programmer.name,
                            "location": programmer.location,
                            "source": file_path
                        }
                    )
                    documents.append(doc)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        # Load projects
        project_files = glob("data/projects/*.json")
        for file_path in project_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    project = Project.model_validate(data)
                    text = self._project_to_text(project)

                    doc = Document(
                        page_content=text,
                        metadata={
                            "type": "project",
                            "id": project.id,
                            "name": project.name,
                            "client": project.client,
                            "source": file_path
                        }
                    )
                    documents.append(doc)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        # Load RFPs
        rfp_files = glob("data/rfps/*.json")
        for file_path in rfp_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    rfp = RFP.model_validate(data)
                    text = self._rfp_to_text(rfp)

                    doc = Document(
                        page_content=text,
                        metadata={
                            "type": "rfp",
                            "id": rfp.id,
                            "title": rfp.title,
                            "client": rfp.client,
                            "source": file_path
                        }
                    )
                    documents.append(doc)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def _programmer_to_text(self, programmer: ProgrammerProfile) -> str:
        """Convert programmer profile to searchable text."""
        skills_text = ", ".join([f"{skill.name} (proficiency: {skill.proficiency}, years: {skill.years_experience})"
                                for skill in programmer.skills])

        certs_text = ", ".join([f"{cert.name} from {cert.provider} (obtained: {cert.obtained_date})"
                               for cert in programmer.certifications])

        projects_text = "\n".join([
            f"Project: {exp.project_name} at {exp.client} as {exp.role} "
            f"({exp.start_date} to {exp.end_date or 'present'}). "
            f"Technologies: {', '.join(exp.technologies_used)}. {exp.description}"
            for exp in programmer.project_experience
        ])

        text = f"""
Programmer Profile: {programmer.name}
ID: {programmer.id}
Email: {programmer.email}
Location: {programmer.location}
Hourly Rate: ${programmer.hourly_rate}
Bio: {programmer.bio}

Skills: {skills_text}

Certifications: {certs_text}

Project Experience:
{projects_text}

Availability: {programmer.availability_start or 'To be determined'}
LinkedIn: {programmer.linkedin_url}
GitHub: {programmer.github_url}
        """.strip()

        return text

    def _project_to_text(self, project: Project) -> str:
        """Convert project to searchable text."""
        requirements_text = ", ".join([
            f"{req.skill_name} (min proficiency: {req.min_proficiency}, min years: {req.min_years}, "
            f"mandatory: {req.is_mandatory})"
            for req in project.requirements
        ])

        assigned_text = ", ".join(project.assigned_programmers) if project.assigned_programmers else "None assigned"

        text = f"""
Project: {project.name}
ID: {project.id}
Client: {project.client}
Status: {project.status}
Start Date: {project.start_date}
End Date: {project.end_date or 'Ongoing'}
Duration: {project.estimated_duration_months} months
Team Size: {project.team_size}
Budget: ${project.budget or 'Not specified'}

Description: {project.description}

Skill Requirements: {requirements_text}

Assigned Programmers: {assigned_text}
        """.strip()

        return text

    def _rfp_to_text(self, rfp: RFP) -> str:
        """Convert RFP to searchable text."""
        requirements_text = ", ".join([
            f"{req.skill_name} (min proficiency: {req.min_proficiency}, min years: {req.min_years}, "
            f"mandatory: {req.is_mandatory})"
            for req in rfp.requirements
        ])

        text = f"""
RFP: {rfp.title}
ID: {rfp.id}
Client: {rfp.client}
Project Type: {rfp.project_type}
Duration: {rfp.duration_months} months
Team Size: {rfp.team_size}
Budget Range: {rfp.budget_range or 'Not specified'}
Start Date: {rfp.start_date}
Location: {rfp.location}
Remote Work: {'Allowed' if rfp.remote_allowed else 'Not allowed'}

Description: {rfp.description}

Skill Requirements: {requirements_text}
        """.strip()

        return text

    def index_documents(self, documents: List[Document]):
        """Index documents in ChromaDB."""
        logger.info("Splitting and indexing documents...")

        # Split documents into chunks
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            all_chunks.extend(chunks)

        logger.info(f"Created {len(all_chunks)} text chunks")

        # Clear existing collection
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass
        self.collection = self.client.create_collection(name=self.collection_name)

        # Prepare data for ChromaDB
        chunk_texts = [chunk.page_content for chunk in all_chunks]
        chunk_ids = [f"chunk_{i}" for i in range(len(all_chunks))]
        chunk_metadatas = [chunk.metadata for chunk in all_chunks]

        # Generate embeddings and store in batches
        batch_size = 50
        for i in range(0, len(chunk_texts), batch_size):
            batch_texts = chunk_texts[i:i+batch_size]
            batch_ids = chunk_ids[i:i+batch_size]
            batch_metadatas = chunk_metadatas[i:i+batch_size]

            # Generate embeddings
            embeddings = self.embeddings.embed_documents(batch_texts)

            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids
            )

            logger.info(f"Indexed batch {i//batch_size + 1}/{(len(chunk_texts)-1)//batch_size + 1}")

        logger.info(f"Successfully indexed {len(all_chunks)} document chunks")

    def search_similar_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar document chunks."""
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        for i, doc in enumerate(results["documents"][0]):
            formatted_results.append({
                "content": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return formatted_results

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate answer using retrieved context."""
        # Prepare context
        context_text = "\n\n".join([
            f"Document {i+1} (Type: {chunk['metadata'].get('type', 'unknown')}):\n{chunk['content']}"
            for i, chunk in enumerate(context_chunks)
        ])

        # Create prompt
        prompt = f"""
You are an AI assistant helping with programmer staffing queries. Use the provided context to answer the user's question accurately.

Context:
{context_text}

User Question: {query}

Instructions:
- Answer based only on the provided context
- Be specific and include relevant details (names, numbers, skills, etc.)
- If the context doesn't contain enough information, say so clearly
- For counting questions, count carefully from the context
- For filtering questions, list the specific matches
- For comparison questions, provide detailed comparisons

Answer:"""

        # Generate response
        response = self.llm.invoke(prompt)
        return response.content

    def query(self, question: str, n_results: int = 5) -> QueryResult:
        """Process a query through the naive RAG system."""
        start_time = time.time()

        # Search for relevant chunks
        context_chunks = self.search_similar_chunks(question, n_results)

        # Generate answer
        answer = self.generate_answer(question, context_chunks)

        execution_time = time.time() - start_time

        # Create result
        result = QueryResult(
            query=question,
            answer=answer,
            source_type="naive_rag",
            context=[chunk["content"] for chunk in context_chunks],
            cypher_query=None,  # Not applicable for naive RAG
            execution_time=execution_time,
            confidence_score=None  # Could be calculated from distances
        )

        return result

def test_naive_rag_system():
    """Test the naive RAG system with sample queries."""
    test_queries = [
        "How many Python developers are available?",
        "List developers with AWS certifications",
        "What is the average hourly rate for React developers?",
        "Find developers who have worked on fintech projects",
        "Which skills are most common among our programmers?",
        "Who are the most experienced JavaScript developers?",
        "What projects require machine learning skills?",
        "Find developers available for immediate start"
    ]

    print("\n" + "="*60)
    print("TESTING NAIVE RAG SYSTEM")
    print("="*60)

    rag_system = NaiveRAGSystem()

    # Load and index documents
    print("Loading and indexing documents...")
    documents = rag_system.load_and_process_documents()
    rag_system.index_documents(documents)

    print(f"\nTesting {len(test_queries)} sample queries...")
    print("-" * 60)

    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)

        result = rag_system.query(query)
        results.append(result)

        print(f"Answer: {result.answer}")
        print(f"Execution time: {result.execution_time:.2f}s")
        print(f"Context chunks used: {len(result.context)}")

    # Save results for later comparison
    def json_serializer(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    results_data = [result.model_dump() for result in results]
    with open("results/naive_rag_results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=json_serializer)

    print("\n" + "="*60)
    print("✓ Naive RAG testing completed!")
    print("Results saved to: results/naive_rag_results.json")
    print("\nNext step: Run uv run python 4_graph_rag_system.py")

def main():
    """Main function for naive RAG baseline."""
    print("Naive RAG Baseline System")
    print("=" * 30)

    # Ensure results directory exists
    import os
    os.makedirs("results", exist_ok=True)

    # Test the system
    test_naive_rag_system()

if __name__ == "__main__":
    main()