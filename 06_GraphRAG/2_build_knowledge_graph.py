"""
Knowledge Graph Construction for GraphRAG System
===============================================

Populates Neo4j database with programmer, project, and RFP data to create
a comprehensive knowledge graph for demonstrating GraphRAG capabilities.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import json
from glob import glob
from typing import List, Dict
import logging
from datetime import datetime

from utils.graph_schema import GraphSchema
from utils.models import ProgrammerProfile, Project, RFP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """Builds the knowledge graph from generated data."""

    def __init__(self):
        """Initialize the graph builder."""
        self.graph = GraphSchema()

    def load_programmer_data(self) -> List[ProgrammerProfile]:
        """Load programmer data from JSON files."""
        programmers = []
        programmer_files = glob("data/programmers/*.json")

        logger.info(f"Loading {len(programmer_files)} programmer profiles...")

        for file_path in programmer_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    programmer = ProgrammerProfile.model_validate(data)
                    programmers.append(programmer)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Successfully loaded {len(programmers)} programmer profiles")
        return programmers

    def load_project_data(self) -> List[Project]:
        """Load project data from JSON files."""
        projects = []
        project_files = glob("data/projects/*.json")

        logger.info(f"Loading {len(project_files)} projects...")

        for file_path in project_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    project = Project.model_validate(data)
                    projects.append(project)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Successfully loaded {len(projects)} projects")
        return projects

    def load_rfp_data(self) -> List[RFP]:
        """Load RFP data from JSON files."""
        rfps = []
        rfp_files = glob("data/rfps/*.json")

        logger.info(f"Loading {len(rfp_files)} RFPs...")

        for file_path in rfp_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    rfp = RFP.model_validate(data)
                    rfps.append(rfp)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Successfully loaded {len(rfps)} RFPs")
        return rfps

    def create_programmer_nodes(self, programmers: List[ProgrammerProfile]):
        """Create programmer nodes in the graph."""
        logger.info("Creating programmer nodes...")

        for programmer in programmers:
            # Prepare programmer data for Neo4j
            programmer_data = {
                "id": programmer.id,
                "name": programmer.name,
                "email": programmer.email,
                "phone": programmer.phone,
                "location": programmer.location,
                "hourly_rate": programmer.hourly_rate,
                "availability_start": programmer.availability_start.isoformat() if programmer.availability_start else None,
                "bio": programmer.bio,
                "linkedin_url": programmer.linkedin_url,
                "github_url": programmer.github_url,
                "created_at": programmer.created_at.isoformat()
            }

            self.graph.create_programmer_node(programmer_data)

        logger.info(f"Created {len(programmers)} programmer nodes")

    def create_skill_nodes(self, programmers: List[ProgrammerProfile]):
        """Create skill nodes and relationships."""
        logger.info("Creating skill nodes and relationships...")

        unique_skills = set()
        skill_relationships = []

        # Collect all unique skills
        for programmer in programmers:
            for skill in programmer.skills:
                unique_skills.add((skill.name, skill.category.value))
                skill_relationships.append({
                    "programmer_id": programmer.id,
                    "skill_name": skill.name,
                    "proficiency": skill.proficiency,
                    "years_experience": skill.years_experience
                })

        # Create skill nodes
        for skill_name, category in unique_skills:
            self.graph.create_skill_node(skill_name, category)

        # Create HAS_SKILL relationships
        for rel in skill_relationships:
            programmer_id = rel.pop("programmer_id")
            skill_name = rel.pop("skill_name")
            self.graph.create_has_skill_relationship(programmer_id, skill_name, rel)

        logger.info(f"Created {len(unique_skills)} skill nodes and {len(skill_relationships)} HAS_SKILL relationships")

    def create_certification_nodes(self, programmers: List[ProgrammerProfile]):
        """Create certification nodes and relationships."""
        logger.info("Creating certification nodes and relationships...")

        unique_certifications = set()
        cert_relationships = []

        # Collect all unique certifications
        for programmer in programmers:
            for cert in programmer.certifications:
                unique_certifications.add((cert.name, cert.provider))
                cert_relationships.append({
                    "programmer_id": programmer.id,
                    "cert_name": cert.name,
                    "obtained_date": cert.obtained_date.isoformat(),
                    "expiry_date": cert.expiry_date.isoformat() if cert.expiry_date else None,
                    "credential_id": cert.credential_id
                })

        # Create certification nodes
        for cert_name, provider in unique_certifications:
            cert_data = {
                "name": cert_name,
                "provider": provider,
                "typical_validity_years": 3  # Default validity
            }
            self.graph.create_certification_node(cert_data)

        # Create HAS_CERTIFICATION relationships
        for rel in cert_relationships:
            programmer_id = rel.pop("programmer_id")
            cert_name = rel.pop("cert_name")
            self.graph.create_has_certification_relationship(programmer_id, cert_name, rel)

        logger.info(f"Created {len(unique_certifications)} certification nodes and {len(cert_relationships)} HAS_CERTIFICATION relationships")

    def create_project_nodes_and_relationships(self, programmers: List[ProgrammerProfile], projects: List[Project]):
        """Create project nodes from both programmer experience and standalone projects."""
        logger.info("Creating project nodes and relationships...")

        # Create nodes for standalone projects
        for project in projects:
            project_data = {
                "id": project.id,
                "name": project.name,
                "client": project.client,
                "description": project.description,
                "start_date": project.start_date.isoformat(),
                "end_date": project.end_date.isoformat() if project.end_date else None,
                "estimated_duration_months": project.estimated_duration_months,
                "budget": project.budget,
                "status": project.status.value,
                "team_size": project.team_size,
                "created_at": project.created_at.isoformat()
            }

            self.graph.create_project_node(project_data)

            # Create REQUIRES_SKILL relationships for projects
            for requirement in project.requirements:
                req_data = {
                    "min_proficiency": requirement.min_proficiency,
                    "min_years": requirement.min_years,
                    "is_mandatory": requirement.is_mandatory
                }

                # Create skill node if it doesn't exist
                self.graph.create_skill_node(requirement.skill_name, "General")
                self.graph.create_requires_skill_relationship(project.id, requirement.skill_name, req_data)

        # Create project nodes from programmer experience
        project_experience_nodes = set()
        worked_on_relationships = []

        for programmer in programmers:
            for i, experience in enumerate(programmer.project_experience):
                # Create unique project ID based on programmer and experience
                project_id = f"{programmer.id}_proj_{i+1}"
                project_key = (experience.project_name, experience.client)

                if project_key not in project_experience_nodes:
                    project_experience_nodes.add(project_key)

                    # Calculate project end date or use current date
                    end_date = experience.end_date.isoformat() if experience.end_date else None

                    project_data = {
                        "id": project_id,
                        "name": experience.project_name,
                        "client": experience.client,
                        "description": experience.description,
                        "start_date": experience.start_date.isoformat(),
                        "end_date": end_date,
                        "estimated_duration_months": 6,  # Default estimate
                        "budget": None,
                        "status": "completed" if experience.end_date else "active",
                        "team_size": experience.team_size or 5,
                        "created_at": datetime.now().isoformat()
                    }

                    self.graph.create_project_node(project_data)

                # Create WORKED_ON relationship
                work_data = {
                    "role": experience.role.value,
                    "start_date": experience.start_date.isoformat(),
                    "end_date": experience.end_date.isoformat() if experience.end_date else None,
                    "allocation_percentage": experience.allocation_percentage,
                    "technologies_used": experience.technologies_used,
                    "team_size": experience.team_size
                }

                worked_on_relationships.append((programmer.id, project_id, work_data))

        # Create WORKED_ON relationships
        for programmer_id, project_id, work_data in worked_on_relationships:
            self.graph.create_worked_on_relationship(programmer_id, project_id, work_data)

        logger.info(f"Created {len(projects)} standalone projects and {len(project_experience_nodes)} experience projects")
        logger.info(f"Created {len(worked_on_relationships)} WORKED_ON relationships")

    def create_rfp_nodes_and_relationships(self, rfps: List[RFP]):
        """Create RFP nodes and their skill requirements."""
        logger.info("Creating RFP nodes and relationships...")

        for rfp in rfps:
            rfp_data = {
                "id": rfp.id,
                "title": rfp.title,
                "client": rfp.client,
                "description": rfp.description,
                "project_type": rfp.project_type,
                "duration_months": rfp.duration_months,
                "team_size": rfp.team_size,
                "budget_range": rfp.budget_range,
                "start_date": rfp.start_date.isoformat(),
                "location": rfp.location,
                "remote_allowed": rfp.remote_allowed,
                "created_at": rfp.created_at.isoformat()
            }

            self.graph.create_rfp_node(rfp_data)

            # Create RFP_REQUIRES_SKILL relationships
            for requirement in rfp.requirements:
                req_data = {
                    "min_proficiency": requirement.min_proficiency,
                    "min_years": requirement.min_years,
                    "is_mandatory": requirement.is_mandatory,
                    "preferred_certifications": requirement.preferred_certifications
                }

                # Ensure skill exists
                self.graph.create_skill_node(requirement.skill_name, "General")
                self.graph.create_rfp_requires_skill_relationship(rfp.id, requirement.skill_name, req_data)

        logger.info(f"Created {len(rfps)} RFP nodes with skill requirements")

    def create_collaboration_relationships(self):
        """Create WORKED_WITH relationships based on shared projects."""
        logger.info("Creating collaboration relationships...")
        self.graph.create_worked_with_relationships()

    def build_complete_graph(self) -> Dict[str, int]:
        """Build the complete knowledge graph."""
        logger.info("Starting knowledge graph construction...")

        # Setup database
        logger.info("Setting up database schema...")
        self.graph.create_constraints()
        self.graph.create_indexes()

        # Load data
        programmers = self.load_programmer_data()
        projects = self.load_project_data()
        rfps = self.load_rfp_data()

        if not programmers:
            raise Exception("No programmer data found. Run 1_generate_data.py first.")

        # Clear existing data
        logger.info("Clearing existing data...")
        self.graph.clear_database()

        # Build graph
        self.create_programmer_nodes(programmers)
        self.create_skill_nodes(programmers)
        self.create_certification_nodes(programmers)
        self.create_project_nodes_and_relationships(programmers, projects)
        self.create_rfp_nodes_and_relationships(rfps)
        self.create_collaboration_relationships()

        # Get final statistics
        stats = self.graph.get_database_stats()
        logger.info("Knowledge graph construction completed!")

        return stats

    def close(self):
        """Close the graph connection."""
        self.graph.close()

def display_graph_statistics(stats: Dict[str, int]):
    """Display comprehensive graph statistics."""
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*60)

    print("\nNodes:")
    print(f"  Programmers:     {stats['programmers']:>6}")
    print(f"  Skills:          {stats['skills']:>6}")
    print(f"  Certifications:  {stats['certifications']:>6}")
    print(f"  Projects:        {stats['projects']:>6}")
    print(f"  RFPs:            {stats['rfps']:>6}")

    total_nodes = stats['programmers'] + stats['skills'] + stats['certifications'] + stats['projects'] + stats['rfps']
    print(f"  Total Nodes:     {total_nodes:>6}")

    print("\nRelationships:")
    print(f"  HAS_SKILL:       {stats['has_skill']:>6}")
    print(f"  HAS_CERTIFICATION: {stats['has_certification']:>4}")
    print(f"  WORKED_ON:       {stats['worked_on']:>6}")
    print(f"  WORKED_WITH:     {stats['worked_with']:>6}")
    print(f"  REQUIRES_SKILL:  {stats['requires_skill']:>6}")

    total_relationships = (stats['has_skill'] + stats['has_certification'] +
                          stats['worked_on'] + stats['worked_with'] + stats['requires_skill'])
    print(f"  Total Relationships: {total_relationships:>2}")

    print(f"\nGraph Density:")
    print(f"  Avg skills per programmer: {stats['has_skill'] / max(stats['programmers'], 1):.1f}")
    print(f"  Avg certifications per programmer: {stats['has_certification'] / max(stats['programmers'], 1):.1f}")
    print(f"  Avg projects per programmer: {stats['worked_on'] / max(stats['programmers'], 1):.1f}")

def main():
    """Main knowledge graph construction function."""
    print("Building Knowledge Graph for GraphRAG System")
    print("=" * 50)

    builder = None
    try:
        # Initialize builder
        builder = KnowledgeGraphBuilder()

        # Build the complete graph
        stats = builder.build_complete_graph()

        # Display statistics
        display_graph_statistics(stats)

        # Test sample queries
        print("\nTesting graph connectivity...")
        builder.graph.test_sample_queries()

        print("\n" + "="*60)
        print("✓ Knowledge graph built successfully!")
        print("\nNext steps:")
        print("1. Access Neo4j Browser: http://localhost:7474")
        print("   Username: neo4j, Password: password123")
        print("2. Run: uv run python 3_naive_rag_baseline.py")
        print("3. Run: uv run python 4_graph_rag_system.py")

        print("\nSample Queries to try in Neo4j Browser:")
        print("// Find Python developers")
        print("MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill {name: 'Python'})")
        print("RETURN p.name, s.name LIMIT 10")
        print()
        print("// Find collaboration networks")
        print("MATCH (p1:Programmer)-[w:WORKED_WITH]-(p2:Programmer)")
        print("RETURN p1.name, p2.name, w.collaboration_count LIMIT 10")

    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}")
        raise
    finally:
        if builder:
            builder.close()

if __name__ == "__main__":
    main()