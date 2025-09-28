"""
Neo4j Graph Schema Definitions
=============================

Defines the graph schema and provides utilities for working with Neo4j.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphSchema:
    """Manages Neo4j graph schema and operations."""

    def __init__(self, uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "password123"):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        """Close the Neo4j driver."""
        if self.driver:
            self.driver.close()

    def create_constraints(self):
        """Create uniqueness constraints for the graph."""
        constraints = [
            "CREATE CONSTRAINT programmer_id IF NOT EXISTS FOR (p:Programmer) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT skill_name IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
            "CREATE CONSTRAINT certification_name IF NOT EXISTS FOR (c:Certification) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (pr:Project) REQUIRE pr.id IS UNIQUE",
            "CREATE CONSTRAINT rfp_id IF NOT EXISTS FOR (r:RFP) REQUIRE r.id IS UNIQUE"
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint.split()[2]}")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")

    def create_indexes(self):
        """Create indexes for better query performance."""
        indexes = [
            "CREATE INDEX programmer_name IF NOT EXISTS FOR (p:Programmer) ON (p.name)",
            "CREATE INDEX programmer_location IF NOT EXISTS FOR (p:Programmer) ON (p.location)",
            "CREATE INDEX programmer_rate IF NOT EXISTS FOR (p:Programmer) ON (p.hourly_rate)",
            "CREATE INDEX skill_category IF NOT EXISTS FOR (s:Skill) ON (s.category)",
            "CREATE INDEX project_status IF NOT EXISTS FOR (pr:Project) ON (pr.status)",
            "CREATE INDEX project_start_date IF NOT EXISTS FOR (pr:Project) ON (pr.start_date)",
            "CREATE INDEX certification_provider IF NOT EXISTS FOR (c:Certification) ON (c.provider)"
        ]

        with self.driver.session() as session:
            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"Created index: {index.split()[2]}")
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")

    def clear_database(self):
        """Clear all nodes and relationships from the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")

    def create_programmer_node(self, programmer_data: Dict[str, Any]) -> str:
        """Create a programmer node."""
        query = """
        CREATE (p:Programmer {
            id: $id,
            name: $name,
            email: $email,
            phone: $phone,
            location: $location,
            hourly_rate: $hourly_rate,
            availability_start: date($availability_start),
            bio: $bio,
            linkedin_url: $linkedin_url,
            github_url: $github_url,
            created_at: datetime($created_at)
        })
        RETURN p.id as programmer_id
        """

        with self.driver.session() as session:
            result = session.run(query, **programmer_data)
            return result.single()["programmer_id"]

    def create_skill_node(self, skill_name: str, category: str) -> str:
        """Create or get a skill node."""
        query = """
        MERGE (s:Skill {name: $name})
        ON CREATE SET s.category = $category
        RETURN s.name as skill_name
        """

        with self.driver.session() as session:
            result = session.run(query, name=skill_name, category=category)
            return result.single()["skill_name"]

    def create_certification_node(self, cert_data: Dict[str, Any]) -> str:
        """Create or get a certification node."""
        query = """
        MERGE (c:Certification {name: $name})
        ON CREATE SET
            c.provider = $provider,
            c.typical_validity_years = $typical_validity_years
        RETURN c.name as cert_name
        """

        with self.driver.session() as session:
            result = session.run(query, **cert_data)
            return result.single()["cert_name"]

    def create_project_node(self, project_data: Dict[str, Any]) -> str:
        """Create a project node."""
        query = """
        CREATE (pr:Project {
            id: $id,
            name: $name,
            client: $client,
            description: $description,
            start_date: date($start_date),
            end_date: date($end_date),
            estimated_duration_months: $estimated_duration_months,
            budget: $budget,
            status: $status,
            team_size: $team_size,
            created_at: datetime($created_at)
        })
        RETURN pr.id as project_id
        """

        with self.driver.session() as session:
            result = session.run(query, **project_data)
            return result.single()["project_id"]

    def create_rfp_node(self, rfp_data: Dict[str, Any]) -> str:
        """Create an RFP node."""
        query = """
        CREATE (r:RFP {
            id: $id,
            title: $title,
            client: $client,
            description: $description,
            project_type: $project_type,
            duration_months: $duration_months,
            team_size: $team_size,
            budget_range: $budget_range,
            start_date: date($start_date),
            location: $location,
            remote_allowed: $remote_allowed,
            created_at: datetime($created_at)
        })
        RETURN r.id as rfp_id
        """

        with self.driver.session() as session:
            result = session.run(query, **rfp_data)
            return result.single()["rfp_id"]

    def create_has_skill_relationship(self, programmer_id: str, skill_name: str, skill_data: Dict[str, Any]):
        """Create HAS_SKILL relationship."""
        query = """
        MATCH (p:Programmer {id: $programmer_id})
        MATCH (s:Skill {name: $skill_name})
        CREATE (p)-[r:HAS_SKILL {
            proficiency: $proficiency,
            years_experience: $years_experience
        }]->(s)
        """

        with self.driver.session() as session:
            session.run(query,
                       programmer_id=programmer_id,
                       skill_name=skill_name,
                       **skill_data)

    def create_has_certification_relationship(self, programmer_id: str, cert_name: str, cert_data: Dict[str, Any]):
        """Create HAS_CERTIFICATION relationship."""
        query = """
        MATCH (p:Programmer {id: $programmer_id})
        MATCH (c:Certification {name: $cert_name})
        CREATE (p)-[r:HAS_CERTIFICATION {
            obtained_date: date($obtained_date),
            expiry_date: date($expiry_date),
            credential_id: $credential_id
        }]->(c)
        """

        with self.driver.session() as session:
            session.run(query,
                       programmer_id=programmer_id,
                       cert_name=cert_name,
                       **cert_data)

    def create_worked_on_relationship(self, programmer_id: str, project_id: str, work_data: Dict[str, Any]):
        """Create WORKED_ON relationship."""
        query = """
        MATCH (p:Programmer {id: $programmer_id})
        MATCH (pr:Project {id: $project_id})
        CREATE (p)-[r:WORKED_ON {
            role: $role,
            start_date: date($start_date),
            end_date: date($end_date),
            allocation_percentage: $allocation_percentage,
            technologies_used: $technologies_used,
            team_size: $team_size
        }]->(pr)
        """

        with self.driver.session() as session:
            session.run(query,
                       programmer_id=programmer_id,
                       project_id=project_id,
                       **work_data)

    def create_requires_skill_relationship(self, project_id: str, skill_name: str, requirement_data: Dict[str, Any]):
        """Create REQUIRES_SKILL relationship."""
        query = """
        MATCH (pr:Project {id: $project_id})
        MATCH (s:Skill {name: $skill_name})
        CREATE (pr)-[r:REQUIRES_SKILL {
            min_proficiency: $min_proficiency,
            min_years: $min_years,
            is_mandatory: $is_mandatory
        }]->(s)
        """

        with self.driver.session() as session:
            session.run(query,
                       project_id=project_id,
                       skill_name=skill_name,
                       **requirement_data)

    def create_rfp_requires_skill_relationship(self, rfp_id: str, skill_name: str, requirement_data: Dict[str, Any]):
        """Create RFP_REQUIRES_SKILL relationship."""
        query = """
        MATCH (r:RFP {id: $rfp_id})
        MATCH (s:Skill {name: $skill_name})
        CREATE (r)-[req:RFP_REQUIRES_SKILL {
            min_proficiency: $min_proficiency,
            min_years: $min_years,
            is_mandatory: $is_mandatory,
            preferred_certifications: $preferred_certifications
        }]->(s)
        """

        with self.driver.session() as session:
            session.run(query,
                       rfp_id=rfp_id,
                       skill_name=skill_name,
                       **requirement_data)

    def create_worked_with_relationships(self):
        """Create WORKED_WITH relationships based on shared projects."""
        query = """
        MATCH (p1:Programmer)-[:WORKED_ON]->(pr:Project)<-[:WORKED_ON]-(p2:Programmer)
        WHERE p1.id < p2.id  // Avoid duplicate relationships
        WITH p1, p2, collect(pr.name) as shared_projects, count(pr) as collaboration_count
        CREATE (p1)-[r:WORKED_WITH {
            shared_projects: shared_projects,
            collaboration_count: collaboration_count
        }]->(p2)
        """

        with self.driver.session() as session:
            result = session.run(query)
            summary = result.consume()
            logger.info(f"Created {summary.counters.relationships_created} WORKED_WITH relationships")

    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        queries = {
            "programmers": "MATCH (p:Programmer) RETURN count(p) as count",
            "skills": "MATCH (s:Skill) RETURN count(s) as count",
            "certifications": "MATCH (c:Certification) RETURN count(c) as count",
            "projects": "MATCH (pr:Project) RETURN count(pr) as count",
            "rfps": "MATCH (r:RFP) RETURN count(r) as count",
            "has_skill": "MATCH ()-[r:HAS_SKILL]->() RETURN count(r) as count",
            "has_certification": "MATCH ()-[r:HAS_CERTIFICATION]->() RETURN count(r) as count",
            "worked_on": "MATCH ()-[r:WORKED_ON]->() RETURN count(r) as count",
            "worked_with": "MATCH ()-[r:WORKED_WITH]->() RETURN count(r) as count",
            "requires_skill": "MATCH ()-[r:REQUIRES_SKILL]->() RETURN count(r) as count"
        }

        stats = {}
        with self.driver.session() as session:
            for name, query in queries.items():
                result = session.run(query)
                stats[name] = result.single()["count"]

        return stats

    def test_sample_queries(self):
        """Test some sample queries to verify the graph structure."""
        sample_queries = [
            {
                "name": "Python developers count",
                "query": """
                MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill {name: 'Python'})
                RETURN count(p) as python_developers
                """
            },
            {
                "name": "Top 5 skills by programmer count",
                "query": """
                MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill)
                RETURN s.name as skill, count(p) as programmer_count
                ORDER BY programmer_count DESC
                LIMIT 5
                """
            },
            {
                "name": "AWS certified developers",
                "query": """
                MATCH (p:Programmer)-[:HAS_CERTIFICATION]->(c:Certification)
                WHERE c.name CONTAINS 'AWS'
                RETURN count(DISTINCT p) as aws_certified_developers
                """
            },
            {
                "name": "Average team collaboration",
                "query": """
                MATCH (p:Programmer)-[w:WORKED_WITH]->()
                RETURN avg(w.collaboration_count) as avg_collaborations
                """
            }
        ]

        print("\nSample Query Results:")
        print("=" * 30)

        with self.driver.session() as session:
            for query_info in sample_queries:
                try:
                    result = session.run(query_info["query"])
                    record = result.single()
                    if record:
                        print(f"{query_info['name']}: {list(record.values())[0]}")
                    else:
                        print(f"{query_info['name']}: No results")
                except Exception as e:
                    print(f"{query_info['name']}: Error - {e}")