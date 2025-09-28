"""
Neo4j Utility Functions
=======================

Provides utilities for Neo4j connection, status checking, and data management
for the GraphRAG educational system.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph
import logging

logger = logging.getLogger(__name__)

class Neo4jConnectionError(Exception):
    """Custom exception for Neo4j connection issues."""
    pass

class Neo4jStatusChecker:
    """Handles Neo4j connection and status checking for educational purposes."""

    def __init__(self, uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "password123"):
        """Initialize with connection parameters."""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None

    def check_connection(self) -> Dict[str, Any]:
        """
        Check Neo4j connection and return comprehensive status.

        Returns:
            Dict with connection status, version info, and any errors
        """
        status = {
            'connected': False,
            'version': None,
            'error': None,
            'uri': self.uri,
            'database': 'neo4j'
        }

        try:
            # Try to establish connection
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

            # Test connection with a simple query
            with self.driver.session() as session:
                result = session.run("CALL dbms.components() YIELD name, versions, edition")
                components = list(result)

                if components:
                    status['connected'] = True
                    status['version'] = components[0]['versions'][0]
                    status['edition'] = components[0]['edition']

            logger.info(f"✓ Connected to Neo4j {status['version']}")

        except Exception as e:
            status['error'] = str(e)
            logger.error(f"✗ Failed to connect to Neo4j: {e}")

        finally:
            if self.driver:
                self.driver.close()

        return status

    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics.

        Returns:
            Dict with node counts, relationship counts, and other stats
        """
        if not self._ensure_connection():
            return {'error': 'No connection to Neo4j'}

        try:
            graph = Neo4jGraph()
            stats = {}

            # Basic counts
            queries = {
                'total_nodes': "MATCH (n) RETURN count(n) as count",
                'total_relationships': "MATCH ()-[r]->() RETURN count(r) as count",
                'programmers': "MATCH (n:Programmer) RETURN count(n) as count",
                'skills': "MATCH (n:Skill) RETURN count(n) as count",
                'projects': "MATCH (n:Project) RETURN count(n) as count",
                'companies': "MATCH (n:Company) RETURN count(n) as count",
                'certifications': "MATCH (n:Certification) RETURN count(n) as count"
            }

            for stat_name, query in queries.items():
                try:
                    result = graph.query(query)
                    stats[stat_name] = result[0]['count'] if result else 0
                except Exception as e:
                    stats[stat_name] = 0
                    logger.debug(f"Could not get {stat_name}: {e}")

            # Node types
            try:
                result = graph.query("MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC")
                stats['node_types'] = {row['type']: row['count'] for row in result if row['type']}
            except Exception:
                stats['node_types'] = {}

            # Relationship types
            try:
                result = graph.query("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC")
                stats['relationship_types'] = {row['type']: row['count'] for row in result}
            except Exception:
                stats['relationship_types'] = {}

            # Last update time (if available)
            try:
                result = graph.query("MATCH (n) WHERE n.created_at IS NOT NULL RETURN max(n.created_at) as last_update")
                if result and result[0]['last_update']:
                    stats['last_update'] = result[0]['last_update']
            except Exception:
                stats['last_update'] = None

            graph.close()
            return stats

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}

    def check_data_integrity(self) -> Dict[str, Any]:
        """
        Check data integrity and identify potential issues.

        Returns:
            Dict with integrity check results
        """
        if not self._ensure_connection():
            return {'error': 'No connection to Neo4j'}

        try:
            graph = Neo4jGraph()
            integrity = {
                'issues': [],
                'warnings': [],
                'suggestions': []
            }

            # Check for orphaned nodes
            orphan_query = """
            MATCH (n)
            WHERE NOT (n)--()
            RETURN labels(n)[0] as type, count(n) as count
            ORDER BY count DESC
            """
            orphans = graph.query(orphan_query)
            if orphans:
                for orphan in orphans:
                    if orphan['count'] > 0:
                        integrity['warnings'].append(f"{orphan['count']} orphaned {orphan['type']} nodes")

            # Check for programmers without skills
            no_skills_query = """
            MATCH (p:Programmer)
            WHERE NOT (p)-[:HAS_SKILL]->()
            RETURN count(p) as count
            """
            result = graph.query(no_skills_query)
            if result and result[0]['count'] > 0:
                integrity['issues'].append(f"{result[0]['count']} programmers without skills")

            # Check for skills without categories
            uncategorized_query = """
            MATCH (s:Skill)
            WHERE s.category IS NULL OR s.category = ''
            RETURN count(s) as count
            """
            result = graph.query(uncategorized_query)
            if result and result[0]['count'] > 0:
                integrity['warnings'].append(f"{result[0]['count']} skills without categories")

            graph.close()
            return integrity

        except Exception as e:
            logger.error(f"Error checking data integrity: {e}")
            return {'error': str(e)}

    def clear_database(self, confirm: bool = False) -> bool:
        """
        Clear all data from the database.

        Args:
            confirm: Must be True to actually clear data

        Returns:
            Success status
        """
        if not confirm:
            logger.warning("Clear operation requires explicit confirmation")
            return False

        if not self._ensure_connection():
            return False

        try:
            graph = Neo4jGraph()

            # Clear all data
            graph.query("MATCH (n) DETACH DELETE n")

            # Drop constraints and indexes
            try:
                constraints = graph.query("SHOW CONSTRAINTS")
                for constraint in constraints:
                    if constraint.get('name'):
                        graph.query(f"DROP CONSTRAINT {constraint['name']}")
            except Exception as e:
                logger.debug(f"Could not drop constraints: {e}")

            try:
                indexes = graph.query("SHOW INDEXES")
                for index in indexes:
                    name = index.get('name', '')
                    if name and not name.startswith('__'):
                        graph.query(f"DROP INDEX {name}")
            except Exception as e:
                logger.debug(f"Could not drop indexes: {e}")

            graph.close()
            logger.info("✓ Database cleared successfully")
            return True

        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

    def _ensure_connection(self) -> bool:
        """Check if we can connect to Neo4j."""
        status = self.check_connection()
        return status['connected']

def check_docker_neo4j() -> Dict[str, Any]:
    """
    Check if Neo4j is running in Docker.

    Returns:
        Dict with Docker status information
    """
    docker_status = {
        'docker_available': False,
        'neo4j_container': None,
        'container_status': None,
        'suggestions': []
    }

    try:
        import subprocess

        # Check if Docker is available
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            docker_status['docker_available'] = True

        # Check for Neo4j containers
        result = subprocess.run(['docker', 'ps', '-a', '--filter', 'ancestor=neo4j'],
                              capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and 'neo4j' in result.stdout:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            if lines and lines[0]:  # If there are containers
                container_info = lines[0].split()
                docker_status['neo4j_container'] = container_info[0][:12]  # Container ID
                docker_status['container_status'] = container_info[6]  # Status

        # Add suggestions based on status
        if not docker_status['docker_available']:
            docker_status['suggestions'].append("Install Docker Desktop")
        elif not docker_status['neo4j_container']:
            docker_status['suggestions'].append("Run: docker run -d -p 7474:7474 -p 7687:7687 --env NEO4J_AUTH=neo4j/password123 neo4j")
        elif docker_status['container_status'] != 'Up':
            docker_status['suggestions'].append(f"Start container: docker start {docker_status['neo4j_container']}")

    except Exception as e:
        logger.debug(f"Error checking Docker: {e}")
        docker_status['suggestions'].append("Check if Docker is installed and running")

    return docker_status

def get_sample_queries() -> List[Dict[str, str]]:
    """
    Get sample Cypher queries for educational purposes.

    Returns:
        List of query examples with descriptions
    """
    return [
        {
            'name': 'Count Programmers',
            'description': 'Count total number of programmers',
            'query': 'MATCH (p:Programmer) RETURN count(p) as programmer_count'
        },
        {
            'name': 'Skills Distribution',
            'description': 'Show most common skills',
            'query': '''MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill)
RETURN s.name as skill, count(p) as programmer_count
ORDER BY programmer_count DESC
LIMIT 10'''
        },
        {
            'name': 'Python Developers',
            'description': 'Find Python developers with experience',
            'query': '''MATCH (p:Programmer)-[hs:HAS_SKILL]->(s:Skill {name: 'Python'})
RETURN p.name, hs.proficiency, hs.years_experience
ORDER BY hs.proficiency DESC, hs.years_experience DESC'''
        },
        {
            'name': 'Company Networks',
            'description': 'Show programmers who worked at same companies',
            'query': '''MATCH (p1:Programmer)-[:WORKED_ON]->(pr:Project)-[:FOR_CLIENT]->(c:Company)<-[:FOR_CLIENT]-(pr2:Project)<-[:WORKED_ON]-(p2:Programmer)
WHERE p1 <> p2
RETURN c.name as company, collect(DISTINCT p1.name) + collect(DISTINCT p2.name) as programmers
LIMIT 5'''
        }
    ]