"""
Natural Language to Cypher Query Translation
===========================================

Translates natural language queries into Cypher queries for Neo4j GraphRAG.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

from typing import Dict, List, Optional, Tuple
from langchain_openai import ChatOpenAI
import re
import logging

logger = logging.getLogger(__name__)

class CypherQueryTranslator:
    """Translates natural language queries to Cypher."""

    def __init__(self):
        """Initialize the query translator."""
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Define the graph schema for the LLM
        self.schema_description = """
Neo4j Graph Schema:

Nodes:
- Programmer: id, name, email, location, hourly_rate, availability_start, bio
- Skill: name, category
- Certification: name, provider
- Project: id, name, client, start_date, end_date, status, budget, team_size
- RFP: id, title, client, project_type, duration_months, team_size, location

Relationships:
- (Programmer)-[HAS_SKILL {proficiency: 1-5, years_experience: int}]->(Skill)
- (Programmer)-[HAS_CERTIFICATION {obtained_date, expiry_date, credential_id}]->(Certification)
- (Programmer)-[WORKED_ON {role, start_date, end_date, allocation_percentage, technologies_used}]->(Project)
- (Programmer)-[WORKED_WITH {shared_projects, collaboration_count}]->(Programmer)
- (Project)-[REQUIRES_SKILL {min_proficiency, min_years, is_mandatory}]->(Skill)
- (RFP)-[RFP_REQUIRES_SKILL {min_proficiency, min_years, is_mandatory, preferred_certifications}]->(Skill)

Common Skill Categories:
- Programming Language, Framework, Database, Cloud, DevOps, Frontend, Backend, Mobile, Data Science, Security

Project Statuses:
- planned, active, completed, on_hold, cancelled
"""

        # Query templates for common patterns
        self.query_templates = {
            "count_programmers_with_skill": """
MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill {{name: '{skill}'}})
RETURN count(p) as programmer_count
""",
            "find_programmers_with_skill": """
MATCH (p:Programmer)-[hs:HAS_SKILL]->(s:Skill {{name: '{skill}'}})
RETURN p.name, p.email, p.location, p.hourly_rate, hs.proficiency, hs.years_experience
ORDER BY hs.proficiency DESC, hs.years_experience DESC
""",
            "count_certified_programmers": """
MATCH (p:Programmer)-[:HAS_CERTIFICATION]->(c:Certification)
WHERE c.name CONTAINS '{certification}'
RETURN count(DISTINCT p) as certified_count
""",
            "average_rate_by_skill": """
MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill {{name: '{skill}'}})
RETURN avg(p.hourly_rate) as average_rate, count(p) as programmer_count
""",
            "top_programmers_by_projects": """
MATCH (p:Programmer)-[:WORKED_ON]->(pr:Project)
RETURN p.name, p.location, count(pr) as project_count, avg(p.hourly_rate) as avg_rate
ORDER BY project_count DESC
LIMIT {limit}
""",
            "available_programmers": """
MATCH (p:Programmer)
WHERE p.availability_start <= date('{date}') OR p.availability_start IS NULL
RETURN p.name, p.location, p.hourly_rate, p.availability_start
ORDER BY p.availability_start
""",
            "skills_distribution": """
MATCH (p:Programmer)-[:HAS_SKILL]->(s:Skill)
RETURN s.name as skill, s.category as category, count(p) as programmer_count
ORDER BY programmer_count DESC
LIMIT {limit}
""",
            "collaboration_network": """
MATCH (p1:Programmer)-[w:WORKED_WITH]->(p2:Programmer)
WHERE p1.name CONTAINS '{programmer}' OR p2.name CONTAINS '{programmer}'
RETURN p1.name, p2.name, w.collaboration_count, w.shared_projects
ORDER BY w.collaboration_count DESC
"""
        }

    def identify_query_type(self, query: str) -> Tuple[str, Dict[str, str]]:
        """Identify the type of query and extract parameters."""
        query_lower = query.lower()

        # Count queries
        if any(word in query_lower for word in ["how many", "count", "number of"]):
            if "python" in query_lower:
                return "count_programmers_with_skill", {"skill": "Python"}
            elif "aws" in query_lower and ("certified" in query_lower or "certification" in query_lower):
                return "count_certified_programmers", {"certification": "AWS"}
            elif "java" in query_lower:
                return "count_programmers_with_skill", {"skill": "Java"}
            elif "react" in query_lower:
                return "count_programmers_with_skill", {"skill": "React"}

        # Average/aggregate queries
        elif any(word in query_lower for word in ["average", "avg", "mean"]):
            if "rate" in query_lower or "hourly" in query_lower:
                if "python" in query_lower:
                    return "average_rate_by_skill", {"skill": "Python"}
                elif "react" in query_lower:
                    return "average_rate_by_skill", {"skill": "React"}
                elif "java" in query_lower:
                    return "average_rate_by_skill", {"skill": "Java"}

        # Top/ranking queries
        elif any(word in query_lower for word in ["top", "best", "most", "list"]):
            if "project" in query_lower:
                limit = "10"
                if "5" in query: limit = "5"
                elif "3" in query: limit = "3"
                return "top_programmers_by_projects", {"limit": limit}
            elif "skill" in query_lower:
                limit = "20"
                if "10" in query: limit = "10"
                elif "5" in query: limit = "5"
                return "skills_distribution", {"limit": limit}

        # Availability queries
        elif any(word in query_lower for word in ["available", "availability", "immediate"]):
            from datetime import date
            today = date.today().isoformat()
            return "available_programmers", {"date": today}

        # Find/search queries
        elif any(word in query_lower for word in ["find", "search", "who", "which"]):
            if "python" in query_lower:
                return "find_programmers_with_skill", {"skill": "Python"}
            elif "aws" in query_lower and ("certified" in query_lower or "certification" in query_lower):
                return "count_certified_programmers", {"certification": "AWS"}
            elif "worked with" in query_lower or "collaborated" in query_lower:
                # Extract programmer name if mentioned
                return "collaboration_network", {"programmer": ""}

        # Fallback to LLM translation
        return "custom", {}

    def translate_to_cypher(self, natural_query: str) -> str:
        """Translate natural language query to Cypher."""
        # First try template matching
        query_type, params = self.identify_query_type(natural_query)

        if query_type != "custom" and query_type in self.query_templates:
            try:
                cypher_query = self.query_templates[query_type].format(**params)
                logger.info(f"Used template: {query_type}")
                return cypher_query.strip()
            except KeyError as e:
                logger.warning(f"Template parameter missing: {e}")

        # Fallback to LLM translation
        return self._llm_translate(natural_query)

    def _llm_translate(self, natural_query: str) -> str:
        """Use LLM to translate complex queries."""
        prompt = f"""
Convert the following natural language query into a Cypher query for Neo4j.

{self.schema_description}

Natural Language Query: {natural_query}

Requirements:
1. Generate valid Cypher syntax
2. Use the exact node labels and relationship types from the schema
3. Include appropriate WHERE clauses for filtering
4. Use RETURN statements that provide meaningful results
5. Add ORDER BY and LIMIT clauses when appropriate
6. For date comparisons, use date() function: date('2025-01-01')
7. For partial string matching, use CONTAINS: name CONTAINS 'Python'
8. For counting, use count() function
9. For aggregations, use avg(), sum(), min(), max() as needed

Return only the Cypher query without explanations.

Cypher Query:"""

        response = self.llm.invoke(prompt)
        cypher_query = response.content.strip()

        # Clean up the response
        if cypher_query.startswith("```"):
            cypher_query = cypher_query.split("\n", 1)[1]
        if cypher_query.endswith("```"):
            cypher_query = cypher_query.rsplit("\n", 1)[0]

        logger.info("Used LLM translation")
        return cypher_query

    def validate_cypher_query(self, cypher_query: str) -> bool:
        """Basic validation of Cypher query syntax."""
        # Check for required MATCH or WITH clauses
        if not any(keyword in cypher_query.upper() for keyword in ["MATCH", "WITH", "CREATE", "MERGE"]):
            return False

        # Check for RETURN clause
        if "RETURN" not in cypher_query.upper():
            return False

        # Check for balanced parentheses and brackets
        if cypher_query.count("(") != cypher_query.count(")"):
            return False
        if cypher_query.count("[") != cypher_query.count("]"):
            return False
        if cypher_query.count("{") != cypher_query.count("}"):
            return False

        # Basic syntax checks
        forbidden_patterns = [
            "DELETE",  # Prevent destructive operations
            "DROP",
            "REMOVE"
        ]

        for pattern in forbidden_patterns:
            if pattern in cypher_query.upper():
                return False

        return True

    def get_example_queries(self) -> List[Dict[str, str]]:
        """Get example natural language queries with their Cypher translations."""
        examples = [
            {
                "natural": "How many Python developers do we have?",
                "cypher": self.query_templates["count_programmers_with_skill"].format(skill="Python"),
                "description": "Count programmers with Python skills"
            },
            {
                "natural": "What is the average hourly rate for React developers?",
                "cypher": self.query_templates["average_rate_by_skill"].format(skill="React"),
                "description": "Calculate average rate for React developers"
            },
            {
                "natural": "List the top 5 developers by number of projects",
                "cypher": self.query_templates["top_programmers_by_projects"].format(limit="5"),
                "description": "Rank developers by project count"
            },
            {
                "natural": "Find developers with AWS certifications",
                "cypher": self.query_templates["count_certified_programmers"].format(certification="AWS"),
                "description": "Find AWS certified developers"
            },
            {
                "natural": "Show the most common skills",
                "cypher": self.query_templates["skills_distribution"].format(limit="10"),
                "description": "Distribution of skills across programmers"
            }
        ]

        return examples

def test_query_translator():
    """Test the query translator with sample queries."""
    translator = CypherQueryTranslator()

    test_queries = [
        "How many Python developers are available?",
        "What is the average hourly rate for React developers?",
        "List the top 5 developers by project count",
        "Find developers with AWS certifications",
        "Who are the most experienced JavaScript developers?",
        "Show me developers available for immediate start",
        "What are the most common skills among programmers?",
        "Find developers who worked on fintech projects"
    ]

    print("\n" + "="*60)
    print("TESTING CYPHER QUERY TRANSLATOR")
    print("="*60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Natural Query: {query}")
        print("-" * 40)

        cypher = translator.translate_to_cypher(query)
        is_valid = translator.validate_cypher_query(cypher)

        print(f"Cypher Query:\n{cypher}")
        print(f"Valid: {is_valid}")

    # Show examples
    print("\n" + "="*60)
    print("EXAMPLE QUERY TRANSLATIONS")
    print("="*60)

    examples = translator.get_example_queries()
    for example in examples:
        print(f"\nNatural: {example['natural']}")
        print(f"Cypher: {example['cypher']}")
        print(f"Description: {example['description']}")

if __name__ == "__main__":
    test_query_translator()