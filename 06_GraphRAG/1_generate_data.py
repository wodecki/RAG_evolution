"""
GraphRAG Data Generation - Single Integrated Module
==================================================

Generates realistic programmer profiles and PDF CVs for GraphRAG educational demonstration.
Uses LLM to create unique, unstructured CVs in markdown format, then converts to PDF.

CRITICAL: No fallbacks, no mock data. All dependencies must be available.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
import random
from datetime import date, datetime, timedelta
from faker import Faker
from typing import List
from langchain_openai import ChatOpenAI
import markdown
from weasyprint import HTML, CSS

fake = Faker()


class GraphRAGDataGenerator:
    """Integrated generator for programmer profiles and realistic PDF CVs."""

    def __init__(self):
        """Initialize with required dependencies - fail fast if missing."""
        # Validate environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=api_key
        )

    def generate_programmer_profiles(self, num_profiles: int) -> List[dict]:
        """Generate realistic programmer profiles."""
        if num_profiles <= 0:
            raise ValueError("Number of profiles must be positive")

        profiles = []
        for i in range(num_profiles):
            profile = {
                "id": i + 1,
                "name": fake.name(),
                "email": fake.email(),
                "location": fake.city(),
                "skills": self._generate_skills(),
                "projects": self._generate_projects(),
                "certifications": self._generate_certifications(),
            }
            profiles.append(profile)

        return profiles

    def _generate_skills(self) -> List[dict]:
        """Generate realistic programming skills with proficiency levels."""
        all_skills = [
            "Python", "JavaScript", "TypeScript", "Java", "C++", "Go", "Rust",
            "React", "Vue.js", "Angular", "Node.js", "Django", "Flask", "FastAPI",
            "PostgreSQL", "MongoDB", "Redis", "MySQL",
            "AWS", "Docker", "Kubernetes", "Jenkins", "Git",
            "Machine Learning", "Data Science", "DevOps", "Microservices"
        ]

        proficiency_levels = [
            "Beginner", "Intermediate", "Advanced", "Expert"
        ]

        num_skills = random.randint(5, 12)
        selected_skills = random.sample(all_skills, num_skills)

        skills_with_proficiency = []
        for skill in selected_skills:
            # Weight proficiency levels - more intermediate/advanced than beginner/expert
            proficiency = random.choices(
                proficiency_levels,
                weights=[10, 40, 35, 15]  # Beginner, Intermediate, Advanced, Expert
            )[0]

            skills_with_proficiency.append({
                "name": skill,
                "proficiency": proficiency
            })

        return skills_with_proficiency

    def _generate_projects(self) -> List[str]:
        """Generate realistic project names."""
        project_types = [
            "E-commerce Platform", "Data Analytics Dashboard", "Mobile App",
            "API Gateway", "Machine Learning Pipeline", "Web Application",
            "Microservices Architecture", "Real-time Chat System",
            "Content Management System", "Payment Processing System"
        ]
        num_projects = random.randint(2, 5)
        return random.sample(project_types, num_projects)

    def _generate_certifications(self) -> List[str]:
        """Generate realistic certifications."""
        certs = [
            "AWS Certified Solutions Architect",
            "Google Cloud Professional",
            "Certified Kubernetes Administrator",
            "Microsoft Azure Developer",
            "Scrum Master Certification",
            "Docker Certified Associate"
        ]
        num_certs = random.randint(0, 3)
        return random.sample(certs, num_certs) if num_certs > 0 else []

    def generate_cv_markdown(self, profile: dict) -> str:
        """Generate realistic CV in markdown format using LLM."""

        # Format skills with proficiency levels for the prompt
        skills_text = []
        for skill in profile['skills']:
            skills_text.append(f"{skill['name']} ({skill['proficiency']})")

        prompt = f"""
Create a professional CV in markdown format for a programmer with the following details:

Name: {profile['name']}
Email: {profile['email']}
Location: {profile['location']}
Skills: {', '.join(skills_text)}
Projects: {', '.join(profile['projects'])}
Certifications: {', '.join(profile['certifications'])}

Requirements:
1. Use proper markdown formatting (headers, lists, emphasis)
2. Create realistic content with specific details and achievements
3. Include sections like: Summary, Experience, Skills, Projects, Education, etc.
4. Make it unique and personal - vary the structure and tone
5. Add realistic company names, dates, and project descriptions
6. Include specific metrics and achievements where appropriate
7. IMPORTANT: Use the proficiency levels provided for each skill (Beginner, Intermediate, Advanced, Expert) in your skills sections

Make each CV feel authentic and written by a real person, not a template.
Use markdown syntax like # for headers, - for bullet points, **bold**, etc.
Incorporate the skill proficiency levels naturally in the CV (e.g., "Advanced Python", "Expert React developer", etc.).

IMPORTANT: Return ONLY the CV content in markdown format. Do NOT include any code block markers like ```markdown or ``` in your response.
"""

        response = self.llm.invoke(prompt)
        content = response.content

        # Clean up markdown artifacts
        content = content.replace("```markdown", "").replace("```", "")
        content = content.strip()

        if not content:
            raise ValueError(f"LLM returned empty content for {profile['name']}")

        return content

    def save_cv_as_pdf(self, markdown_content: str, filename: str, output_dir: str) -> str:
        """Convert markdown CV to PDF."""
        os.makedirs(output_dir, exist_ok=True)

        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content)

        # Professional CSS styling
        css_content = """
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
        }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        strong { color: #2c3e50; }
        ul { margin-left: 20px; }
        """

        # Generate PDF
        pdf_path = os.path.join(output_dir, f"{filename}.pdf")
        HTML(string=html_content).write_pdf(
            pdf_path,
            stylesheets=[CSS(string=css_content)]
        )

        return pdf_path

    def generate_all_data(self, num_programmers: int = 10) -> dict:
        """Generate all data: profiles and CVs."""
        if num_programmers <= 0:
            raise ValueError("Number of programmers must be positive")

        print(f"Generating {num_programmers} programmer profiles and CVs...")

        # Create output directory
        output_dir = "data/programmers"
        os.makedirs(output_dir, exist_ok=True)

        # Generate programmer profiles
        profiles = self.generate_programmer_profiles(num_programmers)

        # Generate CVs
        generated_files = []
        for i, profile in enumerate(profiles, 1):
            print(f"Generating CV {i}/{num_programmers}: {profile['name']}")

            # Generate markdown CV
            cv_markdown = self.generate_cv_markdown(profile)

            # Save as PDF
            safe_name = profile['name'].replace(" ", "_").replace(".", "")
            filename = f"cv_{profile['id']:03d}_{safe_name}"

            file_path = self.save_cv_as_pdf(cv_markdown, filename, output_dir)
            generated_files.append(file_path)

        # Save profiles as JSON
        profiles_path = os.path.join(output_dir, "programmer_profiles.json")
        with open(profiles_path, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, default=str)

        print(f"✅ Generated {len(generated_files)} CVs in {output_dir}/")
        print(f"✅ Saved profiles to {profiles_path}")

        return {
            "profiles": profiles,
            "cv_files": generated_files,
            "profiles_file": profiles_path
        }


def main():
    """Generate data for GraphRAG demonstration."""
    try:
        generator = GraphRAGDataGenerator()
        result = generator.generate_all_data(10)

        print(f"\nGenerated files:")
        for file_path in result["cv_files"]:
            print(f"  - {file_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Ensure all dependencies are installed: uv sync")
        print("Ensure OPENAI_API_KEY is set in .env file")
        raise


if __name__ == "__main__":
    main()