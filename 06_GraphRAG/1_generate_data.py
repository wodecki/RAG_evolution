"""
Synthetic Data Generation for GraphRAG System
=============================================

Generates realistic synthetic data for 50 programmers, 20 projects, and 3 RFPs
to demonstrate GraphRAG capabilities.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import json
import random
from datetime import date, datetime, timedelta
from faker import Faker
from typing import List
import os

from utils.models import (
    ProgrammerProfile, Skill, Certification, ProjectExperience, Project,
    ProjectRequirement, RFP, RFPRequirement, SkillCategory, ProjectStatus,
    ProgrammerRole
)

fake = Faker()
random.seed(42)  # For reproducible data

# Skill definitions with realistic distributions
SKILLS_DATA = {
    SkillCategory.PROGRAMMING_LANGUAGE: [
        "Python", "JavaScript", "Java", "TypeScript", "C#", "Go", "Rust",
        "C++", "PHP", "Ruby", "Kotlin", "Swift", "Scala"
    ],
    SkillCategory.FRAMEWORK: [
        "React", "Vue.js", "Angular", "Django", "Flask", "FastAPI", "Express.js",
        "Spring Boot", "ASP.NET", "Laravel", "Ruby on Rails", "Flutter", "React Native"
    ],
    SkillCategory.DATABASE: [
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
        "DynamoDB", "Oracle", "SQL Server", "SQLite", "Neo4j", "InfluxDB"
    ],
    SkillCategory.CLOUD: [
        "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Terraform",
        "CloudFormation", "Serverless", "Lambda", "EC2", "S3", "RDS"
    ],
    SkillCategory.DEVOPS: [
        "Jenkins", "GitLab CI", "GitHub Actions", "CircleCI", "Ansible",
        "Chef", "Puppet", "Nagios", "Prometheus", "Grafana", "ELK Stack"
    ],
    SkillCategory.FRONTEND: [
        "HTML5", "CSS3", "Sass", "Webpack", "Vite", "Bootstrap", "Tailwind CSS",
        "Material-UI", "Figma", "Adobe XD", "Responsive Design", "PWA"
    ],
    SkillCategory.BACKEND: [
        "REST APIs", "GraphQL", "Microservices", "Event Sourcing", "CQRS",
        "Message Queues", "WebSockets", "gRPC", "API Gateway", "Load Balancing"
    ],
    SkillCategory.MOBILE: [
        "iOS Development", "Android Development", "React Native", "Flutter",
        "Xamarin", "Ionic", "Cordova", "Swift", "Kotlin", "Objective-C"
    ],
    SkillCategory.DATA_SCIENCE: [
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "scikit-learn",
        "Pandas", "NumPy", "Data Analysis", "Statistics", "R", "Jupyter", "Apache Spark"
    ],
    SkillCategory.SECURITY: [
        "Cybersecurity", "Penetration Testing", "OWASP", "SSL/TLS", "OAuth",
        "JWT", "Encryption", "Vulnerability Assessment", "Security Auditing"
    ]
}

CERTIFICATIONS_DATA = {
    "AWS": ["AWS Solutions Architect", "AWS Developer", "AWS DevOps Engineer", "AWS Security Specialist"],
    "Azure": ["Azure Solutions Architect", "Azure Developer", "Azure Administrator", "Azure Security Engineer"],
    "Google Cloud": ["GCP Professional Cloud Architect", "GCP Professional Data Engineer", "GCP Associate Cloud Engineer"],
    "Security": ["CISSP", "CEH", "CISM", "CompTIA Security+", "OSCP"],
    "Project Management": ["PMP", "Scrum Master", "Product Owner", "Agile Coach"],
    "Development": ["Oracle Certified Java Programmer", "Microsoft Certified Developer", "MongoDB Certified Developer"]
}

COMPANIES = [
    "TechCorp", "DataSystems Inc", "CloudNative Solutions", "FinTech Innovations",
    "E-commerce Giants", "Healthcare Systems", "EduTech Solutions", "Gaming Studios",
    "IoT Devices Corp", "Blockchain Ventures", "AI Research Labs", "Cybersecurity Inc",
    "Mobile First", "Social Media Platform", "Streaming Services", "Logistics Tech"
]

PROJECT_TYPES = [
    "Web Application", "Mobile App", "Data Pipeline", "Machine Learning Platform",
    "E-commerce System", "CRM Solution", "Analytics Dashboard", "API Gateway",
    "Microservices Migration", "Cloud Migration", "Security Audit", "DevOps Pipeline"
]

def generate_programmer_skills() -> List[Skill]:
    """Generate realistic skill set for a programmer."""
    skills = []

    # Core programming languages (1-3)
    prog_langs = random.sample(SKILLS_DATA[SkillCategory.PROGRAMMING_LANGUAGE], random.randint(1, 3))
    for lang in prog_langs:
        skills.append(Skill(
            name=lang,
            category=SkillCategory.PROGRAMMING_LANGUAGE,
            proficiency=random.randint(3, 5),
            years_experience=random.randint(2, 10)
        ))

    # Frameworks related to programming languages
    if "Python" in prog_langs:
        frameworks = random.sample(["Django", "Flask", "FastAPI"], random.randint(1, 2))
        for fw in frameworks:
            skills.append(Skill(
                name=fw,
                category=SkillCategory.FRAMEWORK,
                proficiency=random.randint(2, 5),
                years_experience=random.randint(1, 8)
            ))

    if "JavaScript" in prog_langs or "TypeScript" in prog_langs:
        frameworks = random.sample(["React", "Vue.js", "Angular", "Express.js"], random.randint(1, 3))
        for fw in frameworks:
            skills.append(Skill(
                name=fw,
                category=SkillCategory.FRAMEWORK,
                proficiency=random.randint(2, 5),
                years_experience=random.randint(1, 6)
            ))

    # Add other skills
    for category in [SkillCategory.DATABASE, SkillCategory.CLOUD, SkillCategory.DEVOPS]:
        category_skills = random.sample(SKILLS_DATA[category], random.randint(1, 3))
        for skill_name in category_skills:
            skills.append(Skill(
                name=skill_name,
                category=category,
                proficiency=random.randint(2, 4),
                years_experience=random.randint(1, 7)
            ))

    # Specialty skills (data science, security, etc.)
    if random.random() < 0.3:  # 30% chance
        specialty = random.choice([SkillCategory.DATA_SCIENCE, SkillCategory.SECURITY, SkillCategory.MOBILE])
        specialty_skills = random.sample(SKILLS_DATA[specialty], random.randint(1, 2))
        for skill_name in specialty_skills:
            skills.append(Skill(
                name=skill_name,
                category=specialty,
                proficiency=random.randint(3, 5),
                years_experience=random.randint(2, 8)
            ))

    return skills

def generate_certifications(skills: List[Skill]) -> List[Certification]:
    """Generate certifications based on programmer's skills."""
    certifications = []

    # AWS certifications for cloud skills
    has_aws = any(skill.name == "AWS" for skill in skills)
    if has_aws and random.random() < 0.7:
        cert_name = random.choice(CERTIFICATIONS_DATA["AWS"])
        obtained = fake.date_between(start_date="-3y", end_date="today")
        expiry = obtained + timedelta(days=365*3)  # 3-year validity

        certifications.append(Certification(
            name=cert_name,
            provider="Amazon Web Services",
            obtained_date=obtained,
            expiry_date=expiry,
            credential_id=fake.uuid4()[:12]
        ))

    # Similar logic for other cloud providers
    has_azure = any(skill.name == "Azure" for skill in skills)
    if has_azure and random.random() < 0.6:
        cert_name = random.choice(CERTIFICATIONS_DATA["Azure"])
        obtained = fake.date_between(start_date="-2y", end_date="today")
        expiry = obtained + timedelta(days=365*2)

        certifications.append(Certification(
            name=cert_name,
            provider="Microsoft",
            obtained_date=obtained,
            expiry_date=expiry,
            credential_id=fake.uuid4()[:12]
        ))

    # Security certifications
    has_security = any(skill.category == SkillCategory.SECURITY for skill in skills)
    if has_security and random.random() < 0.8:
        cert_name = random.choice(CERTIFICATIONS_DATA["Security"])
        obtained = fake.date_between(start_date="-4y", end_date="today")

        certifications.append(Certification(
            name=cert_name,
            provider="Security Institute",
            obtained_date=obtained,
            expiry_date=None if cert_name == "CISSP" else obtained + timedelta(days=365*3),
            credential_id=fake.uuid4()[:12]
        ))

    return certifications

def generate_project_experience(programmer_id: str, skills: List[Skill]) -> List[ProjectExperience]:
    """Generate realistic project experience."""
    projects = []
    num_projects = random.randint(3, 8)

    current_date = date.today()
    project_end = current_date - timedelta(days=random.randint(30, 90))

    for i in range(num_projects):
        duration_months = random.randint(3, 18)
        start_date = project_end - timedelta(days=duration_months * 30)

        # Select technologies based on programmer's skills
        tech_skills = [s.name for s in skills if s.category in [
            SkillCategory.PROGRAMMING_LANGUAGE, SkillCategory.FRAMEWORK, SkillCategory.DATABASE
        ]]
        technologies = random.sample(tech_skills, min(len(tech_skills), random.randint(2, 5)))

        project = ProjectExperience(
            project_name=f"{random.choice(PROJECT_TYPES)} for {random.choice(COMPANIES)}",
            client=random.choice(COMPANIES),
            role=random.choice(list(ProgrammerRole)),
            start_date=start_date,
            end_date=project_end if i > 0 else None,  # Current project has no end date
            allocation_percentage=random.choice([50, 80, 100]),
            description=fake.paragraph(),
            technologies_used=technologies,
            team_size=random.randint(3, 12)
        )

        projects.append(project)
        project_end = start_date - timedelta(days=random.randint(7, 60))

    return projects

def generate_programmers(count: int = 50) -> List[ProgrammerProfile]:
    """Generate synthetic programmer profiles."""
    programmers = []

    for i in range(count):
        programmer_id = f"dev_{i+1:03d}"

        skills = generate_programmer_skills()
        certifications = generate_certifications(skills)
        project_experience = generate_project_experience(programmer_id, skills)

        # Calculate hourly rate based on skills and experience
        avg_proficiency = sum(s.proficiency for s in skills) / len(skills)
        max_experience = max(s.years_experience for s in skills)
        base_rate = 80 + (avg_proficiency - 1) * 20 + max_experience * 5
        hourly_rate = round(base_rate + random.uniform(-20, 30), 2)

        # Availability
        availability = None
        if random.random() < 0.3:  # 30% immediately available
            availability = date.today()
        elif random.random() < 0.6:  # 60% available in the future
            availability = date.today() + timedelta(days=random.randint(30, 180))

        programmer = ProgrammerProfile(
            id=programmer_id,
            name=fake.name(),
            email=fake.email(),
            phone=fake.phone_number(),
            location=fake.city() + ", " + fake.state_abbr(),
            hourly_rate=hourly_rate,
            availability_start=availability,
            bio=fake.paragraph(),
            linkedin_url=f"https://linkedin.com/in/{fake.user_name()}",
            github_url=f"https://github.com/{fake.user_name()}",
            skills=skills,
            certifications=certifications,
            project_experience=project_experience
        )

        programmers.append(programmer)

    return programmers

def generate_projects(count: int = 20) -> List[Project]:
    """Generate synthetic project data."""
    projects = []

    for i in range(count):
        project_id = f"proj_{i+1:03d}"

        # Generate requirements
        requirements = []
        num_requirements = random.randint(3, 7)

        # Select random skills for requirements
        all_skills = []
        for skills_list in SKILLS_DATA.values():
            all_skills.extend(skills_list)

        required_skills = random.sample(all_skills, num_requirements)
        for skill in required_skills:
            requirements.append(ProjectRequirement(
                skill_name=skill,
                min_proficiency=random.randint(2, 4),
                min_years=random.randint(1, 5),
                is_mandatory=random.random() < 0.7  # 70% mandatory
            ))

        # Project dates
        start_date = fake.date_between(start_date="-6m", end_date="+6m")
        duration_months = random.randint(2, 12)
        end_date = start_date + timedelta(days=duration_months * 30)

        project = Project(
            id=project_id,
            name=f"{random.choice(PROJECT_TYPES)} - {fake.company()}",
            client=fake.company(),
            description=fake.paragraph(),
            start_date=start_date,
            end_date=end_date,
            estimated_duration_months=duration_months,
            budget=random.randint(50000, 500000),
            status=random.choice(list(ProjectStatus)),
            team_size=random.randint(2, 8),
            requirements=requirements,
            assigned_programmers=[]  # Will be populated later
        )

        projects.append(project)

    return projects

def generate_rfps(count: int = 3) -> List[RFP]:
    """Generate synthetic RFP data."""
    rfps = []

    rfp_scenarios = [
        {
            "title": "FinTech Payment Platform Development",
            "client": "NextGen Bank",
            "description": "Build a secure, scalable payment processing platform with real-time fraud detection and multi-currency support.",
            "project_type": "Financial Technology",
            "skills": ["Python", "Django", "PostgreSQL", "Redis", "AWS", "Docker", "Machine Learning"]
        },
        {
            "title": "E-commerce Mobile Application",
            "client": "RetailMax Corporation",
            "description": "Develop cross-platform mobile application for e-commerce with AR features, social integration, and advanced analytics.",
            "project_type": "Mobile Development",
            "skills": ["React Native", "TypeScript", "Node.js", "MongoDB", "AWS", "GraphQL", "AR/VR"]
        },
        {
            "title": "Healthcare Data Analytics Platform",
            "client": "MedTech Solutions",
            "description": "Create HIPAA-compliant analytics platform for processing medical data with ML-powered insights and predictive modeling.",
            "project_type": "Healthcare Technology",
            "skills": ["Python", "TensorFlow", "PostgreSQL", "Kubernetes", "Azure", "HIPAA Compliance", "Data Science"]
        }
    ]

    for i, scenario in enumerate(rfp_scenarios):
        rfp_id = f"rfp_{i+1:03d}"

        requirements = []
        for skill in scenario["skills"]:
            requirements.append(RFPRequirement(
                skill_name=skill,
                min_proficiency=random.randint(3, 5),
                min_years=random.randint(2, 7),
                is_mandatory=random.random() < 0.8,
                preferred_certifications=[]
            ))

        # Add some preferred certifications
        if "AWS" in scenario["skills"]:
            requirements[0].preferred_certifications = ["AWS Solutions Architect", "AWS Developer"]

        rfp = RFP(
            id=rfp_id,
            title=scenario["title"],
            client=scenario["client"],
            description=scenario["description"],
            project_type=scenario["project_type"],
            duration_months=random.randint(6, 18),
            team_size=random.randint(4, 10),
            budget_range=f"${random.randint(200, 800)}K - ${random.randint(800, 1500)}K",
            start_date=date.today() + timedelta(days=random.randint(30, 90)),
            requirements=requirements,
            location=fake.city() + ", " + fake.state_abbr(),
            remote_allowed=random.choice([True, False])
        )

        rfps.append(rfp)

    return rfps

def save_data_to_files(programmers: List[ProgrammerProfile], projects: List[Project], rfps: List[RFP]):
    """Save generated data to JSON files."""

    # Custom JSON encoder for datetime and date objects
    def json_serializer(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Save programmers
    for programmer in programmers:
        filename = f"data/programmers/{programmer.id}.json"
        with open(filename, 'w') as f:
            json.dump(programmer.model_dump(), f, indent=2, default=json_serializer)

    # Save projects
    for project in projects:
        filename = f"data/projects/{project.id}.json"
        with open(filename, 'w') as f:
            json.dump(project.model_dump(), f, indent=2, default=json_serializer)

    # Save RFPs
    for rfp in rfps:
        filename = f"data/rfps/{rfp.id}.json"
        with open(filename, 'w') as f:
            json.dump(rfp.model_dump(), f, indent=2, default=json_serializer)

    print(f"✓ Saved {len(programmers)} programmer profiles")
    print(f"✓ Saved {len(projects)} projects")
    print(f"✓ Saved {len(rfps)} RFPs")

def generate_summary_statistics(programmers: List[ProgrammerProfile], projects: List[Project], rfps: List[RFP]):
    """Generate and display summary statistics."""

    print("\n" + "="*50)
    print("DATA GENERATION SUMMARY")
    print("="*50)

    # Programmer statistics
    total_skills = sum(len(p.skills) for p in programmers)
    total_certs = sum(len(p.certifications) for p in programmers)
    avg_experience = sum(max(s.years_experience for s in p.skills) for p in programmers) / len(programmers)

    print(f"\nProgrammer Statistics:")
    print(f"- Total programmers: {len(programmers)}")
    print(f"- Total skills: {total_skills}")
    print(f"- Total certifications: {total_certs}")
    print(f"- Average max experience: {avg_experience:.1f} years")

    # Skill distribution
    skill_counts = {}
    for programmer in programmers:
        for skill in programmer.skills:
            skill_counts[skill.name] = skill_counts.get(skill.name, 0) + 1

    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 Skills:")
    for skill, count in top_skills:
        print(f"- {skill}: {count} programmers")

    # Project statistics
    print(f"\nProject Statistics:")
    print(f"- Total projects: {len(projects)}")
    total_requirements = sum(len(p.requirements) for p in projects)
    print(f"- Total skill requirements: {total_requirements}")

    # RFP statistics
    print(f"\nRFP Statistics:")
    print(f"- Total RFPs: {len(rfps)}")
    for rfp in rfps:
        print(f"- {rfp.title}: {len(rfp.requirements)} requirements, {rfp.team_size} team size")

def main():
    """Main data generation function."""
    print("Generating Synthetic Data for GraphRAG System")
    print("=" * 50)

    # Generate data
    print("Generating programmer profiles...")
    programmers = generate_programmers(50)

    print("Generating projects...")
    projects = generate_projects(20)

    print("Generating RFPs...")
    rfps = generate_rfps(3)

    # Save to files
    print("\nSaving data to files...")
    save_data_to_files(programmers, projects, rfps)

    # Display statistics
    generate_summary_statistics(programmers, projects, rfps)

    print("\n" + "="*50)
    print("✓ Data generation completed successfully!")
    print("\nNext steps:")
    print("1. Run: uv run python 2_build_knowledge_graph.py")
    print("2. Verify data in data/ directories")

if __name__ == "__main__":
    main()