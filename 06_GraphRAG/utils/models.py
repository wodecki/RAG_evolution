"""
Data Models for GraphRAG Programmer Staffing System
==================================================

Pydantic models for type safety and data validation.
"""

from typing import List, Optional, Dict, Any
from datetime import date, datetime
from pydantic import BaseModel, Field
from enum import Enum

class SkillCategory(str, Enum):
    PROGRAMMING_LANGUAGE = "Programming Language"
    FRAMEWORK = "Framework"
    DATABASE = "Database"
    CLOUD = "Cloud"
    DEVOPS = "DevOps"
    FRONTEND = "Frontend"
    BACKEND = "Backend"
    MOBILE = "Mobile"
    DATA_SCIENCE = "Data Science"
    SECURITY = "Security"

class ProjectStatus(str, Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"

class ProgrammerRole(str, Enum):
    JUNIOR = "Junior Developer"
    MID_LEVEL = "Mid-level Developer"
    SENIOR = "Senior Developer"
    LEAD = "Lead Developer"
    ARCHITECT = "Architect"
    TECH_LEAD = "Tech Lead"
    PRINCIPAL = "Principal Engineer"

class Skill(BaseModel):
    name: str = Field(..., description="Name of the skill")
    category: SkillCategory = Field(..., description="Category of the skill")
    proficiency: int = Field(..., ge=1, le=5, description="Proficiency level 1-5")
    years_experience: int = Field(..., ge=0, description="Years of experience")

class Certification(BaseModel):
    name: str = Field(..., description="Name of the certification")
    provider: str = Field(..., description="Certification provider")
    obtained_date: date = Field(..., description="Date certification was obtained")
    expiry_date: Optional[date] = Field(None, description="Expiry date if applicable")
    credential_id: Optional[str] = Field(None, description="Credential ID")

class ProjectExperience(BaseModel):
    project_name: str = Field(..., description="Name of the project")
    client: str = Field(..., description="Client or company name")
    role: ProgrammerRole = Field(..., description="Role in the project")
    start_date: date = Field(..., description="Project start date")
    end_date: Optional[date] = Field(None, description="Project end date")
    allocation_percentage: int = Field(..., ge=0, le=100, description="Percentage allocation")
    description: str = Field(..., description="Project description")
    technologies_used: List[str] = Field(default_factory=list, description="Technologies used")
    team_size: Optional[int] = Field(None, ge=1, description="Team size")

class ProgrammerProfile(BaseModel):
    id: str = Field(..., description="Unique programmer ID")
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: str = Field(..., description="Current location")
    hourly_rate: float = Field(..., ge=0, description="Hourly rate in USD")
    availability_start: Optional[date] = Field(None, description="Next available start date")
    bio: str = Field(..., description="Professional bio")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")
    github_url: Optional[str] = Field(None, description="GitHub profile URL")
    skills: List[Skill] = Field(default_factory=list, description="List of skills")
    certifications: List[Certification] = Field(default_factory=list, description="List of certifications")
    project_experience: List[ProjectExperience] = Field(default_factory=list, description="Project history")
    created_at: datetime = Field(default_factory=datetime.now, description="Profile creation timestamp")

class ProjectRequirement(BaseModel):
    skill_name: str = Field(..., description="Required skill name")
    min_proficiency: int = Field(..., ge=1, le=5, description="Minimum proficiency level")
    min_years: int = Field(..., ge=0, description="Minimum years of experience")
    is_mandatory: bool = Field(True, description="Whether this skill is mandatory")

class Project(BaseModel):
    id: str = Field(..., description="Unique project ID")
    name: str = Field(..., description="Project name")
    client: str = Field(..., description="Client company")
    description: str = Field(..., description="Project description")
    start_date: date = Field(..., description="Project start date")
    end_date: Optional[date] = Field(None, description="Project end date")
    estimated_duration_months: int = Field(..., ge=1, description="Estimated duration in months")
    budget: Optional[float] = Field(None, ge=0, description="Project budget")
    status: ProjectStatus = Field(..., description="Current project status")
    team_size: int = Field(..., ge=1, description="Required team size")
    requirements: List[ProjectRequirement] = Field(default_factory=list, description="Skill requirements")
    assigned_programmers: List[str] = Field(default_factory=list, description="List of assigned programmer IDs")
    created_at: datetime = Field(default_factory=datetime.now, description="Project creation timestamp")

class RFPRequirement(BaseModel):
    skill_name: str = Field(..., description="Required skill")
    min_proficiency: int = Field(..., ge=1, le=5, description="Minimum proficiency")
    min_years: int = Field(..., ge=0, description="Minimum years experience")
    is_mandatory: bool = Field(True, description="Is this skill mandatory")
    preferred_certifications: List[str] = Field(default_factory=list, description="Preferred certifications")

class RFP(BaseModel):
    id: str = Field(..., description="Unique RFP ID")
    title: str = Field(..., description="RFP title")
    client: str = Field(..., description="Client company")
    description: str = Field(..., description="Detailed RFP description")
    project_type: str = Field(..., description="Type of project")
    duration_months: int = Field(..., ge=1, description="Project duration in months")
    team_size: int = Field(..., ge=1, description="Required team size")
    budget_range: Optional[str] = Field(None, description="Budget range")
    start_date: date = Field(..., description="Desired start date")
    requirements: List[RFPRequirement] = Field(default_factory=list, description="Skill requirements")
    location: str = Field(..., description="Work location")
    remote_allowed: bool = Field(True, description="Is remote work allowed")
    created_at: datetime = Field(default_factory=datetime.now, description="RFP creation timestamp")

class QueryResult(BaseModel):
    """Result of a RAG query."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    source_type: str = Field(..., description="naive_rag or graph_rag")
    context: List[str] = Field(default_factory=list, description="Retrieved context")
    cypher_query: Optional[str] = Field(None, description="Generated Cypher query if GraphRAG")
    execution_time: float = Field(..., description="Query execution time in seconds")
    confidence_score: Optional[float] = Field(None, description="Confidence score if available")
    timestamp: datetime = Field(default_factory=datetime.now, description="Query timestamp")