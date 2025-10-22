"""
Simplified Service Description Schemas for Agentic Marketplace

This module defines simple data structures for service descriptions that will be
stored on IPFS. These schemas focus on essential information needed for the
current stage of development.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class ServiceCategory(str, Enum):
    """Basic categories of services available in the marketplace."""
    AI_ANALYSIS = "ai_analysis"
    DATA_PROCESSING = "data_processing"
    OTHER = "other"


class ServiceRequirements(BaseModel):
    """Simple requirements for a service."""
    
    max_response_time: str = Field(
        default="24 hours",
        description="Maximum time to deliver results"
    )
    
    input_description: str = Field(
        description="Description of expected input data"
    )
    
    output_description: str = Field(
        description="Description of what will be delivered"
    )


class ServiceDescription(BaseModel):
    """
    Simple service description that will be stored on IPFS.
    
    Contains only essential information for the current marketplace stage.
    """
    
    # Basic Information
    title: str = Field(description="Service title/name")
    description: str = Field(description="Detailed service description")
    category: ServiceCategory = Field(description="Service category")
    
    # What the service provides
    deliverables: List[str] = Field(
        description="List of what the buyer will receive"
    )
    
    # Simple requirements
    requirements: ServiceRequirements = Field(description="Service requirements")
    
    # Metadata
    version: str = Field(default="1.0", description="Service description version")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    # Tags for searchability
    tags: List[str] = Field(
        default=[],
        description="Tags for better discoverability"
    )
    
    @field_validator('deliverables')
    @classmethod
    def validate_deliverables(cls, v):
        """Ensure at least one deliverable is specified."""
        if not v or len(v) == 0:
            raise ValueError("At least one deliverable must be specified")
        return v
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        """Ensure tags are lowercase and unique."""
        return list(set(tag.lower().strip() for tag in v if tag.strip()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(mode='json')
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceDescription":
        """Create instance from dictionary."""
        return cls.model_validate(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ServiceDescription":
        """Create instance from JSON string."""
        return cls.model_validate_json(json_str)


# Helper functions for creating common service types

def create_ai_analysis_service() -> ServiceDescription:
    """Create a sample AI analysis service description."""
    return ServiceDescription(
        title="AI Data Analysis Service",
        description="Comprehensive AI-powered analysis of your data to extract insights and patterns",
        category=ServiceCategory.AI_ANALYSIS,
        deliverables=[
            "Detailed analysis report in JSON format",
            "Key insights and recommendations",
            "Data visualization summary"
        ],
        requirements=ServiceRequirements(
            max_response_time="48 hours",
            input_description="CSV or JSON data file with structured data",
            output_description="Analysis report with insights, statistics, and recommendations"
        ),
        tags=["ai", "analysis", "insights", "data"]
    )


def create_data_processing_service() -> ServiceDescription:
    """Create a sample data processing service description."""
    return ServiceDescription(
        title="Data Processing and Cleaning",
        description="Professional data cleaning, transformation, and preparation service",
        category=ServiceCategory.DATA_PROCESSING,
        deliverables=[
            "Cleaned and processed dataset",
            "Data quality report",
            "Processing methodology documentation"
        ],
        requirements=ServiceRequirements(
            max_response_time="24 hours",
            input_description="Raw data in CSV, JSON, or similar format",
            output_description="Clean, processed dataset ready for analysis or ML"
        ),
        tags=["data", "cleaning", "processing", "etl"]
    )


def create_simple_service(title: str, description: str, 
                         category: ServiceCategory = ServiceCategory.OTHER) -> ServiceDescription:
    """Create a simple service description with minimal required fields."""
    return ServiceDescription(
        title=title,
        description=description,
        category=category,
        deliverables=["Service results in JSON format"],
        requirements=ServiceRequirements(
            max_response_time="24 hours",
            input_description="Input data as specified in service description",
            output_description="Results delivered as JSON"
        )
    )