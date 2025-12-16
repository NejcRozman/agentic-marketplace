"""
Service Generator - ReAct agent for generating service requirements.

Uses ReAct pattern to select and customize service templates for auctions.
"""

import logging
from typing import Dict, Any, List
from pathlib import Path
import json

from ..config import Config

logger = logging.getLogger(__name__)


class ServiceGenerator:
    """
    ReAct agent for generating service requirement descriptions.
    
    Selects from predefined templates and customizes parameters.
    """
    
    def __init__(self, config: Config):
        """Initialize service generator."""
        self.config = config
        self.templates_dir = Path(__file__).parent.parent / "templates" / "services"
        
        logger.info("ServiceGenerator initialized")
    
    async def generate_service(
        self,
        template_name: str = "literature_review",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate service requirements from template.
        
        Args:
            template_name: Name of service template to use
            **kwargs: Template-specific parameters
            
        Returns:
            Service requirements dict ready for IPFS upload
        """
        logger.info(f"Generating service from template: {template_name}")
        
        # Load template
        template = self._load_template(template_name)
        
        # Customize template with provided parameters
        service = self._customize_template(template, **kwargs)
        
        logger.info(f"✓ Service generated: {service['title']}")
        return service
    
    def _load_template(self, template_name: str) -> Dict[str, Any]:
        """Load service template from file or return default."""
        
        # Default literature review template
        if template_name == "literature_review":
            return {
                "title": "Literature Review Service",
                "description": "Comprehensive literature review with analysis and synthesis",
                "service_type": "literature_review",
                "prompts": [
                    "What is the main research question or topic addressed in this paper?",
                    "What methodology or approach does the paper use?",
                    "What are the key findings or contributions?",
                    "What are the limitations or areas for future work?"
                ],
                "input_files_cid": None,  # To be set during auction creation
                "complexity": "medium",
                "expected_duration_minutes": 30,
                "quality_criteria": {
                    "completeness": "All prompts must be answered",
                    "depth": "Answers should be detailed and well-supported",
                    "citations": "Include relevant citations from the paper",
                    "clarity": "Clear and well-structured responses"
                }
            }
        
        # Could load other templates from files
        raise ValueError(f"Unknown template: {template_name}")
    
    def _customize_template(
        self,
        template: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Customize template with specific parameters.
        
        Args:
            template: Base template
            **kwargs: Parameters to override (e.g., prompts, complexity, input_files_cid)
            
        Returns:
            Customized service requirements
        """
        service = template.copy()
        
        # Override with provided parameters
        for key, value in kwargs.items():
            if value is not None:
                service[key] = value
        
        return service
    
    def get_available_templates(self) -> List[str]:
        """Get list of available service templates."""
        return ["literature_review"]
