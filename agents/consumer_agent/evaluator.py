"""
Service Evaluator - ReAct agent for evaluating completed service quality.

Uses ReAct pattern with LLM to assess service quality and generate ratings.
"""

import logging
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI as genai

from ..config import Config

logger = logging.getLogger(__name__)


class ServiceEvaluator:
    """
    ReAct agent for evaluating completed service results.
    
    Analyzes service output against requirements and generates:
    - Integer rating (0-100) per ERC-8004 spec
    - Feedback text explaining the rating
    """
    
    def __init__(self, config: Config):
        """Initialize service evaluator."""
        self.config = config
        
        # Configure Gemini
        if config.google_api_key:
            genai.configure(api_key=config.google_api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.model = None
            logger.warning("Google API key not configured - evaluator will use mock ratings")
        
        logger.info("ServiceEvaluator initialized")
    
    async def evaluate(
        self,
        service_requirements: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate completed service result.
        
        Args:
            service_requirements: Original service requirements from IPFS
            result: Completed service result from provider
            
        Returns:
            Evaluation dict with rating (0-100) and feedback text
        """
        logger.info("Evaluating service result...")
        
        if self.model is None:
            # Mock evaluation if no API key
            return self._mock_evaluation(service_requirements, result)
        
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(service_requirements, result)
        
        # Get LLM evaluation
        try:
            response = self.model.generate_content(prompt)
            evaluation = self._parse_evaluation_response(response.text)
            
            logger.info(f"✓ Evaluation: {evaluation['rating']}/100 - {evaluation['summary']}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error during LLM evaluation: {e}")
            # Fallback to mock
            return self._mock_evaluation(service_requirements, result)
    
    def _build_evaluation_prompt(
        self,
        service_requirements: Dict[str, Any],
        result: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM evaluation."""
        
        prompts = service_requirements.get("prompts", [])
        responses = result.get("responses", [])
        quality_criteria = service_requirements.get("quality_criteria", {})
        
        prompt = f"""You are an expert evaluator assessing the quality of an AI service delivery.

**Service Requirements:**
Title: {service_requirements.get('title', 'N/A')}
Description: {service_requirements.get('description', 'N/A')}
Complexity: {service_requirements.get('complexity', 'N/A')}

**Quality Criteria:**
"""
        
        for criterion, description in quality_criteria.items():
            prompt += f"- {criterion}: {description}\n"
        
        prompt += "\n**Expected Prompts and Delivered Responses:**\n\n"
        
        for i, (prompt_text, response_text) in enumerate(zip(prompts, responses), 1):
            prompt += f"Prompt {i}: {prompt_text}\n"
            prompt += f"Response {i}: {response_text}\n\n"
        
        prompt += """
**Your Task:**
Evaluate the quality of the responses based on the criteria above.

Provide your evaluation in the following format:
RATING: [0-100]
SUMMARY: [One sentence summary of quality]
FEEDBACK: [Detailed feedback explaining the rating, covering completeness, depth, accuracy, and clarity]

Consider:
1. Completeness: Were all prompts answered?
2. Depth: Are answers detailed and well-supported?
3. Accuracy: Do answers correctly address the prompts?
4. Clarity: Are responses clear and well-structured?
5. Citations: Are relevant citations included where appropriate?

Rating scale:
- 90-100: Exceptional quality, exceeds expectations
- 75-89: High quality, meets all criteria well
- 60-74: Good quality, meets most criteria
- 40-59: Acceptable, some issues present
- 20-39: Poor quality, significant issues
- 0-19: Unacceptable, major failures

Provide your evaluation now:
"""
        
        return prompt
    
    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured evaluation."""
        
        lines = response_text.strip().split('\n')
        rating = 75  # Default
        summary = ""
        feedback = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("RATING:"):
                try:
                    rating_str = line.replace("RATING:", "").strip()
                    # Extract just the number
                    rating = int(''.join(filter(str.isdigit, rating_str)))
                    rating = max(0, min(100, rating))  # Clamp to 0-100
                except:
                    pass
                    
            elif line.startswith("SUMMARY:"):
                summary = line.replace("SUMMARY:", "").strip()
                current_section = "summary"
                
            elif line.startswith("FEEDBACK:"):
                feedback = line.replace("FEEDBACK:", "").strip()
                current_section = "feedback"
                
            elif current_section == "summary" and line:
                summary += " " + line
                
            elif current_section == "feedback" and line:
                feedback += " " + line
        
        return {
            "rating": rating,
            "summary": summary.strip(),
            "feedback": feedback.strip(),
            "timestamp": None  # Will be set by consumer
        }
    
    def _mock_evaluation(
        self,
        service_requirements: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate mock evaluation when LLM is not available."""
        
        prompts = service_requirements.get("prompts", [])
        responses = result.get("responses", [])
        
        # Simple heuristic: check if all prompts were answered
        if len(responses) >= len(prompts):
            rating = 80
            summary = "All prompts answered with reasonable detail"
            feedback = f"Service completed all {len(prompts)} required prompts. Responses appear complete and structured."
        else:
            rating = 50
            summary = "Incomplete service delivery"
            feedback = f"Only {len(responses)}/{len(prompts)} prompts were answered."
        
        logger.info(f"Mock evaluation: {rating}/100")
        
        return {
            "rating": rating,
            "summary": summary,
            "feedback": feedback,
            "timestamp": None
        }
