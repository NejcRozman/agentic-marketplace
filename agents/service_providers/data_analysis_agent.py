"""Data Analysis Service Provider Agent using LangGraph."""

import json
import logging
from typing import Any, Dict, List
from datetime import datetime

from ..core.base_agent import BaseAgent, GraphState
from ..core.config import config

logger = logging.getLogger(__name__)


class DataAnalysisAgent(BaseAgent):
    """
    AI agent that provides data analysis services in the marketplace.
    
    This agent can:
    - Analyze datasets provided by clients
    - Generate insights and visualizations
    - Register as a service provider on the blockchain
    - Handle service requests and payments
    """
    
    def __init__(self, agent_id: str = "data-analysis-agent"):
        super().__init__(agent_id)
        self.service_type = "data_analysis"
        self.capabilities = [
            "statistical_analysis",
            "data_visualization", 
            "trend_analysis",
            "predictive_modeling",
            "data_cleaning"
        ]
        
    async def _custom_input_processing(self, state: GraphState) -> None:
        """Process input specific to data analysis requests."""
        # Extract any file attachments or data references
        latest_message = state.messages[-1].content if state.messages else ""
        
        # Check if this is a service registration request
        if "register" in latest_message.lower():
            state.metadata["action"] = "register_service"
            state.metadata["needs_blockchain"] = True
        # Check if this is a data analysis request
        elif any(keyword in latest_message.lower() for keyword in ["analyze", "data", "dataset"]):
            state.metadata["action"] = "analyze_data"
            state.metadata["needs_blockchain"] = False
        # Check if this is a service completion
        elif "complete" in latest_message.lower() or "finished" in latest_message.lower():
            state.metadata["action"] = "complete_service"
            state.metadata["needs_blockchain"] = True
        
        state.metadata["service_type"] = self.service_type
        state.metadata["capabilities"] = self.capabilities
    
    async def _get_analysis_prompt(self, state: GraphState) -> str:
        """Get the analysis prompt for the LLM."""
        action = state.metadata.get("action", "unknown")
        
        if action == "register_service":
            return f"""
You are a data analysis service provider agent in a decentralized marketplace.
The user wants to register your data analysis service.

Your capabilities include: {', '.join(self.capabilities)}

Analyze the user's request and determine:
1. What specific data analysis services they want to offer
2. What pricing they want to set (in tokens)
3. Any special requirements or limitations
4. The reputation score they're claiming (if any)

Respond with a JSON object containing:
- service_description: Brief description of the service
- capabilities: List of specific capabilities
- price_per_request: Suggested price in tokens
- estimated_completion_time: Time estimate in minutes
- requirements: Any special requirements for data format

Current request: {state.messages[-1].content if state.messages else ""}
"""
        
        elif action == "analyze_data":
            return f"""
You are a data analysis expert. The user has requested data analysis services.

Analyze their request and determine:
1. What type of analysis they need
2. What data format they're providing
3. What deliverables they expect
4. Estimated time and complexity

Respond with your analysis plan and any questions you need answered before proceeding.

Current request: {state.messages[-1].content if state.messages else ""}
"""
        
        elif action == "complete_service":
            return f"""
You are completing a data analysis service. 

Review the work done and prepare:
1. Summary of analysis performed
2. Key findings and insights
3. Deliverables provided
4. Quality assessment of the work

Current request: {state.messages[-1].content if state.messages else ""}
"""
        
        else:
            return f"""
You are a data analysis service provider agent. 

Analyze the user's request and determine what they need:
- Service registration
- Data analysis request  
- Service completion
- General inquiry

Current request: {state.messages[-1].content if state.messages else ""}
"""
    
    async def _parse_analysis(self, state: GraphState, analysis: str) -> None:
        """Parse the LLM analysis and update state."""
        action = state.metadata.get("action", "unknown")
        
        if action == "register_service":
            try:
                # Try to extract JSON from the analysis
                json_start = analysis.find('{')
                json_end = analysis.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    service_data = json.loads(analysis[json_start:json_end])
                    state.metadata["service_data"] = service_data
                else:
                    # Fallback: extract key information
                    state.metadata["service_data"] = {
                        "service_description": "Data analysis and insights generation",
                        "capabilities": self.capabilities,
                        "price_per_request": 100,  # Default price in tokens
                        "estimated_completion_time": 60,  # 1 hour
                        "requirements": "CSV, JSON, or structured data format"
                    }
            except json.JSONDecodeError:
                logger.warning("Could not parse service data JSON, using defaults")
                state.metadata["service_data"] = {
                    "service_description": "Data analysis and insights generation",
                    "capabilities": self.capabilities,
                    "price_per_request": 100,
                    "estimated_completion_time": 60,
                    "requirements": "CSV, JSON, or structured data format"
                }
        
        elif action == "analyze_data":
            state.metadata["analysis_plan"] = analysis
        
        elif action == "complete_service":
            state.metadata["completion_summary"] = analysis
    
    async def _perform_blockchain_action(self, state: GraphState) -> Dict[str, Any]:
        """Perform blockchain-specific actions."""
        action = state.metadata.get("action", "unknown")
        
        if action == "register_service":
            # Register as a service provider
            service_data = state.metadata.get("service_data", {})
            
            try:
                # This would call the marketplace contract to register the service
                # tx_hash = await self.blockchain_client.send_transaction(
                #     "marketplace",
                #     "registerService",
                #     self.service_type,
                #     service_data["price_per_request"],
                #     json.dumps(service_data)
                # )
                
                # For now, simulate the registration
                tx_hash = f"0x{'1234567890abcdef' * 4}"
                
                return {
                    "action": "service_registered",
                    "transaction_hash": tx_hash,
                    "service_data": service_data,
                    "agent_address": self.blockchain_client.account.address if self.blockchain_client.account else "0x123...",
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error registering service: {e}")
                return {"error": str(e), "action": "register_service"}
        
        elif action == "complete_service":
            # Mark service as completed and trigger payment
            try:
                # This would call the marketplace contract to complete the service
                # tx_hash = await self.blockchain_client.send_transaction(
                #     "marketplace", 
                #     "completeService",
                #     state.task_id or "default_task"
                # )
                
                # For now, simulate the completion
                tx_hash = f"0x{'fedcba0987654321' * 4}"
                
                return {
                    "action": "service_completed",
                    "transaction_hash": tx_hash,
                    "task_id": state.task_id,
                    "completion_time": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error completing service: {e}")
                return {"error": str(e), "action": "complete_service"}
        
        return {"error": "Unknown blockchain action"}
    
    async def _get_response_prompt(self, state: GraphState) -> str:
        """Get the response generation prompt."""
        action = state.metadata.get("action", "unknown")
        blockchain_data = state.blockchain_data or {}
        
        if action == "register_service":
            return f"""
Generate a response confirming the service registration.

Blockchain result: {json.dumps(blockchain_data, indent=2)}

Include:
1. Confirmation that the service has been registered
2. Service details and capabilities
3. Transaction hash for verification
4. Next steps for receiving service requests

Be professional and informative.
"""
        
        elif action == "analyze_data":
            return f"""
Generate a response with your data analysis plan.

Analysis plan: {state.metadata.get('analysis_plan', 'No plan available')}

Include:
1. What analysis you'll perform
2. Expected deliverables
3. Timeline and next steps
4. Any requirements or clarifications needed

Be clear and professional.
"""
        
        elif action == "complete_service":
            return f"""
Generate a service completion response.

Completion summary: {state.metadata.get('completion_summary', 'No summary available')}
Blockchain result: {json.dumps(blockchain_data, indent=2)}

Include:
1. Summary of work completed
2. Key insights and deliverables
3. Payment/transaction confirmation
4. Thank you message

Be professional and summarize the value provided.
"""
        
        else:
            return "Generate a helpful response to the user's inquiry about data analysis services."