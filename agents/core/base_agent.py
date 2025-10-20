"""Base agent class for agentic marketplace using LangGraph."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .config import config, LLMConfig
from .blockchain_client import BlockchainClient

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_BLOCKCHAIN = "waiting_for_blockchain"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class GraphState:
    """State object for LangGraph workflow."""
    messages: List[BaseMessage]
    agent_state: AgentState
    task_id: Optional[str] = None
    blockchain_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC, Generic[T]):
    """
    Base class for all marketplace agents using LangGraph.
    
    This class provides the foundation for building AI agents that can:
    - Interact with blockchain contracts
    - Process natural language requests
    - Maintain state across complex workflows
    - Handle errors and retries
    """
    
    def __init__(
        self,
        agent_id: str,
        llm_config: Optional[LLMConfig] = None,
        blockchain_client: Optional[BlockchainClient] = None
    ):
        self.agent_id = agent_id
        self.llm_config = llm_config or config.llm
        self.blockchain_client = blockchain_client or BlockchainClient()
        
        # Initialize LLM
        self.llm = self._create_llm()
        
        # Build the agent graph
        self.graph = self._build_graph()
        
        # Agent state
        self.current_state = AgentState.IDLE
        self.active_tasks: Dict[str, Any] = {}
        
        logger.info(f"Initialized agent {agent_id}")
    
    def _create_llm(self) -> BaseChatModel:
        """Create and configure the language model."""
        if self.llm_config.provider.lower() == "openai":
            return ChatOpenAI(
                model=self.llm_config.model_name,
                api_key=self.llm_config.api_key,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                timeout=self.llm_config.timeout,
            )
        elif self.llm_config.provider.lower() == "anthropic":
            return ChatAnthropic(
                model=self.llm_config.model_name,
                api_key=self.llm_config.api_key,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
                timeout=self.llm_config.timeout,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_config.provider}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("analyze_request", self._analyze_request)
        workflow.add_node("interact_blockchain", self._interact_blockchain)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("handle_error", self._handle_error)
        
        # Add edges
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "analyze_request")
        workflow.add_conditional_edges(
            "analyze_request",
            self._should_interact_blockchain,
            {
                "blockchain": "interact_blockchain",
                "response": "generate_response",
                "error": "handle_error"
            }
        )
        workflow.add_edge("interact_blockchain", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def _process_input(self, state: GraphState) -> GraphState:
        """Process and validate input."""
        try:
            logger.debug(f"Processing input for agent {self.agent_id}")
            state.agent_state = AgentState.PROCESSING
            
            # Validate messages
            if not state.messages:
                raise ValueError("No messages provided")
            
            # Add agent-specific processing
            await self._custom_input_processing(state)
            
            return state
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            state.error = str(e)
            state.agent_state = AgentState.ERROR
            return state
    
    async def _analyze_request(self, state: GraphState) -> GraphState:
        """Analyze the request using LLM."""
        try:
            logger.debug(f"Analyzing request for agent {self.agent_id}")
            
            # Get the analysis prompt
            analysis_prompt = await self._get_analysis_prompt(state)
            
            # Add analysis prompt to messages
            state.messages.append(HumanMessage(content=analysis_prompt))
            
            # Get LLM response
            response = await self.llm.ainvoke(state.messages)
            state.messages.append(response)
            
            # Parse the analysis
            await self._parse_analysis(state, response.content)
            
            return state
        except Exception as e:
            logger.error(f"Error analyzing request: {e}")
            state.error = str(e)
            state.agent_state = AgentState.ERROR
            return state
    
    async def _interact_blockchain(self, state: GraphState) -> GraphState:
        """Interact with blockchain contracts."""
        try:
            logger.debug(f"Interacting with blockchain for agent {self.agent_id}")
            state.agent_state = AgentState.WAITING_FOR_BLOCKCHAIN
            
            # Perform blockchain interaction
            blockchain_result = await self._perform_blockchain_action(state)
            state.blockchain_data = blockchain_result
            
            state.agent_state = AgentState.PROCESSING
            return state
        except Exception as e:
            logger.error(f"Error interacting with blockchain: {e}")
            state.error = str(e)
            state.agent_state = AgentState.ERROR
            return state
    
    async def _generate_response(self, state: GraphState) -> GraphState:
        """Generate the final response."""
        try:
            logger.debug(f"Generating response for agent {self.agent_id}")
            
            # Create response prompt
            response_prompt = await self._get_response_prompt(state)
            state.messages.append(HumanMessage(content=response_prompt))
            
            # Generate response
            response = await self.llm.ainvoke(state.messages)
            state.messages.append(response)
            
            # Set result
            state.result = {
                "response": response.content,
                "blockchain_data": state.blockchain_data,
                "metadata": state.metadata
            }
            
            state.agent_state = AgentState.COMPLETED
            return state
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state.error = str(e)
            state.agent_state = AgentState.ERROR
            return state
    
    async def _handle_error(self, state: GraphState) -> GraphState:
        """Handle errors in the workflow."""
        logger.error(f"Handling error for agent {self.agent_id}: {state.error}")
        
        # Create error response
        state.result = {
            "error": state.error,
            "agent_id": self.agent_id,
            "state": state.agent_state.value
        }
        
        return state
    
    def _should_interact_blockchain(self, state: GraphState) -> str:
        """Determine if blockchain interaction is needed."""
        if state.agent_state == AgentState.ERROR:
            return "error"
        
        # Check if blockchain interaction is needed
        if state.metadata.get("needs_blockchain", False):
            return "blockchain"
        
        return "response"
    
    # Abstract methods to be implemented by subclasses
    @abstractmethod
    async def _custom_input_processing(self, state: GraphState) -> None:
        """Custom input processing for specific agent types."""
        pass
    
    @abstractmethod
    async def _get_analysis_prompt(self, state: GraphState) -> str:
        """Get the analysis prompt for the LLM."""
        pass
    
    @abstractmethod
    async def _parse_analysis(self, state: GraphState, analysis: str) -> None:
        """Parse the LLM analysis and update state."""
        pass
    
    @abstractmethod
    async def _perform_blockchain_action(self, state: GraphState) -> Dict[str, Any]:
        """Perform blockchain-specific actions."""
        pass
    
    @abstractmethod
    async def _get_response_prompt(self, state: GraphState) -> str:
        """Get the response generation prompt."""
        pass
    
    # Public interface
    async def process_request(
        self,
        message: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a request through the agent workflow.
        
        Args:
            message: The input message to process
            task_id: Optional task identifier
            metadata: Additional metadata for the request
            
        Returns:
            Dictionary containing the result or error information
        """
        try:
            # Create initial state
            initial_state = GraphState(
                messages=[HumanMessage(content=message)],
                agent_state=AgentState.IDLE,
                task_id=task_id,
                metadata=metadata or {}
            )
            
            # Run the workflow
            result = await self.graph.ainvoke(initial_state)
            
            return result.result or {"error": "No result generated"}
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {"error": str(e), "agent_id": self.agent_id}
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "state": self.current_state.value,
            "active_tasks": len(self.active_tasks),
            "blockchain_connected": await self.blockchain_client.is_connected()
        }