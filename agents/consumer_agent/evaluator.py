"""
Service Evaluator - ReAct agent for evaluating completed service quality.

Uses ReAct pattern with LLM to assess service quality and generate ratings.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from ..config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)


class EvaluatorState(TypedDict):
    """Minimal state for evaluation workflow."""
    # Input
    service_requirements: Dict[str, Any]
    result: Dict[str, Any]
    
    # Internal reasoning (helps ReAct agent think clearly)
    quality_scores: Optional[Dict[str, int]]
    
    # Output (goes to blockchain)
    overall_rating: Optional[int]  # 0-100 for ERC-8004
    
    # Control
    error: Optional[str]
    messages: List[Any]  # ReAct agent message history


class ServiceEvaluator:
    """
    ReAct agent for evaluating completed service results.
    
    Analyzes service output against requirements and generates:
    - Integer rating (0-100) per ERC-8004 spec
    """
    
    def __init__(self, config: Config):
        """Initialize service evaluator."""
        self.config = config
        
        # Build tools and graph
        self._tools = self._build_tools()
        self.graph = self._build_graph()
        
        logger.info("ServiceEvaluator initialized with ReAct agent and LangGraph")
    
    def _build_tools(self) -> List:
        """Build tools for the ReAct agent."""
        
        @tool
        def extract_prompt_response_pairs(
            service_requirements: str,  # JSON string
            service_result: str  # JSON string
        ) -> Dict[str, Any]:
            """Extract structured prompt-response pairs for evaluation.
            
            Args:
                service_requirements: JSON service requirements with prompts and criteria
                service_result: JSON service result with responses array
                
            Returns:
                Structured data ready for LLM evaluation
            """
            try:
                requirements = json.loads(service_requirements)
                result = json.loads(service_result)
                
                # Extract from literature_review result structure
                responses_data = result.get("responses", [])
                
                # Each response has {"prompt": "...", "response": "..."}
                pairs = []
                for item in responses_data:
                    pairs.append({
                        "prompt": item.get("prompt", ""),
                        "response": item.get("response", "")
                    })
                
                return {
                    "pair_count": len(pairs),
                    "pairs": pairs,
                    "quality_criteria": requirements.get("quality_criteria", {}),
                    "complexity": requirements.get("complexity", "medium"),
                    "service_type": requirements.get("service_type", "")
                }
            except Exception as e:
                return {"error": str(e)}
        
        @tool
        def finalize_evaluation(
            dimension_scores: str  # JSON like '{"completeness": 90, "depth": 75, "clarity": 85}'
        ) -> Dict[str, Any]:
            """Calculate final rating from quality dimension scores.
            
            Args:
                dimension_scores: JSON object mapping dimension names to scores (0-100)
                
            Returns:
                Overall rating and quality scores breakdown
            """
            try:
                scores = json.loads(dimension_scores)
                
                if not scores:
                    return {
                        "overall_rating": 0,
                        "quality_scores": {}
                    }
                
                # Calculate average (equal weighting for simplicity)
                overall = int(sum(scores.values()) / len(scores))
                
                # Ensure 0-100 range
                overall = max(0, min(100, overall))
                
                return {
                    "overall_rating": overall,
                    "quality_scores": scores
                }
            except Exception as e:
                return {"error": str(e), "overall_rating": 0, "quality_scores": {}}
        
        return [extract_prompt_response_pairs, finalize_evaluation]
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for evaluation."""
        workflow = StateGraph(EvaluatorState)
        
        workflow.add_node("react_evaluation", self._react_evaluation_node)
        workflow.add_node("validate_rating", self._validate_rating_node)
        
        workflow.set_entry_point("react_evaluation")
        workflow.add_edge("react_evaluation", "validate_rating")
        workflow.add_edge("validate_rating", END)
        
        return workflow.compile()
    
    async def _react_evaluation_node(self, state: EvaluatorState) -> EvaluatorState:
        """Use ReAct agent to evaluate service quality."""
        logger.info("🤔 ReAct evaluation in progress...")
        
        try:
            requirements = state["service_requirements"]
            result = state["result"]
            
            # Serialize inputs for tools
            requirements_json = json.dumps(requirements)
            result_json = json.dumps(result)
            
            # Build system prompt
            system_prompt = f"""You are an expert service quality evaluator for literature review services.

**Your Task:**
1. Use extract_prompt_response_pairs() to get structured data
2. Analyze each prompt-response pair against the quality criteria
3. Assign a score (0-100) for each quality dimension
4. Use finalize_evaluation() to calculate the overall rating

**Evaluation Process:**
- Read the quality_criteria from the extracted data
- For each criterion (e.g., completeness, depth, clarity):
  * Review all prompt-response pairs
  * Evaluate how well responses meet that criterion
  * Assign a score 0-100
- Be objective and fair - evaluate based on actual content quality

**Service Requirements:**
```json
{requirements_json}
```

**Service Result:**
```json
{result_json}
```

Start by extracting the prompt-response pairs, then evaluate systematically."""

            # Create ReAct agent with rate limiting
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=0.1,
                check_every_n_seconds=0.1,
                max_bucket_size=1
            )
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.config.google_api_key,
                temperature=0.3,
                rate_limiter=rate_limiter
            )
            
            react_agent = create_agent(llm, self._tools)
            
            # Run agent
            agent_result = await react_agent.ainvoke({
                "messages": [HumanMessage(content=system_prompt)]
            })
            
            # Log reasoning trace
            logger.info("\n" + "=" * 80)
            logger.info("🤖 EVALUATOR REASONING TRACE")
            logger.info("=" * 80)
            for i, msg in enumerate(agent_result["messages"], 1):
                msg_class = msg.__class__.__name__
                
                if hasattr(msg, 'content') and msg.content:
                    content = str(msg.content)
                    if len(content) > 500:
                        content = content[:500] + "..."
                    logger.info(f"\n[{i}] {msg_class}:\n{content}")
                
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    logger.info(f"\n[{i}] {msg_class} - Tool Calls:")
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        tool_args = tc.get('args', {})
                        logger.info(f"  🔧 {tool_name}({list(tool_args.keys())})")
                
                if msg_class == "ToolMessage":
                    tool_output = str(msg.content)[:300]
                    logger.info(f"\n[{i}] ToolMessage:\n  ✅ {tool_output}")
            
            logger.info("=" * 80 + "\n")
            
            # Extract results from tool calls
            overall_rating = None
            quality_scores = None
            
            for msg in reversed(agent_result["messages"]):
                if msg.__class__.__name__ == "ToolMessage" and "overall_rating" in str(msg.content):
                    try:
                        # Parse the finalize_evaluation output
                        result_data = eval(msg.content) if isinstance(msg.content, str) else msg.content
                        if isinstance(result_data, dict):
                            overall_rating = result_data.get("overall_rating")
                            quality_scores = result_data.get("quality_scores")
                            break
                    except:
                        pass
            
            # Fallback if agent didn't use tools properly
            if overall_rating is None:
                logger.warning("Agent did not produce rating via tools, using fallback")
                overall_rating = 75
                quality_scores = {"fallback": 75}
            
            state["overall_rating"] = overall_rating
            state["quality_scores"] = quality_scores
            state["messages"] = agent_result["messages"]
            
            logger.info(f"✅ ReAct evaluation completed: rating={overall_rating}, scores={quality_scores}")
            
        except Exception as e:
            logger.error(f"Error in ReAct evaluation: {e}", exc_info=True)
            state["error"] = str(e)
            state["overall_rating"] = 0
            state["quality_scores"] = {}
        
        return state
    
    async def _validate_rating_node(self, state: EvaluatorState) -> EvaluatorState:
        """Validate rating is within ERC-8004 spec (0-100)."""
        rating = state.get("overall_rating", 0)
        state["overall_rating"] = max(0, min(100, rating))
        
        logger.info(f"✅ Final validated rating: {state['overall_rating']}/100")
        return state
    
    async def evaluate(
        self,
        service_requirements: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate completed service result using ReAct agent.
        
        Args:
            service_requirements: Original service requirements from IPFS
            result: Completed service result from provider
            
        Returns:
            Evaluation dict with rating (0-100) and quality_scores
        """
        logger.info("Evaluating service result with ReAct agent...")
        
        initial_state: EvaluatorState = {
            "service_requirements": service_requirements,
            "result": result,
            "quality_scores": None,
            "overall_rating": None,
            "error": None,
            "messages": []
        }
        
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "rating": final_state.get("overall_rating", 0),
            "quality_scores": final_state.get("quality_scores", {}),
            "error": final_state.get("error")
        }
