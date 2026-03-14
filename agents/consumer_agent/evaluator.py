"""
Service Evaluator - ReAct agent for evaluating completed service quality.

Uses ReAct pattern with LLM to assess service quality and generate ratings.
"""

import json
import logging
import ast
import re
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

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
            service_requirements: Any,
            service_result: Any
        ) -> Dict[str, Any]:
            """Extract structured prompt-response pairs for evaluation.
            
            Args:
                service_requirements: JSON service requirements with prompts and criteria
                service_result: JSON service result with responses array
                
            Returns:
                Structured data ready for LLM evaluation
            """
            def safe_json_loads(value):
                """Parse tool input robustly (dict/list, JSON string, or Python-literal string)."""
                if isinstance(value, (dict, list)):
                    return value
                if value is None:
                    return {}
                if isinstance(value, str):
                    text = value.strip()

                    # Strip markdown code fences if present
                    if text.startswith("```"):
                        lines = text.splitlines()
                        if len(lines) >= 2:
                            text = "\n".join(lines[1:-1]).strip()

                    # Prefer strict JSON
                    try:
                        return json.loads(text)
                    except Exception:
                        pass

                    # Fallback: Python literal dict/list from model output
                    try:
                        return ast.literal_eval(text)
                    except Exception:
                        pass

                    # Last attempt: recover likely JSON object substring
                    if "{" in text and "}" in text:
                        start = text.find("{")
                        end = text.rfind("}")
                        if start < end:
                            snippet = text[start:end + 1]
                            try:
                                return json.loads(snippet)
                            except Exception:
                                try:
                                    return ast.literal_eval(snippet)
                                except Exception:
                                    pass

                    raise ValueError("Unable to parse tool argument as JSON/object")
                raise TypeError(f"Unsupported input type: {type(value).__name__}")
            try:
                requirements = safe_json_loads(service_requirements)
                result = safe_json_loads(service_result)
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
            except json.JSONDecodeError as e:
                return {"error": f"JSON parsing error: {str(e)}"}
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

    def _parse_rating_from_messages(self, messages: List[Any]) -> tuple[Optional[int], Optional[Dict[str, int]]]:
        """Extract overall rating and quality scores from agent messages."""
        for msg in reversed(messages):
            try:
                # Preferred: tool output from finalize_evaluation
                if msg.__class__.__name__ == "ToolMessage" and "overall_rating" in str(msg.content):
                    if isinstance(msg.content, str):
                        try:
                            result_data = json.loads(msg.content)
                        except Exception:
                            result_data = ast.literal_eval(msg.content)
                    else:
                        result_data = msg.content

                    if isinstance(result_data, dict):
                        return result_data.get("overall_rating"), result_data.get("quality_scores")

                # Secondary: model returned final JSON directly (no tool message)
                if msg.__class__.__name__ == "AIMessage" and "overall_rating" in str(getattr(msg, "content", "")):
                    content = getattr(msg, "content", "")
                    if isinstance(content, str) and content.strip():
                        try:
                            result_data = json.loads(content)
                        except Exception:
                            try:
                                result_data = ast.literal_eval(content)
                            except Exception:
                                result_data = None
                        if isinstance(result_data, dict):
                            return result_data.get("overall_rating"), result_data.get("quality_scores")

                # Tertiary: model returned plain text like "Overall Rating: 92"
                if msg.__class__.__name__ == "AIMessage":
                    content = getattr(msg, "content", "")
                    if isinstance(content, str) and content.strip():
                        parsed = self._parse_rating_from_text(content)
                        if parsed[0] is not None:
                            return parsed
            except Exception:
                continue

        return None, None

    def _parse_rating_from_text(self, content: str) -> tuple[Optional[int], Optional[Dict[str, int]]]:
        """Parse rating from non-JSON model output when tool/JSON output is absent."""
        try:
            # Common patterns: "Overall Rating: 92" or "rating=75/100"
            m = re.search(r"overall\s*rating\s*[:=]\s*(\d{1,3})", content, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"rating\s*[:=]\s*(\d{1,3})\s*(?:/\s*100)?", content, flags=re.IGNORECASE)
            if not m:
                return None, None

            overall = max(0, min(100, int(m.group(1))))

            # Try to recover per-dimension scores when present in text.
            quality_scores: Dict[str, int] = {}
            for dim in ["completeness", "depth", "citations", "clarity"]:
                dm = re.search(rf"{dim}\s*[:=]\s*(\d{{1,3}})", content, flags=re.IGNORECASE)
                if dm:
                    quality_scores[dim] = max(0, min(100, int(dm.group(1))))

            return overall, quality_scores
        except Exception:
            return None, None

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

            # Create ReAct agent without rate limiting (use API defaults)
            llm = ChatOpenAI(
                model="openai/gpt-oss-20b",
                api_key=self.config.openrouter_api_key,
                base_url=self.config.openrouter_base_url,
                temperature=0.3
            )
            
            react_agent = create_agent(llm, self._tools)
            
            recursion_limit = int(getattr(self.config, "evaluator_recursion_limit", 25))

            # Run agent
            agent_result = await react_agent.ainvoke({
                "messages": [HumanMessage(content=system_prompt)]
            }, config={"recursion_limit": recursion_limit})
            
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
            overall_rating, quality_scores = self._parse_rating_from_messages(agent_result["messages"])

            # Retry once with stricter instruction if no usable rating was produced
            if overall_rating is None:
                logger.warning("Agent did not produce rating on first pass; retrying with stricter tool-use instruction")
                retry_msg = HumanMessage(content=(
                    "You must call tools to finish: "
                    "1) extract_prompt_response_pairs(service_requirements, service_result), "
                    "2) compute dimension scores (0-100), "
                    "3) call finalize_evaluation(dimension_scores as JSON string). "
                    "Return only final JSON from finalize_evaluation."
                ))
                retry_result = await react_agent.ainvoke({
                    "messages": [HumanMessage(content=system_prompt), retry_msg]
                }, config={"recursion_limit": recursion_limit})
                agent_result = retry_result
                overall_rating, quality_scores = self._parse_rating_from_messages(agent_result["messages"])

            # Final LLM-only strict JSON pass
            if overall_rating is None:
                logger.warning("Agent still did not produce a valid rating; retrying with strict JSON-only LLM pass")
                strict_json_prompt = f"""You are an evaluator. Score this service result against requirements.

                Return ONLY valid JSON with this exact schema:
                {{
                \"overall_rating\": <int 0-100>,
                \"quality_scores\": {{
                    \"completeness\": <int 0-100>,
                    \"depth\": <int 0-100>,
                    \"citations\": <int 0-100>,
                    \"clarity\": <int 0-100>
                }}
                }}

                Requirements JSON:
                {requirements_json}

                Result JSON:
                {result_json}
                """
                strict_resp = await llm.ainvoke([HumanMessage(content=strict_json_prompt)])
                strict_text = strict_resp.content if isinstance(strict_resp.content, str) else str(strict_resp.content)

                try:
                    parsed = json.loads(strict_text)
                except Exception:
                    try:
                        parsed = ast.literal_eval(strict_text)
                    except Exception:
                        parsed = None

                if isinstance(parsed, dict):
                    overall_rating = parsed.get("overall_rating")
                    quality_scores = parsed.get("quality_scores")

                if overall_rating is None:
                    txt_rating, txt_scores = self._parse_rating_from_text(strict_text)
                    overall_rating = txt_rating
                    quality_scores = txt_scores

            # If all LLM attempts fail, set explicit error and neutral empty scores.
            if overall_rating is None:
                logger.error("Evaluator failed to produce valid rating after all LLM passes")
                state["error"] = "LLM evaluator did not return a valid rating"
                overall_rating = 0
                quality_scores = {}
            
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
        if isinstance(result, dict) and result.get("service_failed"):
            logger.warning("Service result is marked as failed; assigning rating=0")
            return {
                "rating": 0,
                "quality_scores": {"execution_failure": 0},
                "error": result.get("error")
            }

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
