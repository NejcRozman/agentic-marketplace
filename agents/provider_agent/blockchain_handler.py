"""
BlockchainHandler - Agentic handler for blockchain operations.

This agent uses LangGraph to reason about blockchain operations:
- Monitoring auctions and deciding which to participate in
- Calculating optimal bids based on risk assessment
- Managing service completion and payment release

Built with LangGraph for agentic reasoning and decision-making.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ..infrastructure.blockchain_client import BlockchainClient
from ..infrastructure.auction_data import AuctionInfo
from ..infrastructure.contract_abis import (
    get_reverse_auction_abi,
    get_identity_registry_abi,
    get_reputation_registry_abi
)
from ..infrastructure.feedback_auth import generate_feedback_auth, verify_feedback_auth_format
from ..config import config

logger = logging.getLogger(__name__)


class BlockchainState(TypedDict):
    """State for blockchain agent workflow."""
    agent_id: int
    action: str  # "complete_service" or "monitor"
    
    # Service completion fields
    auction_id: Optional[int]
    client_address: Optional[str]
    feedback_auth: Optional[bytes]
    
    # Block tracking fields
    last_processed_block: Optional[int]  # Track last processed block to avoid reprocessing
    current_block: Optional[int]  # Current block number when fetching events
    
    # Event monitoring fields
    unprocessed_events: List[Dict[str, Any]]  # Events waiting to be processed
    processed_events: List[Dict[str, Any]]  # Events that have been processed
    processed_event: Optional[Dict[str, Any]]  # Current event being processed
    event_type: Optional[str]  # "AuctionCreated" or "AuctionEnded"
    
    # Auction analysis fields
    ipfs_data: Optional[Dict[str, Any]]
    auction_state: Optional[Dict[str, Any]]
    reputations: Optional[Dict[str, Any]]
    estimated_cost: Optional[int]
    
    # Bidding decision fields
    should_bid: bool
    bid_amount: Optional[int]
    bid_reasoning: Optional[str]
    won_auction: Optional[bool]
    
    # Transaction results
    tx_hash: Optional[str]
    error: Optional[str]
    messages: List[Any]


class BlockchainHandler:
    """
    Agentic blockchain handler using LangGraph.
    
    Handles blockchain operations with reasoning capabilities.
    """
    
    def __init__(self, agent_id: int, blockchain_client: Optional[BlockchainClient] = None):
        """Initialize the blockchain handler agent."""
        self.agent_id = agent_id
        self.client = blockchain_client or BlockchainClient()
        self.config = config
        
        self.contracts_loaded = False
        self.reverse_auction_contract = None
        self.identity_registry_contract = None
        self.reputation_registry_contract = None
        
        # Block tracking - persists across graph invocations
        self._last_processed_block: int = 0
        
        self.graph = self._build_graph()
        logger.info(f"BlockchainHandler initialized for agent {agent_id}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for blockchain operations."""
        workflow = StateGraph(BlockchainState)
        
        # Router node
        workflow.add_node("router", self._router_node)
        
        # Complete service path
        workflow.add_node("generate_feedback_auth", self._generate_feedback_auth_node)
        workflow.add_node("call_complete_service", self._call_complete_service_node)
        
        # Monitor path - event processing
        workflow.add_node("fetch_events", self._fetch_events_node)
        workflow.add_node("process_events", self._process_events_node)
        
        # Auction workflow (for AuctionCreated events)
        workflow.add_node("fetch_ipfs_data", self._fetch_ipfs_data_node)
        workflow.add_node("get_auction_state", self._get_auction_state_node)
        workflow.add_node("get_reputations", self._get_reputations_node)
        workflow.add_node("estimate_cost", self._estimate_cost_node)
        workflow.add_node("reason_bid", self._reason_bid_node)
        workflow.add_node("submit_bid", self._submit_bid_node)
        
        # Entry point
        workflow.set_entry_point("router")
        
        # Router conditional edges
        workflow.add_conditional_edges(
            "router",
            self._route_action,
            {
                "complete_service": "generate_feedback_auth",
                "monitor": "fetch_events"
            }
        )
        
        # Complete service path
        workflow.add_edge("generate_feedback_auth", "call_complete_service")
        workflow.add_edge("call_complete_service", END)
        
        # Monitor path
        workflow.add_edge("fetch_events", "process_events")
        workflow.add_conditional_edges(
            "process_events",
            self._route_event_type,
            {
                "auction_created": "fetch_ipfs_data",
                "auction_ended": "check_more_events",
                "no_events": END
            }
        )
        
        # Auction workflow path
        workflow.add_edge("fetch_ipfs_data", "get_auction_state")
        workflow.add_edge("get_auction_state", "get_reputations")
        workflow.add_edge("get_reputations", "estimate_cost")
        workflow.add_edge("estimate_cost", "reason_bid")
        workflow.add_conditional_edges(
            "reason_bid",
            self._route_bid_decision,
            {
                "submit": "submit_bid",
                "skip": "check_more_events"
            }
        )
        workflow.add_edge("submit_bid", "check_more_events")
        
        # Check for more events node (loops back or ends)
        workflow.add_node("check_more_events", self._check_more_events_node)
        workflow.add_conditional_edges(
            "check_more_events",
            self._route_more_events,
            {
                "more_events": "process_events",
                "done": END
            }
        )
        
        return workflow.compile()
    
    async def initialize(self) -> bool:
        """Initialize the blockchain handler by loading contracts."""
        try:
            await self.client._initialize()
            
            if not self.config.reverse_auction_address:
                logger.warning("ReverseAuction address not configured")
                return False
            
            self.reverse_auction_contract = await self.client.load_contract(
                name="ReverseAuction",
                address=self.config.reverse_auction_address,
                abi=get_reverse_auction_abi()
            )
            
            if self.config.identity_registry_address:
                self.identity_registry_contract = await self.client.load_contract(
                    name="IdentityRegistry",
                    address=self.config.identity_registry_address,
                    abi=get_identity_registry_abi()
                )
            
            if self.config.reputation_registry_address:
                self.reputation_registry_contract = await self.client.load_contract(
                    name="ReputationRegistry",
                    address=self.config.reputation_registry_address,
                    abi=get_reputation_registry_abi()
                )
            
            self.contracts_loaded = True
            logger.info("✅ BlockchainHandler initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BlockchainHandler: {e}", exc_info=True)
            return False
    
    # ==================== Router Nodes ====================
    
    def _router_node(self, state: BlockchainState) -> BlockchainState:
        """Router node - entry point that examines action field."""
        logger.info(f"🔀 Router: action={state['action']}")
        state["messages"].append(SystemMessage(content=f"Routing to {state['action']} path"))
        return state
    
    def _route_action(self, state: BlockchainState) -> str:
        """Routing function for action field."""
        action = state.get("action", "monitor")
        if action == "complete_service":
            return "complete_service"
        return "monitor"
    
    def _route_event_type(self, state: BlockchainState) -> str:
        """Routing function for event type."""
        event_type = state.get("event_type")
        if event_type == "AuctionCreated":
            return "auction_created"
        elif event_type == "AuctionEnded":
            return "auction_ended"
        return "no_events"
    
    def _route_bid_decision(self, state: BlockchainState) -> str:
        """Routing function for bid decision."""
        if state.get("should_bid", False):
            return "submit"
        return "skip"
    
    def _route_more_events(self, state: BlockchainState) -> str:
        """Routing function to check if more events need processing."""
        unprocessed = state.get("unprocessed_events", [])
        if unprocessed:
            return "more_events"
        return "done"
    
    # ==================== Complete Service Path ====================
    
    def _generate_feedback_auth_node(self, state: BlockchainState) -> BlockchainState:
        """Generate ERC-8004 feedbackAuth for service completion."""
        logger.info("🔐 Generating feedbackAuth...")
        
        try:
            client_address = state.get("client_address")
            if not client_address:
                raise ValueError("client_address is required for feedback_auth generation")
            
            # Generate feedbackAuth with 1 hour expiry
            expiry = int(datetime.now().timestamp()) + 3600
            feedback_auth = generate_feedback_auth(
                agent_id=self.agent_id,
                client_address=client_address,
                index_limit=1000,
                expiry=expiry,
                chain_id=self.config.chain_id,
                identity_registry_address=self.config.identity_registry_address,
                signer_address=self.client.account.address,
                private_key=self.config.private_key
            )
            
            if not verify_feedback_auth_format(feedback_auth):
                raise ValueError("Generated feedbackAuth has invalid format")
            
            state["feedback_auth"] = feedback_auth
            state["messages"].append(SystemMessage(content="FeedbackAuth generated successfully"))
            logger.info("✅ FeedbackAuth generated")
            
        except Exception as e:
            logger.error(f"Error generating feedbackAuth: {e}")
            state["error"] = str(e)
        
        return state
    
    def _call_complete_service_node(self, state: BlockchainState) -> BlockchainState:
        """Call completeService on ReverseAuction contract."""
        logger.info("📝 Calling completeService on blockchain...")
        
        try:
            auction_id = state.get("auction_id")
            feedback_auth = state.get("feedback_auth")
            
            if auction_id is None:
                raise ValueError("auction_id is required")
            if not feedback_auth:
                raise ValueError("feedback_auth is required")
            
            # Estimate gas
            estimated_gas = asyncio.run(self.client.estimate_gas(
                "ReverseAuction",
                "completeService",
                auction_id,
                feedback_auth
            ))
            
            # Send transaction
            tx_hash = asyncio.run(self.client.send_transaction(
                "ReverseAuction",
                "completeService",
                auction_id,
                feedback_auth,
                gas_limit=estimated_gas + 50000
            ))
            
            # Wait for confirmation
            receipt = asyncio.run(self.client.wait_for_transaction(tx_hash))
            
            if receipt['status'] == 1:
                state["tx_hash"] = tx_hash
                state["messages"].append(SystemMessage(content=f"Service completed: {tx_hash}"))
                logger.info(f"✅ Service completed in block {receipt['blockNumber']}")
            else:
                raise Exception("Transaction failed")
            
        except Exception as e:
            logger.error(f"Error calling completeService: {e}")
            state["error"] = str(e)
        
        return state
    
    # ==================== Monitor Path - Event Processing ====================
    
    def _fetch_events_node(self, state: BlockchainState) -> BlockchainState:
        """Fetch new events from ReverseAuction contract."""
        logger.info("📡 Fetching blockchain events...")
        
        try:
            # Get current block number
            current_block = asyncio.run(self.client.get_block_number())
            state["current_block"] = current_block
            
            # Determine from_block based on last processed
            last_processed = state.get("last_processed_block")
            if last_processed is not None and last_processed > 0:
                # Start from the block after the last processed one
                from_block = last_processed + 1
            else:
                # First run: look back 10 blocks
                from_block = max(0, current_block - 10)
            
            logger.info(f"Fetching events from block {from_block} to {current_block}")
            
            # Fetch events from the block range
            events = asyncio.run(self.client.get_contract_events(
                contract_name="ReverseAuction",
                event_name="AuctionCreated",
                from_block=from_block,
                to_block=current_block
            ))
            
            # Update last processed block to current
            state["last_processed_block"] = current_block
            state["unprocessed_events"] = list(events)  # Queue for processing
            state["processed_events"] = []  # Reset processed list
            state["messages"].append(SystemMessage(content=f"Fetched {len(events)} events from blocks {from_block}-{current_block}"))
            logger.info(f"Fetched {len(events)} AuctionCreated events from blocks {from_block}-{current_block}")
            
        except Exception as e:
            logger.error(f"Error fetching events: {e}")
            state["error"] = str(e)
            state["unprocessed_events"] = []
        
        return state
    
    def _process_events_node(self, state: BlockchainState) -> BlockchainState:
        """Process next event from unprocessed_events queue."""
        logger.info("🔍 Processing events...")
        
        try:
            unprocessed = state.get("unprocessed_events", [])
            
            if not unprocessed:
                state["event_type"] = None
                state["messages"].append(SystemMessage(content="No events to process"))
                logger.info("No events to process")
                return state
            
            # Pop first event from unprocessed queue
            event = unprocessed.pop(0)
            state["unprocessed_events"] = unprocessed
            event_name = event.get("event", "")
            
            logger.info(f"Processing event: {event_name} ({len(unprocessed)} remaining)")
            
            if event_name == "AuctionCreated":
                state["event_type"] = "AuctionCreated"
                state["processed_event"] = event
                state["auction_id"] = event.get("args", {}).get("auctionId")
                state["messages"].append(SystemMessage(content=f"Processing AuctionCreated event for auction {state['auction_id']}"))
                logger.info(f"Processing AuctionCreated for auction {state['auction_id']}")
                
            elif event_name == "AuctionEnded":
                state["event_type"] = "AuctionEnded"
                state["processed_event"] = event
                auction_id = event.get("args", {}).get("auctionId")
                state["auction_id"] = auction_id
                # TODO: Check if we won this auction
                state["won_auction"] = False  # Placeholder
                state["messages"].append(SystemMessage(content=f"AuctionEnded for auction {auction_id}"))
                logger.info(f"AuctionEnded for auction {auction_id}")
            
            else:
                # Unknown event type, skip it
                state["event_type"] = None
                logger.warning(f"Unknown event type: {event_name}")
            
        except Exception as e:
            logger.error(f"Error processing events: {e}")
            state["error"] = str(e)
            state["event_type"] = None
        
        return state
    
    def _check_more_events_node(self, state: BlockchainState) -> BlockchainState:
        """Check if there are more events to process and prepare for next iteration."""
        # Add current event to processed list
        processed_event = state.get("processed_event")
        if processed_event:
            processed_events = state.get("processed_events", [])
            processed_events.append(processed_event)
            state["processed_events"] = processed_events
        
        unprocessed = state.get("unprocessed_events", [])
        logger.info(f"🔄 Check more events: {len(unprocessed)} remaining")
        
        if unprocessed:
            # Reset per-event state for next iteration
            state["processed_event"] = None
            state["event_type"] = None
            state["auction_id"] = None
            state["ipfs_data"] = None
            state["auction_state"] = None
            state["reputations"] = None
            state["estimated_cost"] = None
            state["should_bid"] = False
            state["bid_amount"] = None
            state["bid_reasoning"] = None
            state["won_auction"] = None
            state["tx_hash"] = None
            # Keep error if any, but could reset: state["error"] = None
            state["messages"].append(SystemMessage(content=f"Moving to next event ({len(unprocessed)} remaining)"))
        else:
            state["messages"].append(SystemMessage(content=f"All {len(state.get('processed_events', []))} events processed"))
            logger.info(f"✅ All events processed")
        
        return state
    
    # ==================== Auction Workflow Path ======================================
    
    def _fetch_ipfs_data_node(self, state: BlockchainState) -> BlockchainState:
        """Fetch service requirements from IPFS using CID."""
        logger.info("🌐 Fetching IPFS data...")
        
        try:
            auction_state = state.get("auction_state", {})
            cid = auction_state.get("serviceDescriptionCid", "")
            
            if not cid:
                raise ValueError("No IPFS CID in auction state")
            
            # Fetch from IPFS (using public gateway for now)
            # TODO: Configure IPFS gateway (Pinata, Infura, or local node)
            import aiohttp
            
            ipfs_gateway_url = f"https://ipfs.io/ipfs/{cid}"
            
            async def fetch_ipfs():
                async with aiohttp.ClientSession() as session:
                    async with session.get(ipfs_gateway_url, timeout=10) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            raise Exception(f"IPFS fetch failed with status {response.status}")
            
            ipfs_data = asyncio.run(fetch_ipfs())
            
            state["ipfs_data"] = ipfs_data
            state["messages"].append(SystemMessage(content=f"Fetched IPFS data from {cid}"))
            logger.info(f"✅ Fetched IPFS data: {ipfs_data.get('title', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error fetching IPFS data: {e}")
            state["error"] = str(e)
            state["ipfs_data"] = {}
        
        return state
    
    def _get_auction_state_node(self, state: BlockchainState) -> BlockchainState:
        """Query ReverseAuction contract for current auction details."""
        logger.info("📋 Getting auction state...")
        
        try:
            auction_id = state.get("auction_id")
            if auction_id is None:
                raise ValueError("auction_id is required")
            
            # Call getAuctionDetails
            auction_data = asyncio.run(self.client.call_contract_method(
                "ReverseAuction",
                "getAuctionDetails",
                auction_id
            ))
            
            auction_state = {
                "buyer": auction_data[0],
                "budget": auction_data[1],
                "deadline": auction_data[2],
                "serviceDescriptionCid": auction_data[3],
                "isActive": auction_data[4],
                "selectedProvider": auction_data[5],
                "finalPrice": auction_data[6],
                "serviceCompleted": auction_data[7]
            }
            
            state["auction_state"] = auction_state
            state["messages"].append(SystemMessage(content=f"Auction {auction_id}: budget={auction_state['budget']}, deadline={auction_state['deadline']}"))
            logger.info(f"✅ Auction state: budget={auction_state['budget']}")
            
        except Exception as e:
            logger.error(f"Error getting auction state: {e}")
            state["error"] = str(e)
            state["auction_state"] = {}
        
        return state
    
    def _get_reputations_node(self, state: BlockchainState) -> BlockchainState:
        """Query ERC-8004 ReputationRegistry for self and competitor reputations."""
        logger.info("⭐ Getting reputations...")
        
        try:
            auction_id = state.get("auction_id")
            if auction_id is None:
                raise ValueError("auction_id is required")
            
            # Get list of bidders for this auction
            # TODO: Query contract for existing bids
            bidders = []  # Placeholder - need to query getBids or similar
            
            # Get our reputation
            self_reputation = asyncio.run(self.client.call_contract_method(
                "ReputationRegistry",
                "getReputation",
                self.agent_id
            )) if self.reputation_registry_contract else {"rating": 0, "totalFeedback": 0}
            
            # Get competitor reputations
            competitor_reputations = []
            for bidder_id in bidders:
                if bidder_id != self.agent_id:
                    rep = asyncio.run(self.client.call_contract_method(
                        "ReputationRegistry",
                        "getReputation",
                        bidder_id
                    )) if self.reputation_registry_contract else {"rating": 0, "totalFeedback": 0}
                    competitor_reputations.append({"agent_id": bidder_id, "reputation": rep})
            
            state["reputations"] = {
                "self": self_reputation,
                "competitors": competitor_reputations
            }
            state["messages"].append(SystemMessage(content=f"Self reputation: {self_reputation}, Competitors: {len(competitor_reputations)}"))
            logger.info(f"✅ Reputations fetched: self={self_reputation}, competitors={len(competitor_reputations)}")
            
        except Exception as e:
            logger.error(f"Error getting reputations: {e}")
            state["error"] = str(e)
            state["reputations"] = {"self": {}, "competitors": []}
        
        return state
    
    def _estimate_cost_node(self, state: BlockchainState) -> BlockchainState:
        """Estimate cost of delivering the service."""
        logger.info("💰 Estimating service cost...")
        
        try:
            ipfs_data = state.get("ipfs_data", {})
            auction_state = state.get("auction_state", {})
            
            # Simple cost estimation based on service requirements
            # TODO: Implement sophisticated cost model
            budget = auction_state.get("budget", 0)
            
            # Heuristic: estimate 70% of budget as base cost
            base_cost = int(budget * 0.7)
            
            # Adjust based on complexity (if available in IPFS data)
            complexity = ipfs_data.get("complexity", "medium")
            if complexity == "high":
                base_cost = int(base_cost * 1.2)
            elif complexity == "low":
                base_cost = int(base_cost * 0.8)
            
            state["estimated_cost"] = base_cost
            state["messages"].append(SystemMessage(content=f"Estimated cost: {base_cost} wei"))
            logger.info(f"✅ Estimated cost: {base_cost} wei")
            
        except Exception as e:
            logger.error(f"Error estimating cost: {e}")
            state["error"] = str(e)
            state["estimated_cost"] = 0
        
        return state
    
    def _reason_bid_node(self, state: BlockchainState) -> BlockchainState:
        """Use LLM to reason about bidding decision."""
        logger.info("🤔 Reasoning about bid decision...")
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            
            # Gather all context
            auction_state = state.get("auction_state", {})
            ipfs_data = state.get("ipfs_data", {})
            reputations = state.get("reputations", {})
            estimated_cost = state.get("estimated_cost", 0)
            
            budget = auction_state.get("budget", 0)
            deadline = auction_state.get("deadline", 0)
            
            # Build system prompt with bidding strategy
            system_prompt = """You are a bidding agent for a decentralized service marketplace.
Your goal is to win auctions by bidding competitively while maintaining profitability.

Bidding Strategy:
1. Only bid if estimated cost < budget (profitable)
2. Consider your reputation vs competitors
3. Bid lower if you have better reputation
4. Bid slightly below budget to maximize winning chance
5. Account for deadline urgency
6. Consider service complexity

Output format (JSON):
{
    "should_bid": true/false,
    "bid_amount": <amount in wei>,
    "reasoning": "<brief explanation>"
}"""
            
            # Build context
            context = f"""
Auction Analysis:
- Budget: {budget} wei
- Deadline: {deadline} (unix timestamp)
- Estimated Cost: {estimated_cost} wei
- Profit Margin: {budget - estimated_cost} wei ({((budget - estimated_cost) / budget * 100) if budget > 0 else 0:.1f}%)

Service Requirements:
{ipfs_data}

Reputation Context:
- Self: {reputations.get('self', {})}
- Competitors: {len(reputations.get('competitors', []))} bidders

Should we bid? If yes, what amount?
"""
            
            # Call LLM
            llm = ChatGoogleGenerativeAI(
                model=self.config.gemini_model,
                google_api_key=self.config.gemini_api_key,
                temperature=0.3
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ]
            
            response = llm.invoke(messages)
            
            # Parse response
            import json
            response_text = response.content
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            decision = json.loads(response_text)
            
            state["should_bid"] = decision.get("should_bid", False)
            state["bid_amount"] = decision.get("bid_amount", 0)
            state["bid_reasoning"] = decision.get("reasoning", "")
            
            state["messages"].append(SystemMessage(content=f"Bid decision: {decision}"))
            logger.info(f"✅ Bid decision: should_bid={state['should_bid']}, amount={state['bid_amount']}")
            logger.info(f"   Reasoning: {state['bid_reasoning']}")
            
        except Exception as e:
            logger.error(f"Error in bid reasoning: {e}")
            state["error"] = str(e)
            state["should_bid"] = False
            state["bid_reasoning"] = f"Error: {str(e)}"
        
        return state
    
    def _submit_bid_node(self, state: BlockchainState) -> BlockchainState:
        """Submit bid transaction to blockchain."""
        logger.info("📤 Submitting bid...")
        
        try:
            auction_id = state.get("auction_id")
            bid_amount = state.get("bid_amount", 0)
            
            if auction_id is None:
                raise ValueError("auction_id is required")
            if bid_amount <= 0:
                raise ValueError("bid_amount must be positive")
            
            # Estimate gas
            estimated_gas = asyncio.run(self.client.estimate_gas(
                "ReverseAuction",
                "placeBid",
                auction_id,
                value=0
            ))
            
            # Send transaction
            tx_hash = asyncio.run(self.client.send_transaction(
                "ReverseAuction",
                "placeBid",
                auction_id,
                value=0,
                gas_limit=estimated_gas + 50000
            ))
            
            # Wait for confirmation
            receipt = asyncio.run(self.client.wait_for_transaction(tx_hash))
            
            if receipt['status'] == 1:
                state["tx_hash"] = tx_hash
                state["messages"].append(SystemMessage(content=f"Bid submitted: {tx_hash}"))
                logger.info(f"✅ Bid submitted in block {receipt['blockNumber']}")
            else:
                raise Exception("Transaction failed")
            
        except Exception as e:
            logger.error(f"Error submitting bid: {e}")
            state["error"] = str(e)
        
        return state
    
    # ==================== Helper Methods ====================
    
    async def _fetch_auctions(self, from_block: str = "latest", to_block: str = "latest") -> List[AuctionInfo]:
        """Fetch auctions from blockchain."""
        if not self.contracts_loaded:
            await self.initialize()
        
        try:
            events = await self.client.get_contract_events(
                contract_name="ReverseAuction",
                event_name="AuctionCreated",
                from_block=from_block,
                to_block=to_block
            )
            
            auctions = []
            for event in events:
                args = event.get('args', {})
                auction_id = args.get('auctionId', 0)
                
                auction_data = await self.client.call_contract_method(
                    "ReverseAuction",
                    "getAuctionDetails",
                    auction_id
                )
                
                auction_info = AuctionInfo({
                    'auctionId': auction_id,
                    'buyer': auction_data[0],
                    'budget': auction_data[1],
                    'deadline': auction_data[2],
                    'serviceDescriptionCid': auction_data[3],
                    'isActive': auction_data[4],
                    'selectedProvider': auction_data[5],
                    'finalPrice': auction_data[6],
                    'serviceCompleted': auction_data[7]
                })
                
                auctions.append(auction_info)
            
            return auctions
            
        except Exception as e:
            logger.error(f"Error fetching auctions: {e}")
            return []
    
    async def _submit_bid(self, auction_id: int, bid_amount: int) -> Optional[str]:
        """Submit bid transaction."""
        try:
            logger.info(f"Submitting bid to auction {auction_id}: {bid_amount}")
            
            estimated_gas = await self.client.estimate_gas(
                "ReverseAuction",
                "placeBid",
                auction_id,
                value=0
            )
            
            tx_hash = await self.client.send_transaction(
                "ReverseAuction",
                "placeBid",
                auction_id,
                value=0,
                gas_limit=estimated_gas + 50000
            )
            
            receipt = await self.client.wait_for_transaction(tx_hash)
            
            if receipt['status'] == 1:
                logger.info(f"✅ Bid confirmed in block {receipt['blockNumber']}")
                return tx_hash
            else:
                logger.error("❌ Bid transaction failed")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting bid: {e}")
            return None
    
    async def monitor_auctions(self, from_block: str = "latest", to_block: str = "latest") -> Dict[str, Any]:
        """
        Monitor blockchain for new auction events.
        
        This invokes the graph with action="monitor" to process events.
        """
        initial_state: BlockchainState = {
            "agent_id": self.agent_id,
            "action": "monitor",
            "auction_id": None,
            "client_address": None,
            "feedback_auth": None,
            "last_processed_block": self._last_processed_block,  # Use instance variable for persistence
            "current_block": None,
            "unprocessed_events": [],
            "processed_events": [],
            "processed_event": None,
            "event_type": None,
            "ipfs_data": None,
            "auction_state": None,
            "reputations": None,
            "estimated_cost": None,
            "should_bid": False,
            "bid_amount": None,
            "bid_reasoning": None,
            "won_auction": None,
            "tx_hash": None,
            "error": None,
            "messages": []
        }
        
        result = self.graph.invoke(initial_state)
        
        # Persist the last processed block for next invocation
        if result.get("last_processed_block"):
            self._last_processed_block = result["last_processed_block"]
        
        return {
            "event_type": result.get("event_type"),
            "auction_id": result.get("auction_id"),
            "should_bid": result.get("should_bid"),
            "bid_amount": result.get("bid_amount"),
            "bid_reasoning": result.get("bid_reasoning"),
            "tx_hash": result.get("tx_hash"),
            "won_auction": result.get("won_auction"),
            "last_processed_block": result.get("last_processed_block"),
            "processed_events_count": len(result.get("processed_events", [])),
            "error": result.get("error")
        }
    
    async def complete_service(self, auction_id: int, client_address: str) -> Dict[str, Any]:
        """
        Complete service for an auction.
        
        This invokes the graph with action="complete_service".
        """
        initial_state: BlockchainState = {
            "agent_id": self.agent_id,
            "action": "complete_service",
            "auction_id": auction_id,
            "client_address": client_address,
            "feedback_auth": None,
            "last_processed_block": None,
            "current_block": None,
            "unprocessed_events": [],
            "processed_events": [],
            "processed_event": None,
            "event_type": None,
            "ipfs_data": None,
            "auction_state": None,
            "reputations": None,
            "estimated_cost": None,
            "should_bid": False,
            "bid_amount": None,
            "bid_reasoning": None,
            "won_auction": None,
            "tx_hash": None,
            "error": None,
            "messages": []
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "tx_hash": result.get("tx_hash"),
            "feedback_auth": result.get("feedback_auth"),
            "error": result.get("error")
        }
    
    def create_feedback_auth(self, client_address: str, index_limit: int = 1000, expiry: Optional[int] = None) -> bytes:
        """Generate ERC-8004 compliant feedbackAuth."""
        if expiry is None:
            expiry = int(datetime.now().timestamp()) + 3600
        
        if not self.client.account:
            raise ValueError("No account configured")
        
        feedback_auth = generate_feedback_auth(
            agent_id=self.agent_id,
            client_address=client_address,
            index_limit=index_limit,
            expiry=expiry,
            chain_id=self.config.chain_id,
            identity_registry_address=self.config.identity_registry_address,
            signer_address=self.client.account.address,
            private_key=self.config.private_key
        )
        
        if not verify_feedback_auth_format(feedback_auth):
            raise ValueError("Generated feedbackAuth has invalid format")
        
        logger.info(f"✅ Created feedbackAuth for agent {self.agent_id}")
        return feedback_auth
    
    async def get_agent_address(self) -> str:
        """Get agent's blockchain address."""
        if self.client.account:
            return self.client.account.address
        return "0x0000000000000000000000000000000000000000"
    
    async def get_balance(self, address: Optional[str] = None) -> int:
        """Get balance in wei."""
        return await self.client.get_balance(address)
