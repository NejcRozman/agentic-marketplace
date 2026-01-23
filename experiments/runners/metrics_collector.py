"""
Metrics Collector - Comprehensive data collection for experiment analysis.

Collects:
- System completeness (all phases successful)
- Auction details (bids, winner, payments)
- Service quality (rating, evaluation, responses)
- Reputation changes (before/after, deltas)
- Timing (phase durations, critical path)
- Blockchain (gas costs, transaction success)
- Errors and reliability metrics
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from web3 import Web3

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects comprehensive experiment metrics for analysis."""
    
    def __init__(self, experiment_id: str, metrics_dir: Path, logs_dir: Path):
        """
        Initialize metrics collector.
        
        Args:
            experiment_id: Unique experiment identifier
            metrics_dir: Directory to save metrics JSON
            logs_dir: Directory containing agent logs
        """
        self.experiment_id = experiment_id
        self.metrics_dir = metrics_dir
        self.logs_dir = logs_dir
        
        self.metrics = {
            "experiment_id": experiment_id,
            "success": False,
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0,
            
            # Core data: Array of all auctions (primary data structure)
            "auctions": [],  # List of auction objects with all details
            
            # Aggregated counts (not rates or averages - those are calculated during analysis)
            "summary": {
                "total_auctions": 0,
                "completed_auctions": 0,
                "failed_auctions": 0,
                "total_bids_attempted": 0,
                "total_bids_on_chain": 0
            },
            
            # Reputation tracking over time
            "reputation_evolution": [],  # [{timestamp, provider_id, reputation_score, auction_id}]
            
            # Timing data (raw data, not averages)
            "timing": {
                "phase_durations": {},
                "execution_times": [],  # List of execution times per auction
                "evaluation_times": []  # List of evaluation times per auction
            },
            
            # Blockchain data (raw counts and gas, not averages)
            "blockchain": {
                "total_transactions": 0,
                "failed_transactions": 0,
                "gas_used_per_tx": []  # List of gas used per transaction
            },
            
            # Errors and issues
            "errors": {
                "total_errors": 0,
                "total_warnings": 0,
                "critical_issues": [],
                "bid_failures": []
            }
        }
        
    def set_start_time(self, start_time: datetime):
        """Record experiment start time."""
        self.metrics["start_time"] = start_time.isoformat()
        
    def set_end_time(self, end_time: datetime):
        """Record experiment end time."""
        self.metrics["end_time"] = end_time.isoformat()
        
    def set_contracts(self, reverse_auction: str, payment_token: str, 
                     identity_registry: str, reputation_registry: str):
        """Record deployed contract addresses."""
        self.metrics["contracts"] = {
            "reverse_auction": reverse_auction,
            "payment_token": payment_token,
            "identity_registry": identity_registry,
            "reputation_registry": reputation_registry
        }
        
    def set_agents(self, consumer_id: int, provider_ids: List[int]):
        """Record agent IDs."""
        self.metrics["agents"] = {
            "consumer_id": consumer_id,
            "provider_ids": provider_ids,
            "provider_count": len(provider_ids)
        }
    
    def collect_auction_from_blockchain(self, w3: Web3, auction_contract, auction_id: int, reputation_contract=None) -> Dict[str, Any]:
        """
        Collect complete auction data from blockchain and logs.
        
        This is the primary data collection method that builds the auctions array.
        Each auction object contains:
        - Auction details (id, budget, duration, service_cid)
        - All bid attempts (both successful and failed)
        - Winner information
        - Service execution details
        - Quality evaluation
        - Reputation changes
        - Timing breakdown
        
        Args:
            w3: Web3 instance
            auction_contract: ReverseAuction contract instance
            auction_id: ID of the auction to analyze
            
        Returns:
            Complete auction data dictionary
        """
        auction_data = {
            "auction_id": auction_id,
            "timestamp_created": None,
            "timestamp_ended": None,  # Actual auction end time from AuctionEnded event
            "service_cid": None,
            "budget": 0,
            "duration": 0,
            "creator_id": None,
            
            # Bidding data
            "bids_attempted": [],  # All bids providers tried to place
            "bids_on_chain": [],  # Only bids that succeeded on-chain
            "bid_failures": [],  # Failed bid attempts with reasons
            "winner_id": None,
            "winning_bid_amount": 0,
            "bid_count": 0,
            
            # Execution data
            "service_executed": False,
            "execution_start": None,
            "execution_end": None,
            "execution_duration_seconds": 0,
            "prompts_answered": 0,
            "prompts_total": 0,
            
            # Quality evaluation
            "quality_rating": None,
            "quality_method": "unknown",  # "tools", "fallback", or "failed"
            "quality_details": {},
            "evaluation_duration_seconds": 0,
            
            # Feedback and reputation
            "feedback_submitted": False,
            "feedback_rating": None,
            "reputation_before": None,
            "reputation_after": None,
            "reputation_change": 0,
            
            # Timing
            "timing": {
                "creation_to_first_bid": 0,
                "first_bid_to_close": 0,
                "close_to_execution_start": 0,
                "execution": 0,
                "evaluation": 0,
                "feedback_submission": 0,
                "total_auction_cycle": 0
            },
            
            # Status
            "status": "unknown",  # "completed", "failed", "timeout"
            "errors": []
        }
        
        try:
            # 1. Get on-chain auction data using getAuctionDetails() which properly returns full struct
            # This handles the dynamic array (eligibleAgentIds) correctly
            auction_info = auction_contract.functions.getAuctionDetails(auction_id).call()
            # Auction struct: (id, buyer, serviceDescriptionCid, maxPrice, duration, startTime, eligibleAgentIds, winningAgentId, winningBid, isActive, isCompleted, escrowAmount, reputationWeight)
            auction_data["creator_id"] = auction_info[1]  # buyer address
            auction_data["service_cid"] = auction_info[2]  # serviceDescriptionCid
            auction_data["budget"] = auction_info[3]  # maxPrice
            auction_data["duration"] = auction_info[4]  # duration
            auction_data["timestamp_created"] = auction_info[5]  # startTime
            auction_data["eligible_agent_ids"] = list(auction_info[6])  # eligibleAgentIds array
            
            # 2. Get on-chain bids
            bid_count = auction_contract.functions.getBidCount(auction_id).call()
            auction_data["bid_count"] = bid_count
            
            for i in range(bid_count):
                # Bid struct: (provider, agentId, amount, timestamp, reputation, score)
                bid_info = auction_contract.functions.auctionBids(auction_id, i).call()
                auction_data["bids_on_chain"].append({
                    "provider_address": bid_info[0],  # provider
                    "agent_id": bid_info[1],  # agentId
                    "amount": bid_info[2],  # amount
                    "timestamp": bid_info[3]  # timestamp
                })
            
            # 3. Get winner information from auction struct
            # auction_info[7] = winningAgentId, auction_info[8] = winningBid
            # auction_info[9] = isActive, auction_info[10] = isCompleted
            winning_agent_id = auction_info[7]  # winningAgentId (0 if no bids)
            winning_bid_from_struct = auction_info[8]  # winningBid from struct
            
            # Also get winningBid from separate mapping as it's more reliable
            try:
                winning_bid_from_mapping = auction_contract.functions.winningBid(auction_id).call()
            except:
                winning_bid_from_mapping = 0
            
            # Use mapping value if available, otherwise use struct value
            winning_bid = winning_bid_from_mapping if winning_bid_from_mapping > 0 else winning_bid_from_struct
            
            # Convert to proper types (winningAgentId is uint256, winningBid is uint256)
            auction_data["winner_id"] = int(winning_agent_id) if winning_agent_id else None
            auction_data["winning_bid_amount"] = int(winning_bid) if winning_bid else 0
            
            is_active = auction_info[9]
            is_completed = auction_info[10]
            
            # 3a. Get actual auction end timestamp from AuctionEnded event
            # This is the GROUND TRUTH for when the auction actually closed
            try:
                # Query AuctionEnded events for this auction
                event_filter = auction_contract.events.AuctionEnded.create_filter(
                    from_block=0,
                    argument_filters={'auctionId': auction_id}
                )
                events = event_filter.get_all_entries()
                if events:
                    # Get block timestamp for the AuctionEnded event
                    block = w3.eth.get_block(events[0]['blockNumber'])
                    auction_data["timestamp_ended"] = block['timestamp']
                else:
                    # Fallback: use theoretical end time if event not found
                    auction_data["timestamp_ended"] = auction_data["timestamp_created"] + auction_data["duration"]
            except Exception as e:
                logger.warning(f"Could not fetch AuctionEnded event for auction {auction_id}: {e}")
                # Fallback: use theoretical end time
                auction_data["timestamp_ended"] = auction_data["timestamp_created"] + auction_data["duration"]
            
            # 4. Collect bid attempts from provider logs (includes failures)
            # IMPORTANT: Only collect bids from eligible providers to avoid phantom data
            eligible_provider_ids = auction_data.get("eligible_agent_ids", [])
            
            for provider_id in self.metrics['agents']['provider_ids']:
                # Skip providers that weren't eligible for this auction
                if eligible_provider_ids and provider_id not in eligible_provider_ids:
                    continue
                
                provider_log = self._load_log_file(f"provider_{provider_id}.log")
                if not provider_log:
                    continue
                
                # Find all bid attempts for this auction - look for place_bid tool calls
                # Format: place_bid({'auction_id': 1, 'bid_amount': 60000000})
                # CRITICAL: Use word boundary after auction_id to avoid matching 1 with 10-19, 2 with 20-29, etc.
                bid_pattern = rf"place_bid\({{[^}}]*'auction_id':\s*{auction_id}[^\d][^}}]*'bid_amount':\s*(\d+)"
                for match in re.finditer(bid_pattern, provider_log):
                    bid_amount = int(match.group(1))
                    auction_data["bids_attempted"].append({
                        "provider_id": provider_id,
                        "amount": bid_amount
                    })
                
                # Track bid failures (only for eligible providers)
                # CRITICAL: Use word boundary after auction_id
                error_pattern = rf"Error placing bid.*auction.*{auction_id}\b"
                for match in re.finditer(error_pattern, provider_log, re.IGNORECASE):
                    context = provider_log[max(0, match.start()-200):match.end()+200]
                    auction_data["bid_failures"].append({
                        "provider_id": provider_id,
                        "reason": context.strip(),
                        "error_code": self._extract_error_code(context)
                    })
            
            # 5. Collect execution data from winner's log and job directory
            if auction_data["winner_id"]:
                winner_log = self._load_log_file(f"provider_{auction_data['winner_id']}.log")
                if winner_log:
                    # Extract auction-specific section to avoid cross-contamination
                    winner_section = self._extract_provider_auction_section(winner_log, auction_id)
                    if not winner_section:
                        # Fallback to full log if section extraction fails
                        winner_section = winner_log
                    
                    # Check for service execution markers (case-insensitive)
                    if any(marker.lower() in winner_section.lower() for marker in [
                        "result.json",
                        "result saved",
                        "literature review completed",
                        "processing prompt"
                    ]):
                        auction_data["service_executed"] = True
                    
                    # Extract prompts answered by counting "Processing prompt:" occurrences
                    prompt_count = winner_section.count("Processing prompt:")
                    if prompt_count > 0:
                        auction_data["prompts_answered"] = prompt_count
                    
                    # Set prompts_total to same as prompts_answered (actual execution count)
                    # The service description in logs is often truncated, so use actual execution
                    if prompt_count > 0:
                        auction_data["prompts_total"] = prompt_count
                    
                    # Extract execution timing from log timestamps (within auction-specific section)
                    execution_start_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*(?:Found \d+ PDF|Building vector database|Processing prompt)', winner_section)
                    execution_end_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*(?:Result saved to|Literature review completed)', winner_section)
                    
                    if execution_start_match:
                        auction_data["execution_start"] = execution_start_match.group(1)
                    if execution_end_match:
                        auction_data["execution_end"] = execution_end_match.group(1)
                    
                    # Calculate execution duration
                    if auction_data["execution_start"] and auction_data["execution_end"]:
                        try:
                            from datetime import datetime
                            start = datetime.strptime(auction_data["execution_start"], "%Y-%m-%d %H:%M:%S")
                            end = datetime.strptime(auction_data["execution_end"], "%Y-%m-%d %H:%M:%S")
                            auction_data["execution_duration_seconds"] = int((end - start).total_seconds())
                        except:
                            pass
            
            # 6. Collect evaluation data from consumer log
            # Use consumer agent ID, not the address
            consumer_id = self.metrics['agents']['consumer_id']
            consumer_log = self._load_log_file(f"consumer_{consumer_id}.log")
            if consumer_log:
                # Extract auction-specific section to avoid reusing first auction's data
                auction_section = self._extract_auction_section(consumer_log, auction_id)
                search_log = auction_section if auction_section else consumer_log
                
                # Rating - look for overall_rating first, then fall back to other rating patterns
                rating_match = re.search(r'"overall_rating":\s*(\d+)', search_log)
                if not rating_match:
                    rating_match = re.search(r'rating[=:]\s*(\d+).*?quality_scores', search_log, re.IGNORECASE | re.DOTALL)
                if rating_match:
                    auction_data["quality_rating"] = int(rating_match.group(1))
                    auction_data["feedback_rating"] = auction_data["quality_rating"]
                
                # Evaluation method
                if "fallback" in search_log.lower():
                    auction_data["quality_method"] = "fallback"
                elif "finalize_evaluation" in search_log or "quality_scores" in search_log:
                    auction_data["quality_method"] = "tools"
                    # Extract detailed scores if available
                    scores_match = re.search(r"quality_scores.*?({[^}]+})", search_log)
                    if scores_match:
                        try:
                            auction_data["quality_details"] = json.loads(scores_match.group(1).replace("'", '"'))
                        except:
                            pass
                
                # Feedback submission
                if "Feedback submitted" in search_log or "submitFeedback" in search_log:
                    auction_data["feedback_submitted"] = True
            
            # 7. Collect reputation data if winner exists and reputation contract provided
            if auction_data["winner_id"] and reputation_contract:
                try:
                    # Get current (after) reputation from blockchain using getSummary
                    # getSummary returns (feedbackCount, averageScore)
                    feedback_count, average_score = reputation_contract.functions.getSummary(
                        auction_data["winner_id"],
                        [],  # No client address filter
                        bytes(32),  # No tag1 filter
                        bytes(32)   # No tag2 filter
                    ).call()
                    auction_data["reputation_after"] = int(average_score)
                    
                    # Try to find initial reputation from winner's log
                    winner_log = self._load_log_file(f"provider_{auction_data['winner_id']}.log")
                    if winner_log:
                        # Look for get_reputation tool response: {"rating": 50, "feedback_count": 0}
                        # Use DOTALL flag to handle multiline logs
                        rep_before_match = re.search(r'get_reputation.*?"rating":\s*(\d+)', winner_log, re.IGNORECASE | re.DOTALL)
                        if rep_before_match:
                            auction_data["reputation_before"] = int(rep_before_match.group(1))
                    
                    # If not found in logs, calculate from reputation change
                    # Reputation after = Before + (rating - 50), so Before = After - (rating - 50)
                    if auction_data["reputation_before"] is None and auction_data["feedback_rating"]:
                        expected_change = auction_data["feedback_rating"] - 50
                        auction_data["reputation_before"] = auction_data["reputation_after"] - expected_change
                    
                    # Calculate change
                    if auction_data["reputation_before"] is not None:
                        auction_data["reputation_change"] = auction_data["reputation_after"] - auction_data["reputation_before"]
                except Exception as e:
                    logger.error(f"Failed to collect reputation for winner {auction_data['winner_id']}: {e}", exc_info=True)
            
            # 8. Calculate timing metrics from raw timestamps
            self._calculate_auction_timing(auction_data)
            
            # 9. Determine status based on blockchain state and logs
            if is_completed:
                # Contract marks auction as completed (service delivered and feedback received)
                auction_data["status"] = "completed"
            elif auction_data["feedback_submitted"]:
                # Feedback was submitted, so auction completed successfully
                auction_data["status"] = "completed"
            elif not is_active and auction_data["winner_id"] and auction_data["service_executed"]:
                # Auction ended, has winner, service executed - check for feedback
                auction_data["status"] = "completed" if auction_data["feedback_submitted"] else "incomplete"
            elif not is_active and auction_data["winner_id"]:
                # Auction ended with winner but service not executed yet
                auction_data["status"] = "incomplete"
            elif not is_active and (auction_data["winner_id"] == 0 or auction_data["winner_id"] is None):
                # Auction ended but no bids/winner
                auction_data["status"] = "failed"
            elif len(auction_data["errors"]) > 0:
                # Had errors during collection
                auction_data["status"] = "error"
            else:
                # Still active or other state
                auction_data["status"] = "incomplete"
                
        except Exception as e:
            logger.error(f"Error collecting auction {auction_id}: {e}")
            auction_data["errors"].append(str(e))
            auction_data["status"] = "error"
        
        # Validate data integrity: blockchain is ground truth
        self._validate_auction_data(auction_data)
        
        return auction_data
    
    def _calculate_auction_timing(self, auction_data: Dict[str, Any]):
        """Calculate timing metrics from raw timestamps."""
        timing = auction_data["timing"]
        
        try:
            # Creation to first bid
            if auction_data["bids_on_chain"] and len(auction_data["bids_on_chain"]) > 0:
                first_bid_ts = auction_data["bids_on_chain"][0]["timestamp"]
                timing["creation_to_first_bid"] = first_bid_ts - auction_data["timestamp_created"]
                
                # First bid to auction close
                # Use actual auction end timestamp from AuctionEnded event
                if auction_data.get("timestamp_ended"):
                    timing["first_bid_to_close"] = auction_data["timestamp_ended"] - first_bid_ts
                else:
                    # Fallback: use last bid timestamp as proxy
                    last_bid_ts = auction_data["bids_on_chain"][-1]["timestamp"]
                    timing["first_bid_to_close"] = last_bid_ts - first_bid_ts
            
            # Execution timing from log timestamps
            if auction_data["execution_start"] and auction_data["execution_end"]:
                from datetime import datetime
                start = datetime.strptime(auction_data["execution_start"], "%Y-%m-%d %H:%M:%S")
                end = datetime.strptime(auction_data["execution_end"], "%Y-%m-%d %H:%M:%S")
                timing["execution"] = int((end - start).total_seconds())
                
                # Close to execution start
                # Use actual auction end timestamp for accurate measurement
                if auction_data.get("timestamp_ended"):
                    execution_start_ts = int(start.timestamp())
                    timing["close_to_execution_start"] = execution_start_ts - auction_data["timestamp_ended"]
                    
                    # Note: This CAN be negative if execution started before AuctionEnded event
                    # This happens when provider starts work while auction is still active
                    # or if there's clock skew between blockchain and local system
                elif auction_data["bids_on_chain"]:
                    # Fallback: use last bid time as proxy
                    last_bid_ts = auction_data["bids_on_chain"][-1]["timestamp"]
                    execution_start_ts = int(start.timestamp())
                    timing["close_to_execution_start"] = execution_start_ts - last_bid_ts
            
            # Calculate total cycle time
            # Only calculate if all components are present (some may be 0 or negative)
            if timing["creation_to_first_bid"] > 0 and timing["execution"] > 0:
                timing["total_auction_cycle"] = (
                    timing["creation_to_first_bid"] + 
                    timing["first_bid_to_close"] + 
                    timing["close_to_execution_start"] + 
                    timing["execution"]
                )
        except Exception as e:
            logger.warning(f"Error calculating timing metrics: {e}")
    
    def _extract_error_code(self, text: str) -> Optional[str]:
        """Extract error code from error message."""
        match = re.search(r"0x[0-9a-fA-F]+", text)
        return match.group(0) if match else None
    
    def _validate_auction_data(self, auction_data: Dict[str, Any]):
        """
        Validate auction data integrity using blockchain as ground truth.
        
        This method performs critical sanity checks to catch data collection bugs:
        1. All bids_attempted providers must be in eligible_agent_ids
        2. All bids_on_chain providers must be in eligible_agent_ids
        3. Number of on-chain bids should match bid_count from contract
        4. Winner must be in bids_on_chain
        5. bids_attempted count should be >= bids_on_chain count
        
        Logs warnings for any inconsistencies found.
        """
        auction_id = auction_data["auction_id"]
        eligible = set(auction_data.get("eligible_agent_ids", []))
        
        # Validation 1: All attempted bids must be from eligible providers
        for bid in auction_data["bids_attempted"]:
            provider_id = bid.get("provider_id")
            if provider_id not in eligible:
                logger.error(
                    f"VALIDATION ERROR: Auction {auction_id} - "
                    f"Provider {provider_id} in bids_attempted but not in eligible_agent_ids {eligible}. "
                    f"This indicates a regex bug or incorrect eligibility filtering."
                )
                auction_data["errors"].append(f"Invalid bid attempt from ineligible provider {provider_id}")
        
        # Validation 2: All on-chain bids must be from eligible providers
        for bid in auction_data["bids_on_chain"]:
            agent_id = bid.get("agent_id")
            if agent_id and agent_id not in eligible:
                logger.error(
                    f"VALIDATION ERROR: Auction {auction_id} - "
                    f"Provider {agent_id} has on-chain bid but not in eligible_agent_ids {eligible}. "
                    f"This indicates a blockchain state inconsistency."
                )
                auction_data["errors"].append(f"On-chain bid from ineligible provider {agent_id}")
        
        # Validation 3: On-chain bid count consistency
        if len(auction_data["bids_on_chain"]) != auction_data["bid_count"]:
            logger.warning(
                f"VALIDATION WARNING: Auction {auction_id} - "
                f"Collected {len(auction_data['bids_on_chain'])} on-chain bids but "
                f"contract reports bid_count={auction_data['bid_count']}"
            )
        
        # Validation 4: Winner must have an on-chain bid
        if auction_data["winner_id"]:
            winner_bid = next(
                (b for b in auction_data["bids_on_chain"] if b.get("agent_id") == auction_data["winner_id"]),
                None
            )
            if not winner_bid:
                logger.warning(
                    f"VALIDATION WARNING: Auction {auction_id} - "
                    f"Winner {auction_data['winner_id']} has no on-chain bid recorded"
                )
        
        # Validation 5: Attempted bids should be >= on-chain bids
        attempted_count = len(auction_data["bids_attempted"])
        onchain_count = len(auction_data["bids_on_chain"])
        if attempted_count < onchain_count:
            logger.warning(
                f"VALIDATION WARNING: Auction {auction_id} - "
                f"Only {attempted_count} bids attempted but {onchain_count} made it on-chain. "
                f"This suggests incomplete log parsing."
            )
        
        # Validation 6: Sanity check for excessive bid attempts (likely a bug)
        if attempted_count > onchain_count * 10:
            logger.error(
                f"VALIDATION ERROR: Auction {auction_id} - "
                f"{attempted_count} bids attempted but only {onchain_count} on-chain. "
                f"Ratio of {attempted_count/max(onchain_count,1):.1f}x suggests phantom data from regex bug."
            )
            auction_data["errors"].append(
                f"Excessive bid attempts: {attempted_count} attempted vs {onchain_count} on-chain"
            )
        
        # Validation 7: Check for unreasonable negative timing values
        timing = auction_data.get("timing", {})
        close_to_exec = timing.get("close_to_execution_start", 0)
        
        # Small negative values (< 60s) are acceptable due to clock skew or early execution detection
        # Large negative values (> 60s) indicate a timing calculation bug
        if close_to_exec < -60:
            logger.warning(
                f"VALIDATION WARNING: Auction {auction_id} - "
                f"Execution started {abs(close_to_exec)}s before auction end. "
                f"This may indicate clock skew or incorrect timestamp parsing."
            )
        
        # Total cycle time should always be positive
        total_cycle = timing.get("total_auction_cycle", 0)
        if total_cycle < 0:
            logger.error(
                f"VALIDATION ERROR: Auction {auction_id} - "
                f"Negative total_auction_cycle ({total_cycle}s). "
                f"This indicates a timing calculation bug."
            )
            auction_data["errors"].append(f"Negative total cycle time: {total_cycle}s")

    def _count_transactions_from_logs(self):
        """Count blockchain transactions from all log files."""
        try:
            tx_count = 0
            
            # Check consumer log
            consumer_log = self._load_log_file(f"consumer_{self.metrics['agents']['consumer_id']}.log")
            if consumer_log:
                # Match both formats: with and without 0x prefix
                tx_count += len(re.findall(r'Sent transaction: (?:0x)?[0-9a-fA-F]{64}', consumer_log))
            
            # Check all provider logs
            for provider_id in self.metrics['agents']['provider_ids']:
                provider_log = self._load_log_file(f"provider_{provider_id}.log")
                if provider_log:
                    tx_count += len(re.findall(r'Sent transaction: (?:0x)?[0-9a-fA-F]{64}', provider_log))
            
            self.metrics["blockchain"]["total_transactions"] = tx_count
            logger.debug(f"Counted {tx_count} blockchain transactions from logs")
        except Exception as e:
            logger.warning(f"Error counting transactions: {e}")

    def collect_system_completeness(self) -> Dict[str, Any]:
        """
        Determine which phases completed successfully.
        
        Returns:
            Dictionary with boolean flags for each system phase
        """
        completeness = {
            "auction_created": False,
            "bids_received": 0,
            "winner_selected": False,
            "service_executed": False,
            "service_evaluated": False,
            "feedback_submitted": False,
            "reputation_updated": False
        }
        
        try:
            # Check consumer status
            consumer_status = self._load_json(self.logs_dir / f"consumer_{self.metrics['agents']['consumer_id']}_status.json")
            if consumer_status:
                completeness["auction_created"] = consumer_status.get("completed_auctions", 0) >= 1
                completeness["feedback_submitted"] = consumer_status.get("completed_auctions", 0) >= 1
            
            # Parse consumer log for auction and evaluation details
            consumer_log = self._load_log_file(f"consumer_{self.metrics['agents']['consumer_id']}.log")
            if consumer_log:
                completeness["auction_created"] = "Auction created" in consumer_log or "✅ Auction" in consumer_log
                completeness["service_evaluated"] = "Evaluation complete" in consumer_log or "rating=" in consumer_log
                completeness["feedback_submitted"] = "Feedback submitted" in consumer_log or "✅ Feedback submitted" in consumer_log
            
            # Count bids from provider logs
            for provider_id in self.metrics['agents']['provider_ids']:
                provider_log = self._load_log_file(f"provider_{provider_id}.log")
                if provider_log and ("Bid submitted" in provider_log or "✅ Bid placed" in provider_log):
                    completeness["bids_received"] += 1
                
                # Check if this provider won and executed
                if provider_log and ("Won auction" in provider_log or "✅ Won auction" in provider_log):
                    completeness["winner_selected"] = True
                if provider_log and ("Service completed" in provider_log or "✅ Service completed" in provider_log):
                    completeness["service_executed"] = True
            
            # Reputation update requires blockchain query (set by caller)
            
        except Exception as e:
            logger.error(f"Error collecting system completeness: {e}")
        
        self.metrics["system_completeness"] = completeness
        return completeness
    
    def collect_auction_details(self) -> Dict[str, Any]:
        """
        Extract auction details from logs and status files.
        
        Returns:
            Dictionary with auction ID, bids, winner, amounts, etc.
        """
        details = {
            "auction_id": None,
            "service_cid": None,
            "budget": None,
            "duration": None,
            "bids": [],
            "winner_id": None,
            "winning_bid": None,
            "bid_spread_percent": 0
        }
        
        try:
            consumer_log = self._load_log_file(f"consumer_{self.metrics['agents']['consumer_id']}.log")
            
            # Extract auction ID and service CID
            if consumer_log:
                auction_match = re.search(r"auction.*ID[=:]?\s*(\d+)", consumer_log, re.IGNORECASE)
                if auction_match:
                    details["auction_id"] = int(auction_match.group(1))
                
                cid_match = re.search(r"Service CID:\s*(Qm[a-zA-Z0-9]+)", consumer_log)
                if cid_match:
                    details["service_cid"] = cid_match.group(1)
                
                budget_match = re.search(r"budget[=:]?\s*(\d+)", consumer_log, re.IGNORECASE)
                if budget_match:
                    details["budget"] = int(budget_match.group(1))
                
                duration_match = re.search(r"duration[=:]?\s*(\d+)", consumer_log, re.IGNORECASE)
                if duration_match:
                    details["duration"] = int(duration_match.group(1))
                
                winner_match = re.search(r"winner[=:]?\s*(\d+)", consumer_log, re.IGNORECASE)
                if winner_match:
                    details["winner_id"] = int(winner_match.group(1))
            
            # Collect bids from provider logs
            for provider_id in self.metrics['agents']['provider_ids']:
                provider_log = self._load_log_file(f"provider_{provider_id}.log")
                if provider_log:
                    bid_match = re.search(r"(?:Bid|bid)[=:]?\s*(\d+)", provider_log)
                    if bid_match:
                        bid_amount = int(bid_match.group(1))
                        details["bids"].append({
                            "agent_id": provider_id,
                            "amount": bid_amount
                        })
            
            # Calculate winning bid and spread
            if details["bids"]:
                bid_amounts = [b["amount"] for b in details["bids"]]
                details["winning_bid"] = min(bid_amounts)
                if len(bid_amounts) > 1:
                    max_bid = max(bid_amounts)
                    min_bid = min(bid_amounts)
                    if max_bid > 0:
                        details["bid_spread_percent"] = round(((max_bid - min_bid) / max_bid) * 100, 2)
                
                # Set winner_id from bids if not already set
                if not details["winner_id"]:
                    winning_bid_entry = next((b for b in details["bids"] if b["amount"] == details["winning_bid"]), None)
                    if winning_bid_entry:
                        details["winner_id"] = winning_bid_entry["agent_id"]
            
        except Exception as e:
            logger.error(f"Error collecting auction details: {e}")
        
        self.metrics["auction_details"] = details
        return details
    
    def collect_service_quality(self) -> Dict[str, Any]:
        """
        Extract service quality metrics from consumer evaluation.
        
        Returns:
            Dictionary with rating, scores, evaluation method, etc.
        """
        quality = {
            "rating": None,
            "quality_scores": {},
            "evaluation_method": "unknown",
            "prompts_total": 0,
            "prompts_answered": 0,
            "average_response_length": 0,
            "execution_time_seconds": 0
        }
        
        try:
            consumer_log = self._load_log_file(f"consumer_{self.metrics['agents']['consumer_id']}.log")
            
            if consumer_log:
                # Extract rating
                rating_match = re.search(r"rating[=:]?\s*(\d+)", consumer_log, re.IGNORECASE)
                if rating_match:
                    quality["rating"] = int(rating_match.group(1))
                
                # Check if fallback was used
                if "fallback" in consumer_log.lower():
                    quality["evaluation_method"] = "fallback"
                    quality["quality_scores"] = {"fallback": quality["rating"] or 75}
                elif "finalize_evaluation" in consumer_log or "quality_scores" in consumer_log:
                    quality["evaluation_method"] = "tools"
                    # Try to extract structured scores
                    scores_match = re.search(r"quality_scores[=:]?\s*({[^}]+})", consumer_log)
                    if scores_match:
                        try:
                            quality["quality_scores"] = json.loads(scores_match.group(1).replace("'", '"'))
                        except:
                            pass
                
                # Count prompts
                prompt_matches = re.findall(r"prompt[s]?[=:]?", consumer_log, re.IGNORECASE)
                quality["prompts_total"] = len(prompt_matches) if prompt_matches else 5  # Default from config
                
                response_matches = re.findall(r"response[s]?[=:]?", consumer_log, re.IGNORECASE)
                quality["prompts_answered"] = len(response_matches) if response_matches else quality["prompts_total"]
            
            # Get execution time from winner's log
            winner_id = self.metrics.get("auction_details", {}).get("winner_id")
            if winner_id:
                winner_log = self._load_log_file(f"provider_{winner_id}.log")
                if winner_log:
                    # Look for service completion timing
                    time_match = re.search(r"(?:execution|service).*?(\d+)\s*(?:seconds|s)", winner_log, re.IGNORECASE)
                    if time_match:
                        quality["execution_time_seconds"] = int(time_match.group(1))
            
        except Exception as e:
            logger.error(f"Error collecting service quality: {e}")
        
        self.metrics["service_quality"] = quality
        return quality
    
    def collect_reputation(self, w3: Web3, reputation_contract, winner_id: int) -> Dict[str, Any]:
        """
        Query blockchain for reputation changes.
        
        Args:
            w3: Web3 instance
            reputation_contract: ReputationRegistry contract instance
            winner_id: Winner's agent ID
            
        Returns:
            Dictionary with before/after reputation and delta
        """
        reputation = {
            "winner_id": winner_id,
            "winner_before": None,
            "winner_after": None,
            "reputation_change": 0,
            "feedback_rating": self.metrics.get("service_quality", {}).get("rating")
        }
        
        try:
            # Query final reputation
            winner_reputation = reputation_contract.functions.getReputation(winner_id).call()
            reputation["winner_after"] = winner_reputation
            
            # Try to find initial reputation from logs or assume default
            winner_log = self._load_log_file(f"provider_{winner_id}.log")
            if winner_log:
                initial_match = re.search(r"(?:initial|starting).*reputation[=:]?\s*(\d+)", winner_log, re.IGNORECASE)
                if initial_match:
                    reputation["winner_before"] = int(initial_match.group(1))
            
            # If not found, assume default starting reputation (typically 50 for ERC-8004)
            if reputation["winner_before"] is None:
                reputation["winner_before"] = 50  # Default assumption
            
            reputation["reputation_change"] = reputation["winner_after"] - reputation["winner_before"]
            
        except Exception as e:
            logger.error(f"Error collecting reputation: {e}")
        
        self.metrics["reputation"] = reputation
        return reputation
    
    def collect_timing(self, phase_timings: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate timing metrics from experiment phases.
        
        Args:
            phase_timings: Dictionary of phase names to durations in seconds
            
        Returns:
            Dictionary with all timing breakdowns
        """
        timing = {
            "phase_durations": phase_timings,
            "total_cycle_time": sum(phase_timings.values()),
            "auction_creation_to_first_bid": 0,
            "auction_duration": 0,
            "execution_time": 0,
            "evaluation_time": 0,
            "feedback_submission_time": 0
        }
        
        try:
            # Extract specific timings from logs
            consumer_log = self._load_log_file(f"consumer_{self.metrics['agents']['consumer_id']}.log")
            
            if consumer_log:
                # Parse timestamps to calculate durations
                timestamps = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", consumer_log)
                
                # Get auction duration from config or log
                duration_match = re.search(r"duration[=:]?\s*(\d+)", consumer_log)
                if duration_match:
                    timing["auction_duration"] = int(duration_match.group(1))
            
            # Execution time from service quality
            timing["execution_time"] = self.metrics.get("service_quality", {}).get("execution_time_seconds", 0)
            
        except Exception as e:
            logger.error(f"Error collecting timing: {e}")
        
        self.metrics["timing"] = timing
        return timing
    
    def collect_blockchain_metrics(self, w3: Web3, tx_hashes: List[str]) -> Dict[str, Any]:
        """
        Query blockchain for transaction details and gas costs.
        
        Args:
            w3: Web3 instance
            tx_hashes: List of transaction hashes to analyze
            
        Returns:
            Dictionary with transaction counts and gas usage
        """
        blockchain = {
            "total_transactions": len(tx_hashes),
            "failed_transactions": 0,
            "total_gas_used": 0,
            "gas_by_operation": {},
            "transactions": []
        }
        
        try:
            for tx_hash in tx_hashes:
                try:
                    receipt = w3.eth.get_transaction_receipt(tx_hash)
                    tx = w3.eth.get_transaction(tx_hash)
                    
                    # Check if transaction failed
                    if receipt.status == 0:
                        blockchain["failed_transactions"] += 1
                    
                    gas_used = receipt.gasUsed
                    blockchain["total_gas_used"] += gas_used
                    
                    # Try to identify operation type from transaction data
                    operation = self._identify_operation(tx)
                    if operation:
                        blockchain["gas_by_operation"][operation] = blockchain["gas_by_operation"].get(operation, 0) + gas_used
                    
                    blockchain["transactions"].append({
                        "hash": tx_hash,
                        "gas_used": gas_used,
                        "status": "success" if receipt.status == 1 else "failed",
                        "operation": operation
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to get receipt for tx {tx_hash}: {e}")
            
        except Exception as e:
            logger.error(f"Error collecting blockchain metrics: {e}")
        
        self.metrics["blockchain"] = blockchain
        return blockchain
    
    def collect_errors(self) -> Dict[str, Any]:
        """
        Parse all log files for errors and warnings, focusing on critical issues.
        
        Returns:
            Dictionary with error counts and critical issues
        """
        errors = {
            "total_errors": 0,
            "total_warnings": 0,
            "critical_issues": [],  # Only blocking/important errors
            "bid_failures": []  # Already collected in auction data, kept for reference
        }
        
        try:
            # Scan all log files
            for log_file in self.logs_dir.glob("*.log"):
                content = log_file.read_text()
                
                # Count errors and warnings
                error_lines = [line for line in content.split('\n') if ' ERROR ' in line]
                warning_lines = [line for line in content.split('\n') if ' WARNING ' in line or ' WARN ' in line]
                
                errors["total_errors"] += len(error_lines)
                errors["total_warnings"] += len(warning_lines)
                
                # Identify critical issues (not just failed bids)
                for line in error_lines:
                    # Skip bid failures (already tracked in auction data)
                    if "Error placing bid" in line:
                        continue
                    
                    # Track critical errors only
                    if any(keyword in line.lower() for keyword in [
                        "failed to connect",
                        "connection refused", 
                        "timeout",
                        "contract not found",
                        "deployment failed",
                        "transaction reverted",
                        "insufficient funds",
                        "failed to fetch"
                    ]):
                        errors["critical_issues"].append({
                            "file": log_file.name,
                            "message": line.strip()[:300],
                            "category": self._categorize_error(line)
                        })
            
        except Exception as e:
            logger.error(f"Error collecting errors: {e}")
        
        self.metrics["errors"] = errors
        return errors
    
    def determine_success(self) -> bool:
        """
        Determine if experiment met success criteria.
        
        Returns:
            True if all critical phases completed successfully
        """
        completeness = self.metrics.get("system_completeness", {})
        
        success = (
            completeness.get("auction_created", False) and
            completeness.get("bids_received", 0) >= 1 and
            completeness.get("winner_selected", False) and
            completeness.get("service_executed", False) and
            completeness.get("service_evaluated", False) and
            completeness.get("feedback_submitted", False) and
            self.metrics.get("blockchain", {}).get("failed_transactions", 0) == 0
        )
        
        self.metrics["success"] = success
        return success
    
    def calculate_duration(self):
        """Calculate total experiment duration."""
        if self.metrics["start_time"] and self.metrics["end_time"]:
            start = datetime.fromisoformat(self.metrics["start_time"])
            end = datetime.fromisoformat(self.metrics["end_time"])
            self.metrics["duration_seconds"] = int((end - start).total_seconds())
    
    def save_metrics(self) -> Path:
        """
        Save all collected metrics to JSON file.
        
        Returns:
            Path to saved metrics file
        """
        self.calculate_duration()
        self._count_transactions_from_logs()
        
        metrics_file = self.metrics_dir / "experiment_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"✅ Comprehensive metrics saved to {metrics_file}")
        return metrics_file
    
    # Helper methods
    
    def _load_json(self, file_path: Path) -> Optional[Dict]:
        """Load JSON file safely."""
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
        return None
    
    def _load_log_file(self, filename: str) -> Optional[str]:
        """Load log file content."""
        log_path = self.logs_dir / filename
        try:
            if log_path.exists():
                return log_path.read_text()
        except Exception as e:
            logger.warning(f"Failed to load log {filename}: {e}")
        return None
    
    def _identify_operation(self, tx: Dict) -> Optional[str]:
        """Identify operation type from transaction data."""
        try:
            # Get function selector (first 4 bytes of input data)
            if tx.get('input') and len(tx['input']) >= 10:
                selector = tx['input'][:10]
                
                # Common function selectors (you can expand this)
                selectors = {
                    "0x095ea7b3": "approve_usdc",
                    "0xa9059cbb": "transfer_usdc",
                    # Add more selectors based on your contract ABIs
                }
                
                return selectors.get(selector, "unknown")
        except:
            pass
        return None
    
    def _categorize_error(self, error_line: str) -> str:
        """Categorize error based on message content."""
        error_lower = error_line.lower()
        
        if "self-feedback" in error_lower or "self feedback" in error_lower:
            return "self_feedback_blocked"
        elif "gas" in error_lower:
            return "gas_error"
        elif "revert" in error_lower:
            return "transaction_reverted"
        elif "timeout" in error_lower:
            return "timeout"
        elif "api" in error_lower or "rate limit" in error_lower:
            return "api_error"
        elif "connection" in error_lower or "network" in error_lower:
            return "network_error"
        else:
            return "other"
    
    # ========================================================================
    # Multi-Auction Experiment Methods (E2-E6)
    # ========================================================================
    
    def collect_multiple_auctions(self, auction_count: int) -> List[Dict[str, Any]]:
        """
        Collect metrics for multiple auctions (E2+).
        
        Args:
            auction_count: Expected number of auctions
            
        Returns:
            List of auction dictionaries with detailed metrics
        """
        auctions = []
        
        try:
            consumer_id = self.metrics['agents']['consumer_id']
            consumer_log = self._load_log_file(f"consumer_{consumer_id}.log")
            
            if not consumer_log:
                logger.warning("Could not load consumer log for multi-auction analysis")
                return auctions
            
            # Parse all auction IDs
            auction_ids = re.findall(r"auction.*ID[=:]?\s*(\d+)", consumer_log, re.IGNORECASE)
            unique_auction_ids = sorted(set(int(aid) for aid in auction_ids))
            
            logger.info(f"Found {len(unique_auction_ids)} auctions: {unique_auction_ids}")
            
            for auction_id in unique_auction_ids:
                auction_data = self._collect_single_auction_metrics(auction_id, consumer_log)
                auctions.append(auction_data)
            
        except Exception as e:
            logger.error(f"Error collecting multiple auctions: {e}")
        
        self.metrics["auctions"] = auctions
        return auctions
    
    def _collect_single_auction_metrics(self, auction_id: int, consumer_log: str) -> Dict[str, Any]:
        """
        Collect metrics for a single auction.
        
        Args:
            auction_id: Auction ID to analyze
            consumer_log: Consumer's full log content
            
        Returns:
            Dictionary with auction metrics
        """
        auction = {
            "auction_id": auction_id,
            "service_cid": None,
            "budget": None,
            "bids": [],
            "winner_id": None,
            "winning_bid": None,
            "rating": None,
            "reputation_change": None,
            "timestamps": {}
        }
        
        try:
            # Extract auction-specific logs (look for lines near this auction ID)
            auction_section = self._extract_auction_section(consumer_log, auction_id)
            
            # Parse bids for this auction
            for provider_id in self.metrics['agents']['provider_ids']:
                provider_log = self._load_log_file(f"provider_{provider_id}.log")
                if provider_log:
                    # Find bid for this specific auction
                    # CRITICAL: Use word boundary after auction_id
                    bid_pattern = rf"auction\s*{auction_id}\b.*bid[=:]?\s*(\d+)"
                    bid_match = re.search(bid_pattern, provider_log, re.IGNORECASE)
                    if bid_match:
                        auction["bids"].append({
                            "agent_id": provider_id,
                            "amount": int(bid_match.group(1))
                        })
            
            # Determine winner
            if auction["bids"]:
                auction["winning_bid"] = min(b["amount"] for b in auction["bids"])
                winner = next((b for b in auction["bids"] if b["amount"] == auction["winning_bid"]), None)
                if winner:
                    auction["winner_id"] = winner["agent_id"]
            
            # Extract rating for this auction
            # CRITICAL: Use word boundary after auction_id
            rating_pattern = rf"auction\s*{auction_id}\b.*rating[=:]?\s*(\d+)"
            rating_match = re.search(rating_pattern, auction_section or consumer_log, re.IGNORECASE)
            if rating_match:
                auction["rating"] = int(rating_match.group(1))
            
        except Exception as e:
            logger.warning(f"Error collecting metrics for auction {auction_id}: {e}")
        
        return auction
    
    def _extract_auction_section(self, log: str, auction_id: int) -> Optional[str]:
        """Extract log section relevant to specific auction."""
        lines = log.split('\n')
        relevant_lines = []
        start_idx = None
        end_idx = None
        
        # Use regex patterns with word boundaries to avoid partial matches
        import re
        eval_pattern = re.compile(rf"Evaluating result for auction {auction_id}\b")
        created_pattern = re.compile(rf"(?:Auction|Created auction) {auction_id}\b")
        next_eval_pattern = re.compile(rf"Evaluating result for auction {auction_id + 1}\b")
        next_created_pattern = re.compile(rf"Auction {auction_id + 1}\b")
        feedback_pattern = re.compile(rf"Feedback submitted.*auction {auction_id}\b")
        
        # Find the start of this auction's section
        for i, line in enumerate(lines):
            # Look for evaluation start marker
            if eval_pattern.search(line):
                start_idx = i
                break
            # Fallback to auction creation if evaluation not found
            elif start_idx is None and created_pattern.search(line):
                start_idx = i
        
        # Find the end of this auction's section
        if start_idx is not None:
            for i in range(start_idx, len(lines)):
                line = lines[i]
                # Stop when we see next auction creation or evaluation
                if i > start_idx and (next_eval_pattern.search(line) or 
                                     next_created_pattern.search(line) or
                                     "Creating next auction" in line):
                    end_idx = i
                    break
                # Also stop after feedback submission for this auction
                if feedback_pattern.search(line):
                    end_idx = min(len(lines), i + 5)  # Include a few more lines
                    break
        
        if start_idx is not None and end_idx is not None:
            relevant_lines = lines[start_idx:end_idx]
        elif start_idx is not None:
            # If no end found, take from start to end of file (last auction)
            relevant_lines = lines[start_idx:]
        
        return '\n'.join(relevant_lines) if relevant_lines else None
    
    def _extract_provider_auction_section(self, log: str, auction_id: int) -> Optional[str]:
        """Extract provider log section for a specific auction execution."""
        lines = log.split('\n')
        relevant_lines = []
        start_idx = None
        end_idx = None
        
        import re
        # Look for "Won auction X" or "Executing service for job X"
        won_pattern = re.compile(rf"Won auction {auction_id}\b", re.IGNORECASE)
        executing_pattern = re.compile(rf"Executing service for job {auction_id}\b", re.IGNORECASE)
        # Look for completion or next auction
        completed_pattern = re.compile(rf"Literature review completed", re.IGNORECASE)
        next_won_pattern = re.compile(rf"Won auction {auction_id + 1}\b", re.IGNORECASE)
        
        # Find the start (when won the auction)
        for i, line in enumerate(lines):
            if won_pattern.search(line) or executing_pattern.search(line):
                start_idx = i
                break
        
        # Find the end (completion or next auction win)
        if start_idx is not None:
            for i in range(start_idx + 1, len(lines)):
                line = lines[i]
                # Stop at completion of current auction
                if completed_pattern.search(line):
                    # Include this line and a few more for context
                    end_idx = min(len(lines), i + 5)
                    break
                # Stop if we see next auction
                if next_won_pattern.search(line):
                    end_idx = i
                    break
        
        if start_idx is not None and end_idx is not None:
            relevant_lines = lines[start_idx:end_idx]
        elif start_idx is not None:
            # If no end found, take from start to end (last auction)
            relevant_lines = lines[start_idx:]
        
        return '\n'.join(relevant_lines) if relevant_lines else None
    
    def track_reputation_evolution(self, w3: Web3, reputation_contract, 
                                   auction_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Track reputation changes across multiple auctions.
        
        Args:
            w3: Web3 instance
            reputation_contract: ReputationRegistry contract
            auction_ids: List of auction IDs in order
            
        Returns:
            List of reputation snapshots over time
        """
        evolution = []
        
        try:
            for i, auction_id in enumerate(auction_ids):
                snapshot = {
                    "auction_id": auction_id,
                    "sequence": i + 1,
                    "provider_reputations": {}
                }
                
                # Query current reputation for all providers
                for provider_id in self.metrics['agents']['provider_ids']:
                    try:
                        # getSummary returns (feedbackCount, averageScore)
                        feedback_count, average_score = reputation_contract.functions.getSummary(
                            provider_id,
                            [],  # No client address filter
                            bytes(32),  # No tag1 filter
                            bytes(32)   # No tag2 filter
                        ).call()
                        snapshot["provider_reputations"][provider_id] = {
                            "score": int(average_score),
                            "feedback_count": int(feedback_count)
                        }
                    except Exception as e:
                        logger.warning(f"Could not get reputation for provider {provider_id}: {e}")
                
                evolution.append(snapshot)
            
        except Exception as e:
            logger.error(f"Error tracking reputation evolution: {e}")
        
        self.metrics["reputation_evolution"] = evolution
        return evolution
    
    def calculate_provider_profitability(self) -> Dict[int, Dict[str, float]]:
        """
        Calculate profit/loss for each provider (E4+).
        
        Returns:
            Dictionary mapping provider_id to profitability metrics
        """
        profitability = {}
        
        try:
            for provider_id in self.metrics['agents']['provider_ids']:
                provider_log = self._load_log_file(f"provider_{provider_id}.log")
                if not provider_log:
                    continue
                
                # Count wins
                wins = len(re.findall(r"won auction", provider_log, re.IGNORECASE))
                
                # Sum revenue from winning bids
                revenue = 0
                for auction in self.metrics.get("auctions", []):
                    if auction.get("winner_id") == provider_id:
                        revenue += auction.get("winning_bid", 0)
                
                # Estimate costs (simplified - could be enhanced)
                # Cost = gas + computation (approximate)
                estimated_cost_per_service = 5000000  # 5 USDC per service (estimate)
                total_costs = wins * estimated_cost_per_service
                
                profit = revenue - total_costs
                
                profitability[provider_id] = {
                    "wins": wins,
                    "revenue_usdc": revenue / 1e6,
                    "costs_usdc": total_costs / 1e6,
                    "profit_usdc": profit / 1e6,
                    "profit_margin": (profit / revenue * 100) if revenue > 0 else 0
                }
            
        except Exception as e:
            logger.error(f"Error calculating profitability: {e}")
        
        self.metrics["provider_profitability"] = profitability
        return profitability
    
    def analyze_market_dynamics(self) -> Dict[str, Any]:
        """
        Analyze market trends and equilibrium (E4+).
        
        Returns:
            Dictionary with market analysis
        """
        dynamics = {
            "average_winning_bid": 0,
            "bid_trend": "stable",  # increasing, decreasing, stable
            "price_volatility": 0,
            "market_equilibrium_price": 0,
            "competition_level": "medium"
        }
        
        try:
            auctions = self.metrics.get("auctions", [])
            if not auctions:
                return dynamics
            
            winning_bids = [a["winning_bid"] for a in auctions if a.get("winning_bid")]
            
            if winning_bids:
                # Calculate average
                dynamics["average_winning_bid"] = sum(winning_bids) / len(winning_bids) / 1e6
                
                # Detect trend
                if len(winning_bids) >= 3:
                    first_third = sum(winning_bids[:len(winning_bids)//3]) / (len(winning_bids)//3)
                    last_third = sum(winning_bids[-len(winning_bids)//3:]) / (len(winning_bids)//3)
                    
                    change = ((last_third - first_third) / first_third) * 100
                    if change > 5:
                        dynamics["bid_trend"] = "increasing"
                    elif change < -5:
                        dynamics["bid_trend"] = "decreasing"
                    else:
                        dynamics["bid_trend"] = "stable"
                
                # Calculate volatility (standard deviation)
                mean_bid = sum(winning_bids) / len(winning_bids)
                variance = sum((b - mean_bid) ** 2 for b in winning_bids) / len(winning_bids)
                dynamics["price_volatility"] = (variance ** 0.5) / mean_bid * 100  # CV%
                
                # Estimate equilibrium (median of middle 50%)
                sorted_bids = sorted(winning_bids)
                q1 = len(sorted_bids) // 4
                q3 = 3 * len(sorted_bids) // 4
                middle_bids = sorted_bids[q1:q3]
                if middle_bids:
                    dynamics["market_equilibrium_price"] = sum(middle_bids) / len(middle_bids) / 1e6
                
                # Competition level based on bid count and spread
                avg_bid_count = sum(len(a.get("bids", [])) for a in auctions) / len(auctions)
                if avg_bid_count >= 3:
                    dynamics["competition_level"] = "high"
                elif avg_bid_count >= 2:
                    dynamics["competition_level"] = "medium"
                else:
                    dynamics["competition_level"] = "low"
            
        except Exception as e:
            logger.error(f"Error analyzing market dynamics: {e}")
        
        self.metrics["market_dynamics"] = dynamics
        return dynamics
    
    def analyze_strategy_performance(self, provider_strategies: Dict[int, str]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze performance by provider strategy (E4+).
        
        Args:
            provider_strategies: Mapping of provider_id to strategy name
            
        Returns:
            Dictionary mapping strategy to performance metrics
        """
        strategy_perf = {}
        
        try:
            auctions = self.metrics.get("auctions", [])
            
            # Group providers by strategy
            strategy_providers = {}
            for provider_id, strategy in provider_strategies.items():
                if strategy not in strategy_providers:
                    strategy_providers[strategy] = []
                strategy_providers[strategy].append(provider_id)
            
            # Calculate metrics per strategy
            for strategy, provider_ids in strategy_providers.items():
                wins = 0
                total_auctions = len(auctions)
                total_revenue = 0
                ratings = []
                
                for auction in auctions:
                    winner_id = auction.get("winner_id")
                    if winner_id in provider_ids:
                        wins += 1
                        total_revenue += auction.get("winning_bid", 0)
                        if auction.get("rating"):
                            ratings.append(auction["rating"])
                
                strategy_perf[strategy] = {
                    "provider_count": len(provider_ids),
                    "wins": wins,
                    "win_rate": (wins / total_auctions * 100) if total_auctions > 0 else 0,
                    "total_revenue_usdc": total_revenue / 1e6,
                    "average_rating": sum(ratings) / len(ratings) if ratings else 0,
                    "rating_consistency": self._calculate_std_dev(ratings) if len(ratings) > 1 else 0
                }
            
        except Exception as e:
            logger.error(f"Error analyzing strategy performance: {e}")
        
        self.metrics["strategy_performance"] = strategy_perf
        return strategy_perf
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5
    
    def detect_adversarial_patterns(self) -> Dict[str, Any]:
        """
        Detect potential adversarial behavior patterns (E5+).
        
        Returns:
            Dictionary with detected suspicious patterns
        """
        patterns = {
            "suspicious_providers": [],
            "pattern_types": {},
            "risk_level": "low"
        }
        
        try:
            auctions = self.metrics.get("auctions", [])
            
            for provider_id in self.metrics['agents']['provider_ids']:
                suspicion_score = 0
                detected_patterns = []
                
                # Pattern 1: Consistently lowest bids but poor quality
                provider_auctions = [a for a in auctions if a.get("winner_id") == provider_id]
                if len(provider_auctions) >= 3:
                    avg_rating = sum(a.get("rating", 0) for a in provider_auctions) / len(provider_auctions)
                    if avg_rating < 50:
                        suspicion_score += 2
                        detected_patterns.append("low_quality_winner")
                
                # Pattern 2: Abnormal bidding behavior
                provider_log = self._load_log_file(f"provider_{provider_id}.log")
                if provider_log:
                    error_count = provider_log.lower().count("error")
                    if error_count > 10:
                        suspicion_score += 1
                        detected_patterns.append("high_error_rate")
                
                # Pattern 3: Reputation recovery attempts
                rep_evolution = self.metrics.get("reputation_evolution", [])
                if len(rep_evolution) >= 5:
                    provider_reps = [snapshot["provider_reputations"].get(provider_id, 50) 
                                    for snapshot in rep_evolution]
                    # Check for sharp drops followed by suspicious recovery
                    for i in range(1, len(provider_reps) - 1):
                        if provider_reps[i] < provider_reps[i-1] - 10 and provider_reps[i+1] > provider_reps[i] + 15:
                            suspicion_score += 1
                            detected_patterns.append("reputation_manipulation")
                            break
                
                if suspicion_score >= 2:
                    patterns["suspicious_providers"].append({
                        "provider_id": provider_id,
                        "suspicion_score": suspicion_score,
                        "patterns": detected_patterns
                    })
                    
                    for pattern in detected_patterns:
                        patterns["pattern_types"][pattern] = patterns["pattern_types"].get(pattern, 0) + 1
            
            # Determine overall risk level
            if len(patterns["suspicious_providers"]) >= 2:
                patterns["risk_level"] = "high"
            elif len(patterns["suspicious_providers"]) == 1:
                patterns["risk_level"] = "medium"
            else:
                patterns["risk_level"] = "low"
            
        except Exception as e:
            logger.error(f"Error detecting adversarial patterns: {e}")
        
        self.metrics["adversarial_patterns"] = patterns
        return patterns
