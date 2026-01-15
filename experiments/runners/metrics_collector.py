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
            "system_completeness": {},
            "auctions": [],  # List of all auctions (for multi-auction experiments)
            "auction_details": {},  # Legacy single auction support (E1)
            "service_quality": {},
            "reputation": {},
            "reputation_evolution": [],  # Track reputation over time (E2+)
            "provider_profitability": {},  # Revenue/costs per provider (E4+)
            "market_dynamics": {},  # Price trends, equilibrium (E4+)
            "strategy_performance": {},  # Win rates by strategy (E4+)
            "timing": {},
            "blockchain": {},
            "errors": {}
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
        Parse all log files for errors and warnings.
        
        Returns:
            Dictionary with error counts and types
        """
        errors = {
            "total_errors": 0,
            "total_warnings": 0,
            "error_types": {},
            "error_details": []
        }
        
        try:
            # Scan all log files
            for log_file in self.logs_dir.glob("*.log"):
                content = log_file.read_text()
                
                # Count errors and warnings
                error_lines = [line for line in content.split('\n') if 'ERROR' in line]
                warning_lines = [line for line in content.split('\n') if 'WARNING' in line or 'WARN' in line]
                
                errors["total_errors"] += len(error_lines)
                errors["total_warnings"] += len(warning_lines)
                
                # Categorize errors
                for line in error_lines:
                    error_type = self._categorize_error(line)
                    errors["error_types"][error_type] = errors["error_types"].get(error_type, 0) + 1
                    
                    # Store first 10 error details
                    if len(errors["error_details"]) < 10:
                        errors["error_details"].append({
                            "file": log_file.name,
                            "message": line[:200]  # Truncate long messages
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
                    bid_pattern = rf"auction\s*{auction_id}.*bid[=:]?\s*(\d+)"
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
            rating_pattern = rf"auction\s*{auction_id}.*rating[=:]?\s*(\d+)"
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
        in_auction = False
        
        for line in lines:
            if f"auction {auction_id}" in line.lower() or f"auction={auction_id}" in line.lower():
                in_auction = True
                relevant_lines.append(line)
            elif in_auction:
                relevant_lines.append(line)
                # Stop at next auction or after 50 lines
                if "auction" in line.lower() and str(auction_id) not in line:
                    break
                if len(relevant_lines) > 50:
                    break
        
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
                        reputation = reputation_contract.functions.getReputation(provider_id).call()
                        snapshot["provider_reputations"][provider_id] = reputation
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
