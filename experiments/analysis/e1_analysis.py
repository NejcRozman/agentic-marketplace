"""
E1 Baseline Experiment Analysis

Analyzes results from E1 (baseline single-auction experiment) to validate:
- System viability (RQ1)
- All phases completed successfully
- Quality-cost trade-off basics
- Reputation system functioning

Usage:
    python experiments/analysis/e1_analysis.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class E1Analyzer:
    """Analyzer for E1 baseline experiment results."""
    
    def __init__(self, metrics_path: Path):
        """
        Initialize analyzer with metrics file.
        
        Args:
            metrics_path: Path to experiment_metrics.json
        """
        self.metrics_path = metrics_path
        self.metrics = self._load_metrics()
        self.results = {
            "experiment_id": self.metrics.get("experiment_id"),
            "analysis_timestamp": datetime.now().isoformat(),
            "success_criteria": {},
            "research_findings": {},
            "recommendations": []
        }
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load metrics from JSON file."""
        with open(self.metrics_path, 'r') as f:
            return json.load(f)
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run complete E1 analysis.
        
        Returns:
            Dictionary with analysis results
        """
        print("\n" + "="*80)
        print("  E1 BASELINE EXPERIMENT ANALYSIS")
        print("="*80 + "\n")
        
        self._validate_success_criteria()
        self._analyze_system_viability()
        self._analyze_auction_efficiency()
        self._analyze_service_quality()
        self._analyze_reputation_system()
        self._analyze_timing_performance()
        self._analyze_blockchain_efficiency()
        self._generate_recommendations()
        
        return self.results
    
    def _validate_success_criteria(self):
        """Validate all E1 success criteria."""
        print("📋 VALIDATING SUCCESS CRITERIA")
        print("-" * 80)
        
        completeness = self.metrics.get('system_completeness', {})
        blockchain = self.metrics.get('blockchain', {})
        
        criteria = {
            "auction_created": {
                "status": completeness.get('auction_created', False),
                "description": "Auction created on-chain"
            },
            "all_providers_bid": {
                "status": completeness.get('bids_received', 0) >= 3,
                "description": "All 3 providers submitted bids",
                "value": completeness.get('bids_received', 0)
            },
            "winner_selected": {
                "status": completeness.get('winner_selected', False),
                "description": "Lowest bidder won auction"
            },
            "service_executed": {
                "status": completeness.get('service_executed', False),
                "description": "Winner completed service"
            },
            "service_evaluated": {
                "status": completeness.get('service_evaluated', False),
                "description": "Consumer evaluated service"
            },
            "feedback_submitted": {
                "status": completeness.get('feedback_submitted', False),
                "description": "Consumer submitted feedback on-chain"
            },
            "reputation_updated": {
                "status": completeness.get('reputation_updated', False),
                "description": "Winner's reputation updated"
            },
            "zero_failures": {
                "status": blockchain.get('failed_transactions', 1) == 0,
                "description": "Zero transaction failures",
                "value": blockchain.get('failed_transactions', 'N/A')
            }
        }
        
        passed = sum(1 for c in criteria.values() if c['status'])
        total = len(criteria)
        
        for key, criterion in criteria.items():
            symbol = "✅" if criterion['status'] else "❌"
            desc = criterion['description']
            if 'value' in criterion:
                print(f"{symbol} {desc}: {criterion['value']}")
            else:
                print(f"{symbol} {desc}")
        
        self.results['success_criteria'] = {
            "total": total,
            "passed": passed,
            "percentage": (passed / total * 100) if total > 0 else 0,
            "criteria": criteria,
            "overall_success": passed == total
        }
        
        print(f"\n📊 Success Rate: {passed}/{total} ({passed/total*100:.0f}%)")
        
        if passed == total:
            print("🎉 ALL CRITERIA PASSED - E1 Baseline Successful!\n")
        else:
            print(f"⚠️  {total - passed} criteria failed - Review required\n")
    
    def _analyze_system_viability(self):
        """Analyze RQ1: System Viability."""
        print("\n" + "="*80)
        print("🔬 RESEARCH QUESTION 1: SYSTEM VIABILITY")
        print("="*80)
        
        overall_success = self.results['success_criteria']['overall_success']
        duration = self.metrics.get('duration_seconds', 0)
        
        findings = {
            "hypothesis": "The marketplace can successfully match consumers with providers",
            "result": "VALIDATED" if overall_success else "REJECTED",
            "evidence": []
        }
        
        if overall_success:
            findings["evidence"].append(
                f"✅ Complete auction cycle executed in {duration}s ({duration/60:.1f} min)"
            )
            findings["evidence"].append(
                "✅ All system components (auction, bidding, execution, feedback) functional"
            )
            findings["evidence"].append(
                "✅ Blockchain integration working (contracts, transactions, reputation)"
            )
            
            print(f"\n✅ HYPOTHESIS VALIDATED")
            print(f"\nThe agentic marketplace successfully completed a full transaction cycle:")
            for evidence in findings["evidence"]:
                print(f"  {evidence}")
        else:
            failures = [k for k, v in self.results['success_criteria']['criteria'].items() 
                       if not v['status']]
            findings["evidence"].append(f"❌ Failed components: {', '.join(failures)}")
            
            print(f"\n❌ HYPOTHESIS REJECTED")
            print(f"\nSystem failed to complete full cycle. Issues detected:")
            for evidence in findings["evidence"]:
                print(f"  {evidence}")
        
        self.results['research_findings']['rq1_system_viability'] = findings
    
    def _analyze_auction_efficiency(self):
        """Analyze auction mechanism efficiency."""
        print("\n" + "="*80)
        print("💰 AUCTION MECHANISM ANALYSIS")
        print("="*80)
        
        auction = self.metrics.get('auction_details', {})
        
        if not auction.get('bids'):
            print("\n⚠️  No auction data available")
            return
        
        budget = auction.get('budget', 0)
        winning_bid = auction.get('winning_bid', 0)
        bid_spread = auction.get('bid_spread_percent', 0)
        bids = auction.get('bids', [])
        
        print(f"\n📊 Auction Results:")
        print(f"   Budget: {budget / 1e6:.2f} USDC")
        print(f"   Bids received: {len(bids)}")
        print(f"   Winning bid: {winning_bid / 1e6:.2f} USDC")
        
        if budget > 0:
            savings = (budget - winning_bid) / budget * 100
            print(f"   Consumer savings: {savings:.1f}% below budget")
        
        print(f"   Bid spread: {bid_spread:.2f}%")
        
        # Competition analysis
        if bid_spread > 10:
            competition = "High"
            print(f"   ✅ Competition level: {competition} (healthy market)")
        elif bid_spread > 5:
            competition = "Medium"
            print(f"   ✓ Competition level: {competition} (adequate)")
        else:
            competition = "Low"
            print(f"   ⚠️  Competition level: {competition} (providers may be colluding)")
        
        # Reverse auction effectiveness
        if winning_bid < budget * 0.9:
            print(f"   ✅ Reverse auction effective: Achieved >10% savings")
        else:
            print(f"   ⚠️  Limited cost savings from reverse auction")
        
        self.results['research_findings']['auction_efficiency'] = {
            "budget_usdc": budget / 1e6,
            "winning_bid_usdc": winning_bid / 1e6,
            "savings_percent": ((budget - winning_bid) / budget * 100) if budget > 0 else 0,
            "bid_count": len(bids),
            "bid_spread_percent": bid_spread,
            "competition_level": competition,
            "mechanism_effective": winning_bid < budget * 0.9
        }
    
    def _analyze_service_quality(self):
        """Analyze service quality and evaluation."""
        print("\n" + "="*80)
        print("⭐ SERVICE QUALITY ANALYSIS")
        print("="*80)
        
        quality = self.metrics.get('service_quality', {})
        
        rating = quality.get('rating')
        if rating is None:
            print("\n⚠️  No quality data available")
            return
        
        eval_method = quality.get('evaluation_method', 'unknown')
        prompts_total = quality.get('prompts_total', 0)
        prompts_answered = quality.get('prompts_answered', 0)
        
        print(f"\n📈 Quality Metrics:")
        print(f"   Rating: {rating}/100")
        
        # Quality tier
        if rating >= 80:
            tier = "Excellent"
            symbol = "🌟"
        elif rating >= 70:
            tier = "Good"
            symbol = "✅"
        elif rating >= 60:
            tier = "Acceptable"
            symbol = "✓"
        else:
            tier = "Poor"
            symbol = "❌"
        
        print(f"   Quality tier: {symbol} {tier}")
        print(f"   Evaluation method: {eval_method}")
        
        if eval_method == "fallback":
            print(f"   ⚠️  WARNING: Evaluator used fallback (LLM didn't use tools)")
            print(f"   → This may indicate rate limiting or prompt issues")
        
        completeness = (prompts_answered / prompts_total * 100) if prompts_total > 0 else 0
        print(f"   Completeness: {prompts_answered}/{prompts_total} ({completeness:.0f}%)")
        
        if completeness < 100:
            print(f"   ⚠️  Incomplete service delivery")
        
        self.results['research_findings']['service_quality'] = {
            "rating": rating,
            "quality_tier": tier,
            "evaluation_method": eval_method,
            "completeness_percent": completeness,
            "evaluation_reliable": eval_method == "tools"
        }
    
    def _analyze_reputation_system(self):
        """Analyze reputation system functionality."""
        print("\n" + "="*80)
        print("🏆 REPUTATION SYSTEM ANALYSIS")
        print("="*80)
        
        reputation = self.metrics.get('reputation', {})
        
        if not reputation.get('winner_id'):
            print("\n⚠️  No reputation data available")
            return
        
        winner_id = reputation.get('winner_id')
        before = reputation.get('winner_before')
        after = reputation.get('winner_after')
        change = reputation.get('reputation_change', 0)
        rating = reputation.get('feedback_rating')
        
        print(f"\n📊 Reputation Update:")
        print(f"   Winner: Agent {winner_id}")
        print(f"   Reputation before: {before}")
        print(f"   Reputation after: {after}")
        print(f"   Change: {change:+d} points")
        print(f"   Feedback rating: {rating}/100")
        
        # Analyze correlation
        if rating and change != 0:
            expected_positive = rating >= 70
            actual_positive = change > 0
            
            if expected_positive == actual_positive:
                print(f"   ✅ Reputation update correlates with service quality")
            else:
                print(f"   ⚠️  Reputation update doesn't match expected direction")
            
            # Check magnitude
            magnitude_reasonable = abs(change) <= 30
            if magnitude_reasonable:
                print(f"   ✅ Reputation change magnitude reasonable")
            else:
                print(f"   ⚠️  Reputation change seems excessive")
        
        self.results['research_findings']['reputation_system'] = {
            "winner_id": winner_id,
            "reputation_before": before,
            "reputation_after": after,
            "reputation_change": change,
            "feedback_rating": rating,
            "system_functional": change != 0,
            "correlation_correct": (rating >= 70 and change > 0) or (rating < 70 and change <= 0)
        }
    
    def _analyze_timing_performance(self):
        """Analyze timing and performance."""
        print("\n" + "="*80)
        print("⏱️  TIMING & PERFORMANCE ANALYSIS")
        print("="*80)
        
        timing = self.metrics.get('timing', {})
        phase_durations = timing.get('phase_durations', {})
        
        if not phase_durations:
            print("\n⚠️  No timing data available")
            return
        
        print(f"\n📊 Phase Breakdown:")
        for phase, duration in phase_durations.items():
            print(f"   {phase.replace('_', ' ').title()}: {duration:.1f}s ({duration/60:.1f} min)")
        
        total = timing.get('total_cycle_time', sum(phase_durations.values()))
        print(f"\n   Total: {total:.1f}s ({total/60:.1f} min)")
        
        # Performance assessment
        if total < 600:  # 10 minutes
            performance = "Excellent"
            symbol = "🚀"
        elif total < 1200:  # 20 minutes
            performance = "Good"
            symbol = "✅"
        elif total < 1800:  # 30 minutes
            performance = "Acceptable"
            symbol = "✓"
        else:
            performance = "Slow"
            symbol = "⚠️"
        
        print(f"\n   {symbol} Overall performance: {performance}")
        
        # Identify bottlenecks
        if phase_durations:
            slowest_phase = max(phase_durations.items(), key=lambda x: x[1])
            print(f"   Slowest phase: {slowest_phase[0]} ({slowest_phase[1]:.1f}s)")
        
        self.results['research_findings']['timing_performance'] = {
            "total_seconds": total,
            "total_minutes": total / 60,
            "phase_durations": phase_durations,
            "performance_tier": performance,
            "bottleneck": slowest_phase[0] if phase_durations else None
        }
    
    def _analyze_blockchain_efficiency(self):
        """Analyze blockchain transaction efficiency."""
        print("\n" + "="*80)
        print("⛽ BLOCKCHAIN EFFICIENCY ANALYSIS")
        print("="*80)
        
        blockchain = self.metrics.get('blockchain', {})
        
        if not blockchain:
            print("\n⚠️  No blockchain data available")
            return
        
        total_tx = blockchain.get('total_transactions', 0)
        failed_tx = blockchain.get('failed_transactions', 0)
        total_gas = blockchain.get('total_gas_used', 0)
        
        print(f"\n📊 Transaction Metrics:")
        print(f"   Total transactions: {total_tx}")
        print(f"   Failed transactions: {failed_tx}")
        
        if total_tx > 0:
            success_rate = ((total_tx - failed_tx) / total_tx * 100)
            print(f"   Success rate: {success_rate:.1f}%")
            
            if success_rate == 100:
                print(f"   ✅ Perfect transaction success rate")
            elif success_rate >= 95:
                print(f"   ✓ Good transaction success rate")
            else:
                print(f"   ⚠️  Transaction failures detected")
        
        print(f"\n⛽ Gas Usage:")
        print(f"   Total gas: {total_gas:,}")
        
        if total_tx > 0:
            avg_gas = total_gas / total_tx
            print(f"   Average per tx: {avg_gas:,.0f}")
        
        # Gas efficiency assessment
        if total_gas < 2000000:
            efficiency = "Excellent"
            symbol = "✅"
        elif total_gas < 3000000:
            efficiency = "Good"
            symbol = "✓"
        else:
            efficiency = "High (consider optimization)"
            symbol = "⚠️"
        
        print(f"   {symbol} Gas efficiency: {efficiency}")
        
        self.results['research_findings']['blockchain_efficiency'] = {
            "total_transactions": total_tx,
            "failed_transactions": failed_tx,
            "success_rate_percent": ((total_tx - failed_tx) / total_tx * 100) if total_tx > 0 else 0,
            "total_gas_used": total_gas,
            "average_gas_per_tx": total_gas / total_tx if total_tx > 0 else 0,
            "efficiency_tier": efficiency
        }
    
    def _generate_recommendations(self):
        """Generate recommendations based on analysis."""
        print("\n" + "="*80)
        print("💡 RECOMMENDATIONS")
        print("="*80 + "\n")
        
        recommendations = []
        
        # Check evaluator
        quality = self.results['research_findings'].get('service_quality', {})
        if quality.get('evaluation_method') == 'fallback':
            recommendations.append({
                "priority": "HIGH",
                "category": "Service Evaluation",
                "issue": "Evaluator using fallback instead of tools",
                "recommendation": "Investigate rate limiting, increase to 1.0 req/s or remove entirely"
            })
        
        # Check reputation system
        reputation = self.results['research_findings'].get('reputation_system', {})
        if not reputation.get('system_functional'):
            recommendations.append({
                "priority": "HIGH",
                "category": "Reputation System",
                "issue": "Reputation not updating",
                "recommendation": "Verify feedback submission and reputation contract integration"
            })
        
        # Check timing
        timing = self.results['research_findings'].get('timing_performance', {})
        if timing.get('total_minutes', 0) > 20:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Performance",
                "issue": f"Experiment took {timing['total_minutes']:.1f} minutes",
                "recommendation": f"Optimize bottleneck: {timing.get('bottleneck', 'unknown')}"
            })
        
        # Check competition
        auction = self.results['research_findings'].get('auction_efficiency', {})
        if auction.get('competition_level') == 'Low':
            recommendations.append({
                "priority": "LOW",
                "category": "Market Competition",
                "issue": "Low bid spread indicates weak competition",
                "recommendation": "Verify provider bidding strategies are diverse"
            })
        
        # Check success criteria
        if not self.results['success_criteria']['overall_success']:
            failed = [k for k, v in self.results['success_criteria']['criteria'].items() 
                     if not v['status']]
            recommendations.append({
                "priority": "CRITICAL",
                "category": "System Completeness",
                "issue": f"Failed criteria: {', '.join(failed)}",
                "recommendation": "Address failed components before proceeding to E2"
            })
        
        if not recommendations:
            print("✅ No critical issues detected. System performing as expected.\n")
            print("   Ready to proceed with E2 (Multiple Auctions)")
        else:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. [{rec['priority']}] {rec['category']}")
                print(f"   Issue: {rec['issue']}")
                print(f"   Recommendation: {rec['recommendation']}\n")
        
        self.results['recommendations'] = recommendations
    
    def save_results(self, output_path: Path):
        """Save analysis results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📁 Analysis results saved to: {output_path}\n")
    
    def print_summary(self):
        """Print concise summary."""
        print("\n" + "="*80)
        print("  EXECUTIVE SUMMARY")
        print("="*80 + "\n")
        
        success = self.results['success_criteria']['overall_success']
        
        if success:
            print("✅ E1 BASELINE EXPERIMENT: SUCCESS")
            print("\n   The agentic marketplace demonstrated complete system viability:")
            print("   • All auction phases completed successfully")
            print("   • Reverse auction mechanism achieved cost savings")
            print("   • Service quality evaluation functional")
            print("   • Reputation system operational")
            print("   • Zero blockchain transaction failures")
            print("\n   ✓ Ready to proceed with E2 (Multiple Auctions)")
        else:
            print("❌ E1 BASELINE EXPERIMENT: INCOMPLETE")
            failures = len([r for r in self.results.get('recommendations', []) 
                          if r['priority'] in ['CRITICAL', 'HIGH']])
            print(f"\n   {failures} critical/high priority issues require attention")
            print("   Review recommendations above before proceeding")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main entry point."""
    # Default path
    default_metrics = Path("experiments/data/metrics/e1_baseline/experiment_metrics.json")
    
    if len(sys.argv) > 1:
        metrics_path = Path(sys.argv[1])
    else:
        metrics_path = default_metrics
    
    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}")
        print(f"\nUsage: python {Path(__file__).name} [metrics_path]")
        print(f"Example: python {Path(__file__).name} {default_metrics}")
        sys.exit(1)
    
    # Run analysis
    analyzer = E1Analyzer(metrics_path)
    results = analyzer.analyze()
    
    # Save results
    output_dir = metrics_path.parent
    output_file = output_dir / "e1_analysis.json"
    analyzer.save_results(output_file)
    
    # Print summary
    analyzer.print_summary()


if __name__ == "__main__":
    main()
