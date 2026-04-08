"""Focused regression tests for provider Architecture 2 behavior."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock

from agents.config import Config, get_architecture
from agents.infrastructure.cost_tracker import CostTracker
from agents.infrastructure.prompts import get_prompt
from agents.provider_agent.blockchain_handler import BlockchainHandler


class TestArchitecture2Config(unittest.TestCase):
    """Validate architecture-2 registry wiring."""

    def test_architecture_2_registry_entry(self):
        arch = get_architecture("2")

        self.assertEqual(arch.state_level, 1)
        self.assertEqual(arch.reasoning_mode, "llm_react")
        self.assertEqual(arch.coupling_mode, "one_way")
        self.assertEqual(arch.prompt_template, "2")
        self.assertIn("place_bid", arch.enabled_tools)


class TestArchitecture2Prompts(unittest.TestCase):
    """Validate A1/A2 prompt visibility boundaries."""

    def test_a1_and_a2_share_core_but_differ_in_visibility(self):
        state = {
            "agent_id": 42,
            "eligible_active_auctions": [],
            "agent_reputation": {"rating": 77, "feedback_count": 5},
            "competitors_reputation": [{"agent_id": 9, "rating": 81, "feedback_count": 7}],
            "estimated_service_cost": 0.12,
            "past_execution_costs": [0.11, 0.13, 0.12],
            "current_balance": -2.5,
        }

        prompt_a1 = get_prompt("1", state)
        prompt_a2 = get_prompt("2", state)

        # Shared strategic core should be present in both architectures.
        self.assertIn("Primary objective:", prompt_a1)
        self.assertIn("Primary objective:", prompt_a2)
        self.assertIn("Strategic policy:", prompt_a1)
        self.assertIn("Strategic policy:", prompt_a2)

        # A1: explicit unavailability markers.
        self.assertIn("current_balance_usd: unavailable in this architecture", prompt_a1)
        self.assertIn("past_execution_costs: unavailable in this architecture", prompt_a1)

        # A2: history and balance are visible and summarized.
        self.assertIn("current_balance_usd: -2.5", prompt_a2)
        self.assertIn("past_execution_costs_count: 3", prompt_a2)
        self.assertIn("past_execution_costs_avg_usd: 0.12", prompt_a2)


class TestArchitecture2StateCoupling(unittest.TestCase):
    """Validate A2 state-level coupling behavior in gather-state path."""

    def setUp(self):
        self.config = Config()

    @staticmethod
    def _build_mock_client(auction_count: int = 0):
        mock_client = MagicMock()
        mock_client.get_block_number = AsyncMock(return_value=1000)
        mock_client.get_contract_events = AsyncMock(return_value=[])
        mock_client.get_block = AsyncMock(return_value={"timestamp": 1234567890})

        async def _call_contract_method(_contract_name, method_name, *_args):
            if method_name == "auctionIdCounter":
                return auction_count
            if method_name == "isEligibleAgent":
                return False
            raise AssertionError(f"Unexpected method call in test: {method_name}")

        mock_client.call_contract_method = AsyncMock(side_effect=_call_contract_method)
        return mock_client

    def test_a2_uses_latest_execution_cost_for_estimate(self):
        tracker = CostTracker(agent_id="test_agent", config=self.config)
        tracker.get_execution_cost_history = Mock(return_value=[0.08, 0.11, 0.15])
        tracker.get_net_balance = Mock(return_value=-1.75)

        handler = BlockchainHandler(
            agent_id=1,
            blockchain_client=self._build_mock_client(),
            architecture="2",
            cost_tracker=tracker,
        )
        handler.contracts_loaded = True
        handler.reputation_registry_contract = None

        state = {"agent_id": 1, "action": "monitor", "messages": []}
        result = asyncio.run(handler._gather_state_node(state))

        self.assertEqual(result.get("estimated_service_cost"), 0.15)
        self.assertEqual(result.get("past_execution_costs"), [0.08, 0.11, 0.15])
        self.assertAlmostEqual(result.get("current_balance"), -1.75, places=6)

    def test_a2_falls_back_to_base_cost_when_history_empty(self):
        tracker = CostTracker(agent_id="test_agent", config=self.config)
        tracker.get_execution_cost_history = Mock(return_value=[])
        tracker.get_net_balance = Mock(return_value=0.0)

        handler = BlockchainHandler(
            agent_id=1,
            blockchain_client=self._build_mock_client(),
            architecture="2",
            cost_tracker=tracker,
        )
        handler.contracts_loaded = True
        handler.reputation_registry_contract = None

        state = {"agent_id": 1, "action": "monitor", "messages": []}
        result = asyncio.run(handler._gather_state_node(state))

        self.assertEqual(result.get("estimated_service_cost"), handler.config.bidding_base_cost)
        self.assertEqual(result.get("past_execution_costs"), [])
        self.assertEqual(result.get("current_balance"), 0.0)


class TestArchitecture2ToolAuditContext(unittest.TestCase):
    """Validate tool-audit context includes A2 coupling fields."""

    def test_tool_audit_context_contains_balance_and_history(self):
        handler = BlockchainHandler(
            agent_id=7,
            blockchain_client=MagicMock(),
            architecture="2",
        )

        state = {
            "agent_reputation": {"rating": 70, "feedback_count": 2},
            "estimated_service_cost": 0.055,
            "past_execution_costs": [0.05, 0.06],
            "current_balance": 3.25,
            "eligible_active_auctions": [
                {
                    "auction_id": 123,
                    "max_price": 100000,
                    "winning_bid": 70000,
                    "reputation_weight": 20,
                    "time_remaining": 45,
                }
            ],
        }

        context = handler._build_tool_audit_context(state)

        self.assertEqual(context["agent_id"], 7)
        self.assertEqual(context["agent_reputation"], 70)
        self.assertEqual(context["estimated_service_cost_usd"], 0.055)
        self.assertEqual(context["estimated_service_cost_micro"], 55000)
        self.assertEqual(context["past_execution_costs"], [0.05, 0.06])
        self.assertEqual(context["past_execution_costs_count"], 2)
        self.assertEqual(context["current_balance_usd"], 3.25)
        self.assertEqual(context["auction_count"], 1)
        self.assertIn("123", context["auctions"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
