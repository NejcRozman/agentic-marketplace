"""
Unit tests for BlockchainHandler._heuristic_reasoning.

These tests exercise _heuristic_reasoning directly with a mocked blockchain
client so NO running Anvil or deployed contracts are required.

Strategies covered
------------------
  random_markup   : bid = cost × Uniform(1+min_margin, 1+max_margin)
  feasible_random : bid ~ Uniform[min_profitable_bid, max_competitive_bid]

Test categories
---------------
  TestRandomMarkupStrategy   – margin range, max_price cap, multiple auctions, fixed markup
  TestFeasibleRandomStrategy – no winner, winner present, infeasible skip, stochastic bounds
  TestEdgeCases              – unknown strategy fallback, failed tx, tx exception recovery
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from agents.config import Config
from agents.infrastructure.cost_tracker import CostTracker
from agents.provider_agent.blockchain_handler import AgentState, BlockchainHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_config(
    strategy: str = "random_markup",
    min_margin: float = 0.10,
    max_margin: float = 0.30,
    base_cost: int = 40,          # integer USDC (handler converts ×1e6 internally)
) -> Config:
    """Build a real Config and override only the heuristic-relevant fields."""
    cfg = Config()
    cfg.heuristic_strategy  = strategy
    cfg.heuristic_min_margin = min_margin
    cfg.heuristic_max_margin = max_margin
    cfg.bidding_base_cost   = base_cost
    return cfg


def _make_mock_client() -> MagicMock:
    """Async-capable mock client that simulates a successful on-chain bid."""
    client = MagicMock()
    client.estimate_gas         = AsyncMock(return_value=200_000)
    client.send_transaction      = AsyncMock(return_value="0xdeadbeef")
    client.wait_for_transaction  = AsyncMock(return_value={
        "status": 1,
        "blockNumber": 42,
        "gasUsed": 150_000,
        "effectiveGasPrice": 2_000_000_000,   # 2 gwei
    })
    return client


def _make_handler(cfg: Config, client: MagicMock) -> BlockchainHandler:
    """
    Construct a BlockchainHandler with the mock client, then replace
    handler.config with our test Config so heuristic fields are used.
    """
    handler = BlockchainHandler(
        agent_id=1,
        blockchain_client=client,
        reasoning_mode="heuristic",
    )
    # Override config after construction – _heuristic_reasoning reads only
    # self.config.{heuristic_strategy,heuristic_min/max_margin,bidding_base_cost}
    handler.config = cfg
    return handler


def _auction(
    auction_id: int = 1,
    max_price_usdc: float = 200.0,
    winning_bid: int = 0,
    winning_agent_id: int = 0,
) -> dict:
    return {
        "auction_id":              auction_id,
        "buyer":                   "0xBuyer",
        "service_description_cid": "QmTest",
        "max_price":               int(max_price_usdc * 1e6),
        "duration":                3600,
        "start_time":              1_000_000,
        "end_time":                1_003_600,
        "time_remaining":          3600,
        "winning_agent_id":        winning_agent_id,
        "winning_bid":             winning_bid,
        "is_active":               True,
        "reputation_weight":       50,
        "service_requirements":    {},
    }


def _base_state(
    auctions=None,
    agent_rep: int = 50,
    competitors=None,
) -> AgentState:
    """Minimal AgentState for heuristic reasoning tests."""
    return {
        "agent_id":                1,
        "action":                  "monitor",
        "auction_id":              None,
        "client_address":          None,
        "feedback_auth":           None,
        "won_auctions":            [],
        "eligible_active_auctions": auctions or [],
        "competitors_reputation":  competitors or [],
        "agent_reputation":        {"rating": agent_rep, "feedback_count": 0},
        "estimated_service_cost":  40.0,
        "total_llm_costs":         0.0,
        "total_gas_costs":         0.0,
        "total_revenue":           0.0,
        "bids_placed":             [],
        "tx_hash":                 None,
        "error":                   None,
        "messages":                [],
    }


# ---------------------------------------------------------------------------
# Test: random_markup strategy
# ---------------------------------------------------------------------------

class TestRandomMarkupStrategy(unittest.TestCase):
    """random_markup: bid = int(cost × Uniform(1+min_margin, 1+max_margin))"""

    BASE_COST_USDC = 40  # integer USDC; handler multiplies by 1e6

    def _run(self, state, cfg=None):
        cfg = cfg or _make_config(strategy="random_markup", min_margin=0.10, max_margin=0.30)
        client = _make_mock_client()
        handler = _make_handler(cfg, client)
        return run_async(handler._heuristic_reasoning(state)), handler, cfg

    # ------------------------------------------------------------------
    def test_bid_within_margin_range(self):
        """Bid must fall in [cost×1.10, cost×1.30]."""
        state = _base_state(auctions=[_auction(max_price_usdc=200.0)])
        result, _, _ = self._run(state)

        self.assertIsNone(result["error"])
        self.assertEqual(len(result["bids_placed"]), 1)

        bid      = result["bids_placed"][0]["bid_amount"]
        cost     = int(self.BASE_COST_USDC * 1e6)
        min_bid  = int(cost * 1.10)
        max_bid  = int(cost * 1.30)

        self.assertGreaterEqual(bid, min_bid, f"bid {bid} < min_bid {min_bid}")
        self.assertLessEqual   (bid, max_bid, f"bid {bid} > max_bid {max_bid}")

    # ------------------------------------------------------------------
    def test_bid_skipped_when_exceeds_max_price(self):
        """
        With base_cost=40 USDC and 10-30% markup the bid is always 44-52 USDC,
        so a max_price of 41 USDC must cause the auction to be skipped.
        """
        cfg = _make_config(strategy="random_markup", min_margin=0.10, max_margin=0.30, base_cost=40)
        state = _base_state(auctions=[_auction(max_price_usdc=41.0)])
        result, handler, _ = self._run(state, cfg=cfg)

        self.assertEqual(len(result["bids_placed"]), 0)
        handler.client.send_transaction.assert_not_called()

    # ------------------------------------------------------------------
    def test_no_eligible_auctions(self):
        """Empty auction list produces no bids and no error."""
        state = _base_state(auctions=[])
        result, handler, _ = self._run(state)

        self.assertIsNone(result["error"])
        self.assertEqual(result["bids_placed"], [])
        handler.client.send_transaction.assert_not_called()

    # ------------------------------------------------------------------
    def test_multiple_auctions_all_bid(self):
        """All three eligible auctions within budget receive a bid."""
        auctions = [_auction(auction_id=i, max_price_usdc=200.0) for i in range(1, 4)]
        state = _base_state(auctions=auctions)
        result, _, _ = self._run(state)

        self.assertEqual(len(result["bids_placed"]), 3)

    # ------------------------------------------------------------------
    def test_zero_margin_range_gives_fixed_bid(self):
        """min_margin == max_margin → single deterministic bid amount."""
        cfg = _make_config(strategy="random_markup", min_margin=0.20, max_margin=0.20, base_cost=40)
        state = _base_state(auctions=[_auction(max_price_usdc=200.0)])
        result, _, _ = self._run(state, cfg=cfg)

        self.assertEqual(len(result["bids_placed"]), 1)
        expected = int(int(40 * 1e6) * 1.20)  # = 48_000_000
        self.assertEqual(result["bids_placed"][0]["bid_amount"], expected)


# ---------------------------------------------------------------------------
# Test: feasible_random strategy
# ---------------------------------------------------------------------------

class TestFeasibleRandomStrategy(unittest.TestCase):
    """feasible_random: bid ~ Uniform[min_profitable_bid, max_competitive_bid]"""

    def _run(self, state, cfg=None):
        cfg = cfg or _make_config(strategy="feasible_random", min_margin=0.10, max_margin=0.30)
        client = _make_mock_client()
        handler = _make_handler(cfg, client)
        return run_async(handler._heuristic_reasoning(state)), handler, cfg

    # ------------------------------------------------------------------
    def test_no_current_winner_bids_between_profitable_and_max_price(self):
        """
        With no winner, max_competitive = max_price.
        Bid must be in [min_profitable, max_price].

        min_profitable = int(40e6 × 1.10) = 44_000_000
        max_price      = 200_000_000
        """
        base_cost      = int(40 * 1e6)
        max_price      = int(200 * 1e6)
        state = _base_state(auctions=[_auction(max_price_usdc=200.0, winning_bid=0)])
        result, _, cfg = self._run(state)

        self.assertEqual(len(result["bids_placed"]), 1)
        bid          = result["bids_placed"][0]["bid_amount"]
        min_profitable = int(base_cost * (1.0 + cfg.heuristic_min_margin))

        self.assertGreaterEqual(bid, min_profitable)
        self.assertLessEqual   (bid, max_price)

    # ------------------------------------------------------------------
    def test_bid_beats_current_winner_score(self):
        """
        With winner_bid=80 USDC and winner_rep=50:
          winner_score      = (80e6 × 150) // 100 = 120_000_000
          max_competitive   = (120e6×100 - 1) // 150 = 79_999_999
        Any bid drawn from [min_profitable, 79_999_999] must produce a
        score strictly below winner_score.
        """
        agent_rep    = 50
        winner_rep   = 50
        winner_bid   = int(80 * 1e6)
        winner_score = (winner_bid * (100 + winner_rep)) // 100   # 120_000_000

        state = _base_state(
            auctions    = [_auction(max_price_usdc=200.0,
                                    winning_bid=winner_bid, winning_agent_id=99)],
            agent_rep   = agent_rep,
            competitors = [{"agent_id": 99, "rating": winner_rep, "feedback_count": 0}],
        )
        result, _, _ = self._run(state)

        self.assertEqual(len(result["bids_placed"]), 1)
        bid       = result["bids_placed"][0]["bid_amount"]
        our_score = (bid * (100 + agent_rep)) // 100
        self.assertLess(our_score, winner_score,
                        f"our_score {our_score} should be < winner_score {winner_score}")

    # ------------------------------------------------------------------
    def test_infeasible_region_skips_auction(self):
        """
        winner_bid=30 USDC, winner_rep=50:
          winner_score    = (30e6 × 150) // 100 = 45_000_000
          max_competitive = (45e6×100 - 1) // 150 = 29_999_999
          min_profitable  = int(40e6 × 1.10)      = 44_000_000
        max_competitive (29.99 USDC) < min_profitable (44 USDC) → skip.
        """
        winner_bid = int(30 * 1e6)

        state = _base_state(
            auctions    = [_auction(max_price_usdc=200.0,
                                    winning_bid=winner_bid, winning_agent_id=99)],
            agent_rep   = 50,
            competitors = [{"agent_id": 99, "rating": 50, "feedback_count": 0}],
        )
        result, handler, _ = self._run(state)

        self.assertEqual(len(result["bids_placed"]), 0)
        handler.client.send_transaction.assert_not_called()

    # ------------------------------------------------------------------
    def test_no_eligible_auctions(self):
        """Empty auction list produces no bids."""
        state = _base_state(auctions=[])
        result, handler, _ = self._run(state)

        self.assertEqual(result["bids_placed"], [])
        handler.client.send_transaction.assert_not_called()

    # ------------------------------------------------------------------
    def test_stochastic_bids_always_within_bounds(self):
        """
        Run 20 independent trials.  Every bid must satisfy:
          min_profitable (44 USDC) ≤ bid ≤ max_price (200 USDC).
        """
        base_cost      = int(40 * 1e6)
        cfg            = _make_config(strategy="feasible_random", min_margin=0.10, max_margin=0.30)
        min_profitable = int(base_cost * (1.0 + cfg.heuristic_min_margin))  # 44_000_000
        max_price      = int(200 * 1e6)

        for trial in range(20):
            client  = _make_mock_client()
            handler = _make_handler(cfg, client)
            state   = _base_state(auctions=[_auction(max_price_usdc=200.0, winning_bid=0)])
            result  = run_async(handler._heuristic_reasoning(state))

            self.assertEqual(len(result["bids_placed"]), 1,
                             f"Trial {trial}: expected 1 bid")
            bid = result["bids_placed"][0]["bid_amount"]
            self.assertGreaterEqual(bid, min_profitable,
                                    f"Trial {trial}: bid {bid} < min_profitable {min_profitable}")
            self.assertLessEqual(bid, max_price,
                                 f"Trial {trial}: bid {bid} > max_price {max_price}")


# ---------------------------------------------------------------------------
# Test: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    # ------------------------------------------------------------------
    def test_unknown_strategy_falls_back_to_random_markup(self):
        """
        An unrecognised strategy name should fall back to random_markup
        logic and still produce a profitable bid.
        """
        cfg    = _make_config(strategy="nonexistent_strategy",
                              min_margin=0.10, max_margin=0.30, base_cost=40)
        client = _make_mock_client()
        state  = _base_state(auctions=[_auction(max_price_usdc=200.0)])
        result = run_async(_make_handler(cfg, client)._heuristic_reasoning(state))

        self.assertEqual(len(result["bids_placed"]), 1)
        bid  = result["bids_placed"][0]["bid_amount"]
        cost = int(cfg.bidding_base_cost * 1e6)
        self.assertGreater(bid, cost, "Fallback bid should be above raw cost")

    # ------------------------------------------------------------------
    def test_failed_transaction_not_added_to_bids_placed(self):
        """If the on-chain tx returns status=0 it must NOT appear in bids_placed."""
        cfg    = _make_config(strategy="random_markup")
        client = _make_mock_client()
        client.wait_for_transaction = AsyncMock(return_value={
            "status": 0,        # ← failure
            "blockNumber": 99,
            "gasUsed": 50_000,
            "effectiveGasPrice": 2_000_000_000,
        })
        state  = _base_state(auctions=[_auction(max_price_usdc=200.0)])
        result = run_async(_make_handler(cfg, client)._heuristic_reasoning(state))

        self.assertEqual(len(result["bids_placed"]), 0)

    # ------------------------------------------------------------------
    def test_tx_exception_continues_to_next_auction(self):
        """
        A network exception while bidding on auction 1 must not prevent
        a successful bid on auction 2.
        """
        cfg    = _make_config(strategy="random_markup")
        client = _make_mock_client()

        _call = {"n": 0}

        async def flaky_send(*args, **kwargs):
            _call["n"] += 1
            if _call["n"] == 1:
                raise Exception("Simulated network error on auction 1")
            return "0xdeadbeef"

        client.send_transaction = AsyncMock(side_effect=flaky_send)

        auctions = [
            _auction(auction_id=1, max_price_usdc=200.0),
            _auction(auction_id=2, max_price_usdc=200.0),
        ]
        state  = _base_state(auctions=auctions)
        result = run_async(_make_handler(cfg, client)._heuristic_reasoning(state))

        self.assertEqual(len(result["bids_placed"]), 1,
                         "Should have 1 successful bid (auction 2)")
        self.assertEqual(result["bids_placed"][0]["auction_id"], 2)

    # ------------------------------------------------------------------
    def test_bids_placed_contains_expected_keys(self):
        """Each bid record must contain success, tx_hash, auction_id, bid_amount, block_number."""
        cfg    = _make_config(strategy="random_markup")
        client = _make_mock_client()
        state  = _base_state(auctions=[_auction(max_price_usdc=200.0)])
        result = run_async(_make_handler(cfg, client)._heuristic_reasoning(state))

        self.assertEqual(len(result["bids_placed"]), 1)
        rec = result["bids_placed"][0]
        for key in ("success", "tx_hash", "auction_id", "bid_amount", "block_number"):
            self.assertIn(key, rec, f"Missing key '{key}' in bid record")

        self.assertTrue(rec["success"])
        self.assertEqual(rec["tx_hash"], "0xdeadbeef")
        self.assertEqual(rec["block_number"], 42)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
