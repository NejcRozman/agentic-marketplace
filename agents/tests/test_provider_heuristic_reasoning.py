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


def _make_mock_client(auctions=None) -> MagicMock:
    """Async-capable mock client that simulates a successful on-chain bid.

    For strict pre-bid refresh, this mock also serves getAuctionDetails/get_block.
    """
    client = MagicMock()
    client.estimate_gas         = AsyncMock(return_value=200_000)
    client.send_transaction      = AsyncMock(return_value="0xdeadbeef")
    client.wait_for_transaction  = AsyncMock(return_value={
        "status": 1,
        "blockNumber": 42,
        "gasUsed": 150_000,
        "effectiveGasPrice": 2_000_000_000,   # 2 gwei
    })

    auction_map = {int(a["auction_id"]): a for a in (auctions or [])}

    async def _call_contract_method(contract_name, method_name, *args):
        if contract_name == "ReverseAuction" and method_name == "getAuctionDetails":
            auction_id = int(args[0])
            if auction_id not in auction_map:
                raise ValueError(f"unknown auction_id in mock: {auction_id}")
            a = auction_map[auction_id]
            # Match the expected tuple layout used by BlockchainHandler.
            return [
                auction_id,                                 # 0 id
                a.get("buyer", "0xBuyer"),                # 1 buyer
                a.get("service_description_cid", "QmTest"),  # 2 cid
                int(a.get("max_price", 0)),               # 3 maxPrice
                int(a.get("duration", 3600)),             # 4 duration
                int(a.get("start_time", 1_000_000)),      # 5 startTime
                [],                                         # 6 eligible ids (unused)
                int(a.get("winning_agent_id", 0) or 0),   # 7 winningAgentId
                int(a.get("winning_bid", 0) or 0),        # 8 winningBid
                bool(a.get("is_active", True)),           # 9 isActive
                False,                                      # 10 isCompleted
                0,                                          # 11 escrowAmount
                int(a.get("reputation_weight", 50) or 50),# 12 reputationWeight
            ]
        raise ValueError(f"unexpected mocked call: {contract_name}.{method_name}")

    client.call_contract_method = AsyncMock(side_effect=_call_contract_method)
    client.get_block = AsyncMock(return_value={"timestamp": 1_000_100})
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
        client = _make_mock_client(auctions=state.get("eligible_active_auctions", []))
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
        client = _make_mock_client(auctions=state.get("eligible_active_auctions", []))
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
          normalized_rep    = 50 * 100 = 5000
          winner_bid_comp   = ((200e6 - 80e6) * 10000) // 200e6 = 6000
          winner_score      = (50*5000 + 50*6000) // 100 = 5500
          max_competitive   = 79_980_000 (strictly better score required)
        Any bid drawn from [min_profitable, 79_980_000] must produce a
        score strictly above winner_score.
        """
        agent_rep    = 50
        winner_rep   = 50
        winner_bid   = int(80 * 1e6)
        max_price    = int(200 * 1e6)
        score_precision = 10_000
        rep_component = winner_rep * (score_precision // 100)
        bid_component = ((max_price - winner_bid) * score_precision) // max_price
        winner_score = (50 * rep_component + 50 * bid_component) // 100

        state = _base_state(
            auctions    = [_auction(max_price_usdc=200.0,
                                    winning_bid=winner_bid, winning_agent_id=99)],
            agent_rep   = agent_rep,
            competitors = [{"agent_id": 99, "rating": winner_rep, "feedback_count": 0}],
        )
        result, _, _ = self._run(state)

        self.assertEqual(len(result["bids_placed"]), 1)
        bid       = result["bids_placed"][0]["bid_amount"]
        our_rep_component = agent_rep * (score_precision // 100)
        our_bid_component = ((max_price - bid) * score_precision) // max_price
        our_score = (50 * our_rep_component + 50 * our_bid_component) // 100
        self.assertGreater(our_score, winner_score,
                           f"our_score {our_score} should be > winner_score {winner_score}")

    # ------------------------------------------------------------------
    def test_infeasible_region_skips_auction(self):
        """
        Force infeasible region with high reputation weight and stronger current winner.
        With rep_weight=80, winner_rep=100, winner_bid=50 USDC:
          winner_score is so high that required bid_component exceeds SCORE_PRECISION,
          so max_competitive becomes -1 and auction must be skipped.
        """
        winner_bid = int(50 * 1e6)
        auction = _auction(max_price_usdc=200.0, winning_bid=winner_bid, winning_agent_id=99)
        auction["reputation_weight"] = 80

        state = _base_state(
            auctions=[auction],
            agent_rep=50,
            competitors=[{"agent_id": 99, "rating": 100, "feedback_count": 0}],
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
            auctions = [_auction(max_price_usdc=200.0, winning_bid=0)]
            client  = _make_mock_client(auctions=auctions)
            handler = _make_handler(cfg, client)
            state   = _base_state(auctions=auctions)
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
        auctions = [_auction(max_price_usdc=200.0)]
        client = _make_mock_client(auctions=auctions)
        state  = _base_state(auctions=auctions)
        result = run_async(_make_handler(cfg, client)._heuristic_reasoning(state))

        self.assertEqual(len(result["bids_placed"]), 1)
        bid  = result["bids_placed"][0]["bid_amount"]
        cost = int(cfg.bidding_base_cost * 1e6)
        self.assertGreater(bid, cost, "Fallback bid should be above raw cost")

    # ------------------------------------------------------------------
    def test_failed_transaction_not_added_to_bids_placed(self):
        """If the on-chain tx returns status=0 it must NOT appear in bids_placed."""
        cfg    = _make_config(strategy="random_markup")
        auctions = [_auction(max_price_usdc=200.0)]
        client = _make_mock_client(auctions=auctions)
        client.wait_for_transaction = AsyncMock(return_value={
            "status": 0,        # ← failure
            "blockNumber": 99,
            "gasUsed": 50_000,
            "effectiveGasPrice": 2_000_000_000,
        })
        state  = _base_state(auctions=auctions)
        result = run_async(_make_handler(cfg, client)._heuristic_reasoning(state))

        self.assertEqual(len(result["bids_placed"]), 0)

    # ------------------------------------------------------------------
    def test_tx_exception_continues_to_next_auction(self):
        """
        A network exception while bidding on auction 1 must not prevent
        a successful bid on auction 2.
        """
        cfg    = _make_config(strategy="random_markup")
        auctions = [
            _auction(auction_id=1, max_price_usdc=200.0),
            _auction(auction_id=2, max_price_usdc=200.0),
        ]
        client = _make_mock_client(auctions=auctions)

        _call = {"n": 0}

        async def flaky_send(*args, **kwargs):
            _call["n"] += 1
            if _call["n"] == 1:
                raise Exception("Simulated network error on auction 1")
            return "0xdeadbeef"

        client.send_transaction = AsyncMock(side_effect=flaky_send)

        state  = _base_state(auctions=auctions)
        result = run_async(_make_handler(cfg, client)._heuristic_reasoning(state))

        self.assertEqual(len(result["bids_placed"]), 1,
                         "Should have 1 successful bid (auction 2)")
        self.assertEqual(result["bids_placed"][0]["auction_id"], 2)

    # ------------------------------------------------------------------
    def test_bids_placed_contains_expected_keys(self):
        """Each bid record must contain success, tx_hash, auction_id, bid_amount, block_number."""
        cfg    = _make_config(strategy="random_markup")
        auctions = [_auction(max_price_usdc=200.0)]
        client = _make_mock_client(auctions=auctions)
        state  = _base_state(auctions=auctions)
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
