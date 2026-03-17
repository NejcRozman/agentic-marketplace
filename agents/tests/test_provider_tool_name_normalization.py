"""Unit tests for provider tool-name normalization logic."""

import unittest

from agents.provider_agent.blockchain_handler import BlockchainHandler


class TestToolNameNormalization(unittest.TestCase):
    def test_passthrough(self):
        self.assertEqual(
            BlockchainHandler._normalize_tool_name("place_bid"),
            "place_bid",
        )

    def test_strip_transport_suffix(self):
        self.assertEqual(
            BlockchainHandler._normalize_tool_name("simulate_bid_outcome<|channel|>commentary"),
            "simulate_bid_outcome",
        )

    def test_strip_quotes_and_spaces(self):
        self.assertEqual(
            BlockchainHandler._normalize_tool_name(" `calculate_bid_score<|channel|>commentary` "),
            "calculate_bid_score",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
