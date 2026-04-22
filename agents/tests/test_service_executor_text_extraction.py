"""Unit tests for ServiceExecutor response text extraction."""

import unittest

from agents.provider_agent.service_executor import ServiceExecutor


class TestServiceExecutorTextExtraction(unittest.TestCase):
    """Validate extraction across model/provider content formats."""

    def test_plain_string_content(self):
        content = "Direct answer text"
        extracted = ServiceExecutor._extract_response_text(content)
        self.assertEqual(extracted, "Direct answer text")

    def test_list_content_collects_all_text_blocks(self):
        content = [
            {"type": "text", "text": ""},
            {"type": "text", "text": "First chunk."},
            {"type": "text", "text": "Second chunk."},
        ]
        extracted = ServiceExecutor._extract_response_text(content)
        self.assertEqual(extracted, "First chunk.\nSecond chunk.")

    def test_nested_dict_content_is_extracted(self):
        content = {
            "message": {
                "content": [
                    {"type": "text", "text": "Nested answer."}
                ]
            }
        }
        extracted = ServiceExecutor._extract_response_text(content)
        self.assertEqual(extracted, "Nested answer.")

    def test_non_text_blocks_return_empty_string(self):
        content = [
            {"type": "tool_call", "name": "search_literature", "args": {"query": "x"}},
            {"type": "metadata", "value": 123},
        ]
        extracted = ServiceExecutor._extract_response_text(content)
        self.assertEqual(extracted, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
