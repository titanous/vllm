# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for GPT-OSS structural tag support in reasoning (PR #25515)."""

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.tool_server import ToolServer
from vllm.reasoning.gptoss_reasoning_parser import (
    GptOssReasoningParser,
    from_builtin_tool_to_tag,
    no_func_reaonsing_tag,
    tag_with_builtin_funcs,
)


class TestGptOssReasoningParser:
    """Test cases for GptOssReasoningParser structural tag functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        return tokenizer

    @pytest.fixture
    def reasoning_parser(self, mock_tokenizer):
        """Create a GptOssReasoningParser instance."""
        return GptOssReasoningParser(mock_tokenizer)

    @pytest.fixture
    def mock_tool_server_empty(self):
        """Create a mock ToolServer with no tools."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(return_value=False)
        return tool_server

    @pytest.fixture
    def mock_tool_server_with_browser(self):
        """Create a mock ToolServer with browser tool."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(side_effect=lambda tool: tool == "browser")
        return tool_server

    @pytest.fixture
    def mock_tool_server_with_all_tools(self):
        """Create a mock ToolServer with all builtin tools."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(
            side_effect=lambda tool: tool in ["browser", "python", "container"]
        )
        return tool_server

    def test_prepare_structured_tag_no_tool_server(self, reasoning_parser):
        """Test prepare_structured_tag with no tool server."""
        result = reasoning_parser.prepare_structured_tag(None, None)
        expected = json.dumps(no_func_reaonsing_tag)

        assert result == expected

        # Verify the structure is correct
        parsed = json.loads(result)
        assert parsed["type"] == "structural_tag"
        assert parsed["format"]["type"] == "triggered_tags"
        assert len(parsed["format"]["tags"]) == 1
        assert parsed["format"]["tags"][0]["begin"] == "<|channel|>analysis<|message|>"
        assert parsed["format"]["triggers"] == ["<|channel|>analysis"]

    def test_prepare_structured_tag_with_all_tools(
        self, reasoning_parser, mock_tool_server_with_all_tools
    ):
        """Test prepare_structured_tag with all builtin tools."""
        result = reasoning_parser.prepare_structured_tag(
            None, mock_tool_server_with_all_tools
        )
        parsed = json.loads(result)

        # Browser has 3 functions × 2 channels = 6 tags
        # Python has no functions × 2 channels = 2 tags
        # Container has no functions × 2 channels = 2 tags
        # Plus 1 analysis tag = 11 total
        assert len(parsed["format"]["tags"]) == 11

        # Check that tool-specific tags use correct Harmony format
        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]
        # Browser should have function-specific tags (browser.search, browser.open, browser.find)
        assert any("to=browser.search" in begin for begin in tag_begins)
        assert any("to=browser.open" in begin for begin in tag_begins)
        assert any("to=browser.find" in begin for begin in tag_begins)
        # Python and container should have generic tags
        assert any("to=python<|channel|>" in begin for begin in tag_begins)
        assert any("to=container<|channel|>" in begin for begin in tag_begins)

    def test_prepare_structured_tag_with_original_tag(self, reasoning_parser):
        """Test prepare_structured_tag when original_tag is provided."""
        original_tag = '{"custom": "tag"}'
        result = reasoning_parser.prepare_structured_tag(original_tag, None)

        # Should return the original tag unchanged
        assert result == original_tag

    def test_from_builtin_tool_to_tag(self):
        """Test from_builtin_tool_to_tag function."""
        # Python has no functions, so it should have 2 generic tags (commentary + analysis)
        python_tags = from_builtin_tool_to_tag("python")
        assert len(python_tags) == 2
        assert python_tags[0]["begin"] == "<|start|>assistant to=python<|channel|>commentary<|message|>"
        assert python_tags[0]["content"]["type"] == "any_text"
        assert python_tags[0]["end"] == "<|call|>"
        assert python_tags[1]["begin"] == "<|start|>assistant to=python<|channel|>analysis<|message|>"
        assert python_tags[1]["end"] == "<|call|>"

        # Browser has 3 functions, so it should have 6 tags (3 functions × 2 channels)
        browser_tags = from_builtin_tool_to_tag("browser")
        assert len(browser_tags) == 6
        # Check that browser tags include function names and JSON schemas
        tag_begins = [tag["begin"] for tag in browser_tags]
        assert any("to=browser.search" in begin for begin in tag_begins)
        assert any("to=browser.open" in begin for begin in tag_begins)
        assert any("to=browser.find" in begin for begin in tag_begins)
        # Verify schemas are present (not any_text)
        for tag in browser_tags:
            assert tag["content"]["type"] == "json_schema"
            assert "json_schema" in tag["content"]

    def test_tag_with_builtin_funcs(self):
        """Test tag_with_builtin_funcs function."""
        builtin_tools = ["browser", "python"]
        result = tag_with_builtin_funcs(no_func_reaonsing_tag, builtin_tools)

        assert result["type"] == "structural_tag"
        # Browser: 3 functions × 2 channels = 6 tags
        # Python: 0 functions × 2 channels = 2 tags
        # Plus original analysis tag = 9 total
        assert len(result["format"]["tags"]) == 9

        # Should have added tool call trigger with correct Harmony format
        assert "<|start|>assistant to=" in result["format"]["triggers"]
        assert "<|channel|>analysis" in result["format"]["triggers"]

    def test_tag_structure_invariants(self):
        """Test that the basic tag structure follows expected format."""
        # Test the base no_func_reaonsing_tag structure
        assert no_func_reaonsing_tag["type"] == "structural_tag"
        assert no_func_reaonsing_tag["format"]["type"] == "triggered_tags"
        assert no_func_reaonsing_tag["format"]["stop_after_first"] is False

        # Verify analysis tag structure
        analysis_tag = no_func_reaonsing_tag["format"]["tags"][0]
        assert analysis_tag["begin"] == "<|channel|>analysis<|message|>"
        assert analysis_tag["content"]["type"] == "any_text"
        assert analysis_tag["end"] == "<|end|>"

    def test_json_serialization_valid(
        self, reasoning_parser, mock_tool_server_with_all_tools
    ):
        """Test that all generated tags produce valid JSON."""
        # Test with no tool server
        result1 = reasoning_parser.prepare_structured_tag(None, None)
        json.loads(result1)  # Should not raise

        # Test with empty tool server
        empty_server = Mock(spec=ToolServer)
        empty_server.has_tool = Mock(return_value=False)
        result2 = reasoning_parser.prepare_structured_tag(None, empty_server)
        json.loads(result2)  # Should not raise

        # Test with tools
        result3 = reasoning_parser.prepare_structured_tag(
            None, mock_tool_server_with_all_tools
        )
        json.loads(result3)  # Should not raise

    @pytest.mark.parametrize("tool_name", ["browser", "python", "container"])
    def test_single_tool_integration(self, reasoning_parser, tool_name):
        """Test integration with individual tools."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(side_effect=lambda tool: tool == tool_name)

        result = reasoning_parser.prepare_structured_tag(None, tool_server)
        parsed = json.loads(result)

        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]

        if tool_name == "browser":
            # Browser has 3 functions × 2 channels + 1 analysis = 7 tags
            assert len(parsed["format"]["tags"]) == 7
            # Check for function-specific tags
            assert any("to=browser.search" in begin for begin in tag_begins)
            assert any("to=browser.open" in begin for begin in tag_begins)
            assert any("to=browser.find" in begin for begin in tag_begins)
        else:
            # Python/container: 2 tags (commentary + analysis) + 1 analysis = 3 tags
            assert len(parsed["format"]["tags"]) == 3
            assert any(f"to={tool_name}<|channel|>commentary" in begin for begin in tag_begins)
            assert any(f"to={tool_name}<|channel|>analysis" in begin for begin in tag_begins)
