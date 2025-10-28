# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for GPT-OSS structural tag support in reasoning (PR #25515)."""

import json
from unittest.mock import Mock

import pytest

from vllm.entrypoints.tool_server import ToolServer
from vllm.reasoning.gptoss_reasoning_parser import (
    GptOssReasoningParser,
    create_response_schema_tag,
    from_builtin_tool_to_tag,
    from_custom_function_to_tag,
    no_func_reaonsing_tag,
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
            side_effect=lambda tool: tool in ["browser", "python"]
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
        assert len(parsed["format"]["tags"]) == 3

        # Check all three channels are present
        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]
        assert "<|channel|>analysis<|message|>" in tag_begins
        assert "<|channel|>commentary<|message|>" in tag_begins
        assert "<|channel|>final<|message|>" in tag_begins

        assert parsed["format"]["triggers"] == ["<|channel|>", " to="]

    def test_prepare_structured_tag_with_all_tools(
        self, reasoning_parser, mock_tool_server_with_all_tools
    ):
        """Test prepare_structured_tag with all builtin tools."""
        result = reasoning_parser.prepare_structured_tag(
            None, mock_tool_server_with_all_tools
        )
        parsed = json.loads(result)

        # Browser has 3 functions × 2 channels × 2 formats = 12 tags
        # Python has 2 channels × 2 formats = 4 tags
        # Plus 3 base tags (analysis, commentary, final) = 19 total
        assert len(parsed["format"]["tags"]) == 19

        # Check that tool-specific tags use correct Harmony format with json content_type
        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]
        # Browser should have function-specific tags (browser.search, browser.open, browser.find) in both formats
        assert any("to=browser.search<|channel|>" in begin and " json<|message|>" in begin for begin in tag_begins)
        assert any("<|channel|>commentary to=browser.search json<|message|>" in begin for begin in tag_begins)
        assert any("to=browser.open<|channel|>" in begin and " json<|message|>" in begin for begin in tag_begins)
        assert any("<|channel|>commentary to=browser.open json<|message|>" in begin for begin in tag_begins)
        assert any("to=browser.find<|channel|>" in begin and " json<|message|>" in begin for begin in tag_begins)
        assert any("<|channel|>commentary to=browser.find json<|message|>" in begin for begin in tag_begins)
        # Python should have generic tags with json content_type in both formats
        assert any("to=python<|channel|>" in begin and " json<|message|>" in begin for begin in tag_begins)
        assert any("<|channel|>commentary to=python json<|message|>" in begin for begin in tag_begins)

    def test_prepare_structured_tag_with_original_tag(self, reasoning_parser):
        """Test prepare_structured_tag when original_tag is provided."""
        original_tag = '{"custom": "tag"}'
        result = reasoning_parser.prepare_structured_tag(original_tag, None)

        # Should return the original tag unchanged
        assert result == original_tag

    def test_from_builtin_tool_to_tag(self):
        """Test from_builtin_tool_to_tag function."""
        # Python has no functions, so it should have 4 generic tags (2 channels × 2 formats)
        python_tags = from_builtin_tool_to_tag("python")
        assert len(python_tags) == 4
        tag_begins = [tag["begin"] for tag in python_tags]
        # Check both formats for commentary channel
        assert " to=python<|channel|>commentary json<|message|>" in tag_begins
        assert "<|channel|>commentary to=python json<|message|>" in tag_begins
        # Check both formats for analysis channel
        assert " to=python<|channel|>analysis json<|message|>" in tag_begins
        assert "<|channel|>analysis to=python json<|message|>" in tag_begins
        # All should have any_text content
        for tag in python_tags:
            assert tag["content"]["type"] == "any_text"
            assert tag["end"] == "<|call|>"

        # Browser has 3 functions, so it should have 12 tags (3 functions × 2 channels × 2 formats)
        browser_tags = from_builtin_tool_to_tag("browser")
        assert len(browser_tags) == 12
        # Check that browser tags include function names, JSON schemas, and json content_type
        tag_begins = [tag["begin"] for tag in browser_tags]
        # Check both formats exist for search function
        assert any("to=browser.search<|channel|>" in begin and " json<|message|>" in begin for begin in tag_begins)
        assert any("<|channel|>commentary to=browser.search json<|message|>" in begin for begin in tag_begins)
        # Check both formats exist for open function
        assert any("to=browser.open<|channel|>" in begin and " json<|message|>" in begin for begin in tag_begins)
        assert any("<|channel|>commentary to=browser.open json<|message|>" in begin for begin in tag_begins)
        # Check both formats exist for find function
        assert any("to=browser.find<|channel|>" in begin and " json<|message|>" in begin for begin in tag_begins)
        assert any("<|channel|>commentary to=browser.find json<|message|>" in begin for begin in tag_begins)
        # Verify schemas are present (not any_text)
        for tag in browser_tags:
            assert tag["content"]["type"] == "json_schema"
            assert "json_schema" in tag["content"]
            # Verify json content_type is in begin
            assert " json<|message|>" in tag["begin"]

    def test_tag_structure_invariants(self):
        """Test that the basic tag structure follows expected format."""
        # Test the base no_func_reaonsing_tag structure
        assert no_func_reaonsing_tag["type"] == "structural_tag"
        assert no_func_reaonsing_tag["format"]["type"] == "triggered_tags"
        assert no_func_reaonsing_tag["format"]["stop_after_first"] is False

        # Should have 3 tags: analysis, commentary, final
        assert len(no_func_reaonsing_tag["format"]["tags"]) == 3

        # Verify all tags have correct structure
        for tag in no_func_reaonsing_tag["format"]["tags"]:
            assert tag["begin"].startswith("<|channel|>")
            assert tag["content"]["type"] == "any_text"
            assert tag["end"] == "<|end|>"

        # Verify trigger is correct
        assert no_func_reaonsing_tag["format"]["triggers"] == ["<|channel|>", " to="]

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

    @pytest.mark.parametrize("tool_name", ["browser", "python"])
    def test_single_tool_integration(self, reasoning_parser, tool_name):
        """Test integration with individual tools."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(side_effect=lambda tool: tool == tool_name)

        result = reasoning_parser.prepare_structured_tag(None, tool_server)
        parsed = json.loads(result)

        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]

        if tool_name == "browser":
            # Browser has 3 functions × 2 channels × 2 formats + 3 base = 15 tags
            assert len(parsed["format"]["tags"]) == 15
            # Check for function-specific tags with json content_type (both formats)
            assert any("to=browser.search" in begin and " json<|message|>" in begin for begin in tag_begins)
            assert any("<|channel|>commentary to=browser.search json<|message|>" in begin for begin in tag_begins)
            assert any("to=browser.open" in begin and " json<|message|>" in begin for begin in tag_begins)
            assert any("<|channel|>commentary to=browser.open json<|message|>" in begin for begin in tag_begins)
            assert any("to=browser.find" in begin and " json<|message|>" in begin for begin in tag_begins)
            assert any("<|channel|>commentary to=browser.find json<|message|>" in begin for begin in tag_begins)
        else:
            # Python: 2 channels × 2 formats + 3 base = 7 tags
            assert len(parsed["format"]["tags"]) == 7
            assert any(f"to={tool_name}<|channel|>commentary json<|message|>" in begin for begin in tag_begins)
            assert any(f"<|channel|>commentary to={tool_name} json<|message|>" in begin for begin in tag_begins)
            assert any(f"to={tool_name}<|channel|>analysis json<|message|>" in begin for begin in tag_begins)
            assert any(f"<|channel|>analysis to={tool_name} json<|message|>" in begin for begin in tag_begins)

    def test_from_custom_function_to_tag(self):
        """Test from_custom_function_to_tag function."""
        # Create a mock custom function tool
        mock_tool = Mock()
        mock_tool.type = "function"
        mock_tool.name = "get_weather"
        mock_tool.parameters = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }

        tags = from_custom_function_to_tag(mock_tool)

        # Should have 2 tags (commentary channel only, both formats)
        assert len(tags) == 2

        tag_begins = [tag["begin"] for tag in tags]
        # Check format 1: to= before <|channel|>
        assert " to=functions.get_weather<|channel|>commentary json<|message|>" in tag_begins
        # Check format 2: <|channel|> before to=
        assert "<|channel|>commentary to=functions.get_weather json<|message|>" in tag_begins

        # All tags should have json_schema content and correct schema
        for tag in tags:
            assert tag["content"]["type"] == "json_schema"
            assert tag["content"]["json_schema"] == mock_tool.parameters
            assert tag["end"] == "<|call|>"

        # Verify no analysis channel (custom functions use commentary only)
        assert not any("analysis" in begin for begin in tag_begins)

    def test_prepare_structured_tag_with_custom_functions(self, reasoning_parser):
        """Test prepare_structured_tag with custom function tools."""
        # Create mock custom function tools
        mock_tool1 = Mock()
        mock_tool1.type = "function"
        mock_tool1.name = "get_weather"
        mock_tool1.parameters = {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}

        mock_tool2 = Mock()
        mock_tool2.type = "function"
        mock_tool2.name = "calculate"
        mock_tool2.parameters = {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}

        custom_tools = [mock_tool1, mock_tool2]

        result = reasoning_parser.prepare_structured_tag(None, None, custom_tools=custom_tools)
        parsed = json.loads(result)

        # Should have: 3 base + 2 functions × 1 channel × 2 formats = 7 tags
        assert len(parsed["format"]["tags"]) == 7

        # Check trigger
        assert parsed["format"]["triggers"] == ["<|channel|>", " to="]

        # Check that function tags are present with json content_type (both formats)
        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]
        assert any("to=functions.get_weather" in begin and " json<|message|>" in begin for begin in tag_begins)
        assert any("to=functions.calculate" in begin and " json<|message|>" in begin for begin in tag_begins)
        # Verify both formats exist
        assert any("<|channel|>commentary to=functions.get_weather json<|message|>" in begin for begin in tag_begins)
        assert any("<|channel|>commentary to=functions.calculate json<|message|>" in begin for begin in tag_begins)

    def test_prepare_structured_tag_with_builtin_and_custom(self, reasoning_parser):
        """Test prepare_structured_tag with both builtin and custom tools."""
        # Setup builtin tools
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(side_effect=lambda tool: tool == "python")

        # Setup custom tools
        mock_tool = Mock()
        mock_tool.type = "function"
        mock_tool.name = "custom_func"
        mock_tool.parameters = {"type": "object", "properties": {}}

        result = reasoning_parser.prepare_structured_tag(None, tool_server, custom_tools=[mock_tool])
        parsed = json.loads(result)

        # Should have: 3 base + 4 python (2 channels × 2 formats) + 2 custom (1 channel × 2 formats) = 9 tags
        assert len(parsed["format"]["tags"]) == 9

        # Check trigger
        assert parsed["format"]["triggers"] == ["<|channel|>", " to="]

        # Check tags with json content_type (both formats)
        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]
        assert any("to=python" in begin and " json<|message|>" in begin for begin in tag_begins)
        assert any("to=functions.custom_func" in begin and " json<|message|>" in begin for begin in tag_begins)
        # Verify both formats exist
        assert any("<|channel|>commentary to=python json<|message|>" in begin for begin in tag_begins)
        assert any("<|channel|>commentary to=functions.custom_func json<|message|>" in begin for begin in tag_begins)

    def test_response_schema_restricts_to_final_only(self, reasoning_parser):
        """Test that response_schema disables tools and reasoning."""
        tool_server = Mock(spec=ToolServer)
        tool_server.has_tool = Mock(side_effect=lambda tool: tool == "browser")

        response_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"]
        }

        result = reasoning_parser.prepare_structured_tag(
            None, tool_server, custom_tools=None, response_schema=response_schema
        )
        parsed = json.loads(result)

        # Should have ONLY final channel tag with JSON schema
        assert len(parsed["format"]["tags"]) == 1
        assert parsed["format"]["tags"][0]["begin"] == "<|channel|>final json<|message|>"
        assert parsed["format"]["tags"][0]["content"]["type"] == "json_schema"
        assert parsed["format"]["tags"][0]["content"]["json_schema"] == response_schema
        assert parsed["format"]["stop_after_first"] is True

        # No tool call tags should be present
        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]
        assert not any("to=browser" in begin for begin in tag_begins)

    def test_response_schema_as_string(self, reasoning_parser):
        """Test that response_schema works when passed as JSON string."""
        response_schema_str = '{"type": "object", "properties": {"result": {"type": "number"}}}'

        result = reasoning_parser.prepare_structured_tag(
            None, None, custom_tools=None, response_schema=response_schema_str
        )
        parsed = json.loads(result)

        assert len(parsed["format"]["tags"]) == 1
        assert parsed["format"]["tags"][0]["content"]["type"] == "json_schema"
        # Verify the schema was parsed correctly
        assert parsed["format"]["tags"][0]["content"]["json_schema"]["properties"]["result"]["type"] == "number"

    def test_tool_calls_include_json_content_type(self):
        """Test that all tool call patterns include json content_type."""
        browser_tags = from_builtin_tool_to_tag("browser")

        for tag in browser_tags:
            # All tool calls must have ' json<|message|>' in begin
            assert " json<|message|>" in tag["begin"]
            # Should start with either " to=" or "<|channel|>"
            assert tag["begin"].startswith(" to=") or tag["begin"].startswith("<|channel|>")
            assert "<|channel|>" in tag["begin"]
            # Should have either format 1 (to= before <|channel|>) or format 2 (<|channel|> before to=)
            assert ("to=" in tag["begin"]) and (("<|channel|>" in tag["begin"]))

    def test_commentary_channel_supported(self, reasoning_parser):
        """Test that commentary channel is included in base tags."""
        result = reasoning_parser.prepare_structured_tag(None, None)
        parsed = json.loads(result)

        tag_begins = [tag["begin"] for tag in parsed["format"]["tags"]]
        assert any("<|channel|>commentary<|message|>" in begin for begin in tag_begins)
