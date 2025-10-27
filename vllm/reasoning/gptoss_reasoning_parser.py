# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.harmony_utils import parse_chat_output
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.entrypoints.tool_server import ToolServer
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParser, ReasoningParserManager

logger = init_logger(__name__)

no_func_reaonsing_tag = {
    "type": "structural_tag",
    "format": {
        "type": "triggered_tags",
        "tags": [
            # Analysis channel: reasoning/thinking
            {
                "begin": "<|start|>assistant<|channel|>analysis<|message|>",
                "content": {"type": "any_text"},
                "end": "<|end|>",
            },
            # Commentary channel: internal thoughts
            {
                "begin": "<|start|>assistant<|channel|>commentary<|message|>",
                "content": {"type": "any_text"},
                "end": "<|end|>",
            },
            # Final channel: output to user
            {
                "begin": "<|start|>assistant<|channel|>final<|message|>",
                "content": {"type": "any_text"},
                "end": "<|end|>",
            },
        ],
        "triggers": ["<|start|>assistant"],
        "stop_after_first": False,
    },
}


def create_response_schema_tag(response_schema: dict | str) -> dict:
    """Create structural tag that ONLY allows final output with JSON schema.

    Used when user requests response_format with json_schema.
    Disables tool calls and reasoning - forces direct JSON output.

    Args:
        response_schema: JSON schema as dict or JSON string

    Returns:
        Structural tag dict that restricts to final channel with schema
    """
    schema_dict = (
        json.loads(response_schema)
        if isinstance(response_schema, str)
        else response_schema
    )

    return {
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "tags": [
                {
                    "begin": "<|start|>assistant<|channel|>final json<|message|>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": schema_dict
                    },
                    "end": "<|end|>",
                },
            ],
            "triggers": ["<|start|>assistant"],
            "stop_after_first": True,
        },
    }


def from_builtin_tool_to_tag(tool: str) -> list[dict]:
    """Generate structural tags for a builtin tool with proper JSON schemas.

    Args:
        tool: Tool name (e.g., "browser", "python")

    Returns:
        List of tag dictionaries with correct Harmony format patterns and schemas
    """
    from openai_harmony import ToolNamespaceConfig

    # Get tool configuration with schemas
    if tool == "browser":
        config = ToolNamespaceConfig.browser()
    elif tool == "python":
        config = ToolNamespaceConfig.python()
    else:
        config = None

    tags = []
    channels = ["commentary", "analysis"]  # Support both channels for tool calls

    if config and config.tools:
        # For each function in the tool namespace, create tags with JSON schemas
        for func in config.tools:
            for channel in channels:
                tags.append({
                    "begin": f"<|start|>assistant to={tool}.{func.name}<|channel|>{channel} json<|message|>",
                    "content": {
                        "type": "json_schema",
                        "json_schema": func.parameters
                    },
                    "end": "<|call|>"
                })
    else:
        # Python or tools without function schemas - accept any text
        for channel in channels:
            tags.append({
                "begin": f"<|start|>assistant to={tool}<|channel|>{channel} json<|message|>",
                "content": {"type": "any_text"},
                "end": "<|call|>"
            })

    return tags


def from_custom_function_to_tag(tool) -> list[dict]:
    """Generate structural tags for a custom function tool.

    Args:
        tool: Custom function tool with type="function" (Tool or ChatCompletionToolsParam)

    Returns:
        List of tag dictionaries (commentary + analysis channels)
    """
    # Handle both Tool and ChatCompletionToolsParam types
    if hasattr(tool, 'type') and tool.type != "function":
        raise ValueError(f"Expected function tool, got {tool.type}")

    # Extract name and parameters
    if hasattr(tool, 'function'):
        # ChatCompletionToolsParam format
        name = tool.function.name
        parameters = tool.function.parameters
    else:
        # Tool format
        name = tool.name
        parameters = tool.parameters

    tags = []
    channels = ["commentary", "analysis"]

    for channel in channels:
        tags.append({
            "begin": f"<|start|>assistant to=functions.{name}<|channel|>{channel} json<|message|>",
            "content": {
                "type": "json_schema",
                "json_schema": parameters
            },
            "end": "<|call|>"
        })

    return tags


@ReasoningParserManager.register_module("openai_gptoss")
class GptOssReasoningParser(ReasoningParser):
    """
    Reasoning parser for GptOss model.

    The GptOss model uses harmony to extract reasoning content and this parser
    is only used for detecting the end of the reasoning content.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.reasoning_end_token_ids = self.model_tokenizer.encode(
            "<|start|>assistant<|channel|>final<|message|>"
        )

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        end_token_ids = self.reasoning_end_token_ids
        assert len(end_token_ids) > 0, "reasoning_end_token_ids is empty"
        # Check if the end sequence is present in the input_ids.
        # We search from the end of input_ids to find the last match.
        for i in range(len(input_ids) - len(end_token_ids), -1, -1):
            if input_ids[i : i + len(end_token_ids)] == end_token_ids:
                return True
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        _, content, _ = parse_chat_output(input_ids)
        if content is None:
            return []
        return self.model_tokenizer.encode(content)

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        prev_reasoning, prev_content, _ = parse_chat_output(list(previous_token_ids))
        cur_reasoning, cur_content, _ = parse_chat_output(list(current_token_ids))
        reasoning_delta = None
        content_delta = None
        if cur_reasoning is not None:
            prev_r = prev_reasoning or ""
            if cur_reasoning.startswith(prev_r):
                reasoning_delta = cur_reasoning[len(prev_r) :] or None
            else:
                reasoning_delta = cur_reasoning
        if cur_content is not None:
            prev_c = prev_content or ""
            if cur_content.startswith(prev_c):
                content_delta = cur_content[len(prev_c) :] or None
            else:
                content_delta = cur_content
        if reasoning_delta is None and content_delta is None:
            return None
        return DeltaMessage(reasoning_content=reasoning_delta, content=content_delta)

    def extract_reasoning_content(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> tuple[str | None, str | None]:
        raise NotImplementedError(
            "gpt-oss has a special branch for parsing reasoning in non-streaming mode. This method shouldn't be used."  # noqa: E501
        )

    # This function prepares the structural tag to format reasoning output
    def prepare_structured_tag(
        self,
        original_tag: str | None,
        tool_server: ToolServer | None,
        custom_tools: list | None = None,
        response_schema: dict | str | None = None,
    ) -> str:
        """Prepare structural tag for Harmony-based generation.

        Args:
            original_tag: Pre-existing structural tag (if provided, returned as-is)
            tool_server: Server providing builtin tools (browser, python, container)
            custom_tools: List of custom function tools
            response_schema: JSON schema for response_format (disables tools/reasoning)

        Returns:
            JSON-serialized structural tag
        """
        if original_tag is not None:
            return original_tag

        # If response_schema provided, ONLY allow final output with that schema
        # Disables all tool calls and reasoning
        if response_schema is not None:
            logger.info("Response schema provided - restricting to final output only")
            return json.dumps(create_response_schema_tag(response_schema))

        # Check if we have any tools
        has_builtin_tools = tool_server is not None
        has_custom_tools = custom_tools is not None and len(custom_tools) > 0

        # No tools - return base tag with analysis/commentary/final channels
        if not has_builtin_tools and not has_custom_tools:
            return json.dumps(no_func_reaonsing_tag)

        # Build tag with tools
        import copy

        func_tag = copy.deepcopy(no_func_reaonsing_tag)

        # Add builtin tools
        if tool_server is not None:
            builtin_tool_list = []
            if tool_server.has_tool("browser"):
                builtin_tool_list.append("browser")
            if tool_server.has_tool("python"):
                builtin_tool_list.append("python")

            if len(builtin_tool_list) > 0:
                logger.info("Builtin_tool_list: %s", builtin_tool_list)
                for tool in builtin_tool_list:
                    func_tag["format"]["tags"].extend(from_builtin_tool_to_tag(tool))

        # Add custom function tools
        if has_custom_tools:
            logger.info("Adding %d custom function tools", len(custom_tools))
            for tool in custom_tools:
                if hasattr(tool, "type") and tool.type == "function":
                    func_tag["format"]["tags"].extend(from_custom_function_to_tag(tool))

        return json.dumps(func_tag)
