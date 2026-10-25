"""tool_choice and response_format reach the model as an instruction for the
turn (DeepSeek web has no such switches); they used to be dropped silently."""

import asyncio
import json

from functions import request_directive
from plugin_helper import build_prompt, turn_directive


def test_tool_choice_dialects():
    assert turn_directive("auto") is None
    assert turn_directive(None) is None
    for choice in ("required", {"type": "any"}):
        assert "MUST call at least one tool" in turn_directive(choice)
    for choice in ({"type": "function", "function": {"name": "get_weather"}},  # chat completions
                   {"type": "function", "name": "get_weather"},  # responses
                   {"type": "tool", "name": "get_weather"}):  # anthropic
        assert "MUST call the tool get_weather" in turn_directive(choice)
    for choice in ("none", {"type": "none"}):
        assert "do NOT call any tool" in turn_directive(choice)


def test_response_format():
    assert "JSON object" in turn_directive(None, {"type": "json_object"})
    schema = {"type": "object", "properties": {"fruit": {"type": "string"}}}
    d = turn_directive(None, {"type": "json_schema", "json_schema": {"name": "f", "schema": schema}})
    assert json.dumps(schema) in d
    assert turn_directive(None, {"type": "text"}) is None


def test_directive_ends_the_prompt():
    async def run():
        request_directive.set(turn_directive("required"))
        return await build_prompt([{"role": "user", "content": "hi"}], [], "m", True)

    assert asyncio.run(run()).rstrip().endswith("do not reply with text only.")
