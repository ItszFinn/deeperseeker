"""A chat rebuilt from the full history (session miss: the client compressed its
context, proxy restart, token rotation) must show the model where it stands.

Seen live: a request ending in a tool result was rebuilt with the original
user question re-sent as the current [USER] message and without the
assistant's own tool call in the history, so the model called the same tool
again instead of answering from the result.
"""

import asyncio

from plugin_helper import build_prompt

W = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
U = {"role": "user", "content": "What's the weather in Madrid?"}
CALL = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Madrid"}'}}]}
RESULT = {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 31}'}


def prompt(*msgs):
    return asyncio.run(build_prompt([{"role": "system", "content": "sys"}, *msgs], [W], "m", True))


def test_tool_result_turn_does_not_resend_old_question():
    p = prompt(U, CALL, RESULT)
    assert "[USER]" not in p
    history = p.split("[PREVIOUS CONVERSATION HISTORY]\n", 1)[1].split("[TOOL RESULTS]", 1)[0]
    assert history.index("Madrid?") < history.index("get_weather")
    assert p.index("[TOOL RESULTS]") > p.index("[PREVIOUS CONVERSATION HISTORY]")
    assert '"temp_c": 31' in p.split("[TOOL RESULTS]", 1)[1]


def test_only_new_user_text_is_the_turn():
    answer = {"role": "assistant", "content": "31 degrees."}
    p = prompt(U, CALL, RESULT, answer, {"role": "user", "content": "And tomorrow?"})
    assert p.count("[USER]") == 1
    assert p.split("[USER]\n", 1)[1].strip() == "And tomorrow?"
    history = p.split("[PREVIOUS CONVERSATION HISTORY]\n", 1)[1].split("[USER]", 1)[0]
    # older results stay in order inside the history, not re-sent as the latest
    assert history.index("get_weather") < history.index('"temp_c": 31') < history.index("31 degrees")
    assert "[TOOL RESULTS]" not in p
