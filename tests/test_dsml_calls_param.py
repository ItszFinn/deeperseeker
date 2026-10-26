"""A parameter named like a wrapper tag, written in the name-in-tag form.

Seen live (raw DeepSeek replies, Hermes' tool_call bridge tool whose argument
is "calls"): '<｜｜DSML｜｜calls" string="false">[...]</｜｜DSML｜｜calls>' and
'<｜DSML｜ calls string="false">'. The parameter was dropped (tool_call({})),
and with small stream chunks the JSON inside the value was taken for a call of
its own, so the result depended on chunk size.
"""

import pytest

import functions as F

D2, D1 = "｜｜DSML｜｜", "｜DSML｜ "
QUOTE_FORM = (
    f'<{D2}function_calls>\n<{D2}invoke name="tool_call">\n'
    f'<{D2}calls" string="false">[{{"arguments": {{"action": "poll"}}, "name": "process_manage"}}]</{D2}calls>\n'
    f'</{D2}invoke>\n<{D2}invoke name="terminal">\n<{D2}command" string="true">ls -la</{D2}command>\n'
    f"</{D2}invoke>\n</{D2}function_calls>"
)
SPACE_FORM = (
    f'<{D1}calls>\n<{D1}invoke name="tool_call">\n'
    f'<{D1}calls string="false">[{{"name": "cronjob_manage", "arguments": {{"action": "list"}}}}]</{D1}calls>\n'
    f"</{D1}invoke>\n</{D1}calls>"
)


def fn(name, props, required):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {p: {} for p in props}, "required": required}}}


@pytest.fixture(autouse=True)
def tools():
    token = F.request_tools.set([fn("tool_call", ["calls"], ["calls"]), fn("terminal", ["command"], ["command"])])
    yield
    F.request_tools.reset(token)


def calls(items):
    return [(c["function"]["name"], c["function"]["arguments"]) for c in items]


def stream(text, size):
    p, out = F.StreamToolParser(), []
    for i in range(0, len(text), size):
        out += p.feed(text[i:i + size])
    out += p.flush()
    return calls(r["tool"] for r in out if "tool" in r), "".join(r.get("text", "") for r in out).strip()


@pytest.mark.parametrize("text, want", [
    (QUOTE_FORM, [("tool_call", '{"calls": [{"arguments": {"action": "poll"}, "name": "process_manage"}]}'),
                  ("terminal", '{"command": "ls -la"}')]),
    (SPACE_FORM, [("tool_call", '{"calls": [{"name": "cronjob_manage", "arguments": {"action": "list"}}]}')]),
])
def test_calls_parameter_is_read_at_any_chunk_size(text, want):
    tools, clean = F.parse_tools(text)
    assert (calls(tools), clean) == (want, "")
    for size in (1, 7, 10000):
        assert stream(text, size) == (want, "")
