"""Parameters after a closed invoke: the model forgot its next invoke opener.

Seen live (raw DeepSeek replies from Hermes sessions): a "code" parameter meant
for execute_code right after a closed terminal call, and a habitual "timeout"
after a closed write_file call. The proxy merged them into the closed call and
streamed their values out as message text.
"""

import pytest

import functions as F

B, E = "<｜｜DSML｜｜", "</｜｜DSML｜｜"


def P(n, v, s="true"):
    return f'{B}parameter name="{n}" string="{s}">{v}{E}parameter>\n'


FORGOT_OPENER = (f'{B}function_calls>\n{B}invoke name="terminal">\n' + P("command", "ls ~")
                 + P("timeout", "120", "false") + f"{E}invoke>\n" + P("code", 'print("logged")')
                 + f"{E}parameter>\n{E}invoke>\n{E}function_calls>")
STRAY_PARAM = (f'{B}function_calls>\n{B}invoke name="write_file">\n' + P("path", "/tmp/x.md")
               + P("content", "# hi\n") + f"{E}invoke>\n" + P("timeout", "30", "false")
               + f"{E}invoke>\n{E}function_calls>")


def fn(name, props, required):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {p: {} for p in props}, "required": required}}}


TOOLS = [fn("terminal", ["command", "timeout"], ["command"]), fn("write_file", ["path", "content"], ["path", "content"]),
         fn("execute_code", ["code"], ["code"])]


@pytest.fixture(autouse=True)
def tools():
    token = F.request_tools.set(TOOLS)
    yield
    F.request_tools.reset(token)


def calls(items):
    return [(c["function"]["name"], c["function"]["arguments"]) for c in items]


def stream(text, size):
    p, out = F.StreamToolParser(), []
    for i in range(0, len(text), size):
        out += p.feed(text[i:i + size])
    out += p.flush()
    return calls(r["tool"] for r in out if "tool" in r), "".join(r.get("text", "") for r in out)


def test_orphans_become_the_call_they_satisfy():
    want = [("terminal", '{"command": "ls ~", "timeout": 120}'), ("execute_code", '{"code": "print(\\"logged\\")"}')]
    tools, text = F.parse_tools(FORGOT_OPENER)
    assert (calls(tools), text) == (want, "")
    for size in (1, 7, 50, 10000):
        assert stream(FORGOT_OPENER, size) == (want, "")


def test_stray_orphan_is_dropped_not_merged_or_shown():
    want = [("write_file", '{"path": "/tmp/x.md", "content": "# hi"}')]
    tools, text = F.parse_tools(STRAY_PARAM)
    assert (calls(tools), text) == (want, "")
    for size in (1, 7, 50, 10000):
        assert stream(STRAY_PARAM, size) == (want, "")


def test_orphan_that_completes_the_closed_call_is_merged():
    early = (f'{B}function_calls>\n{B}invoke name="terminal">\n' + P("command", "ls")
             + f"{E}invoke>\n" + P("timeout", "60", "false") + f"{E}function_calls>")
    tools, text = F.parse_tools(early)
    assert (calls(tools), text) == ([("terminal", '{"command": "ls", "timeout": 60}')], "")
