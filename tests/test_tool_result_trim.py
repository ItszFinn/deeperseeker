"""A long tool result keeps its head and its tail.

Seen live: an 8 KB web_search result, 7% over the old 2000-token budget, was
cut to its last half; the model got text starting mid-word ("urced data
points") and never saw the first (best) hit.
"""

from functions import count_tokens
from plugin_helper import _trim_tool_result


def test_cuts_the_middle_not_the_head():
    result = "FIRST HIT " + "filler words here " * 2000 + " LAST LINE: error 42"
    trimmed = _trim_tool_result(result, 500)
    assert count_tokens(trimmed) <= 500
    assert trimmed.startswith("FIRST HIT")
    assert trimmed.endswith("LAST LINE: error 42")
    assert "truncated" in trimmed


def test_fitting_result_is_untouched():
    assert _trim_tool_result("short", 500) == "short"
