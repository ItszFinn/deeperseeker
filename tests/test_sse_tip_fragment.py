"""Fragments that are not the reply stay out of it.

Seen live (DeepSeek chat history): replies end with a fragment
{"type":"TIP","style":"WARNING","content":"This response is AI-generated, for
reference only."}. It was streamed as answer text: 28 Hermes replies carried
it, 15 of them as the only text next to a tool call.
"""

import asyncio
import json

from test_sse_upstream import _collect_send_message

TIP = "This response is AI-generated, for reference only."


def _body(*events):
    return "".join(
        "data: " + json.dumps(e, separators=(",", ":")) + "\n" for e in events
    ).encode()


def test_tip_fragment_is_not_reply_text():
    body = _body(
        {"v": {"response": {"message_id": 2, "fragments": [{"id": 1, "type": "RESPONSE", "content": "Antwort"}]}}},
        {"p": "response/fragments/-1/content", "o": "APPEND", "v": " fertig"},
        {"p": "response/fragments", "o": "APPEND", "v": [
            {"id": 2, "type": "TIP", "style": "WARNING", "hide_on_wip": True, "content": "This response"}]},
        {"p": "response/fragments/-1/content", "o": "APPEND", "v": " is AI-generated,"},
        {"v": " for reference only."},
        {"p": "response/status", "o": "SET", "v": "FINISHED"},
    )
    assert asyncio.run(_collect_send_message(body)) == "Antwort fertig"


def test_tip_in_the_opening_event_is_dropped():
    body = _body(
        {"v": {"response": {"message_id": 2, "fragments": [
            {"id": 1, "type": "RESPONSE", "content": "Antwort"},
            {"id": 2, "type": "TIP", "style": "WARNING", "content": TIP}]}}},
        {"p": "response/status", "o": "SET", "v": "FINISHED"},
    )
    assert asyncio.run(_collect_send_message(body)) == "Antwort"
