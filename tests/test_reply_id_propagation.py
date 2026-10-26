"""The reply id DeepSeek reports reaches the caller of send_message.

send_message drains the upstream in its own task (a copy of the caller's
context), so the id it set there never reached handle_chat, which then saved
every next turn at parent + 2. Live, with Hermes' main agent and its background
review in one chat, each follow-up continued 4 messages early, on the other
agent's branch, and both agents got each other's tool results.
"""

import asyncio
import json

import functions
from test_sse_upstream import _collect_send_message


def test_reply_id_reaches_the_caller_context():
    body = "".join("data: " + json.dumps(e) + "\n" for e in (
        {"request_message_id": 7, "response_message_id": 8},
        {"v": {"response": {"message_id": 8, "fragments": [{"id": 1, "type": "RESPONSE", "content": "ok"}]}}},
        {"p": "response/status", "o": "SET", "v": "FINISHED"},
    )).encode()

    async def run():
        functions.reply_id.set(None)
        text = await _collect_send_message(body)
        return text, functions.reply_id.get()

    assert asyncio.run(run()) == ("ok", 8)
