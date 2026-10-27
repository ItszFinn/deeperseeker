"""DeepSeek answers bursts with a challenge page instead of JSON (seen live:
HTTP 202 text/html on chat_session/create under 4 parallel requests). That
used to crash with a bare 500; it is a rate limit."""

import asyncio
from unittest import mock

import aiohttp

import app as app_module
import functions
from functions import UpstreamError


class FakeResp:
    def __init__(self, status, content_type, body):
        self.status, self.content_type, self.body = status, content_type, body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def json(self):
        if self.content_type != "application/json":
            raise aiohttp.ContentTypeError(None, (), message=f"unexpected mimetype: {self.content_type}")
        return self.body

    async def text(self):
        return str(self.body)


def test_create_new_chat_retries_challenge_then_succeeds():
    answers = [FakeResp(202, "text/html", "<html>challenge</html>"),
               FakeResp(200, "application/json", {"data": {"biz_data": {"chat_session": {"id": "c1"}}}})]

    async def fake_post(*a, **kw):
        return answers.pop(0)

    async def no_sleep(_):
        pass

    with mock.patch.object(functions, "post_with_failover", fake_post), \
            mock.patch.object(functions.asyncio, "sleep", no_sleep):
        assert asyncio.run(functions.create_new_chat("tok")) == "c1"


def test_create_new_chat_gives_up_with_429():
    async def fake_post(*a, **kw):
        return FakeResp(202, "text/html", "<html>challenge</html>")

    async def no_sleep(_):
        pass

    with mock.patch.object(functions, "post_with_failover", fake_post), \
            mock.patch.object(functions.asyncio, "sleep", no_sleep):
        try:
            asyncio.run(functions.create_new_chat("tok"))
        except UpstreamError as e:
            assert e.status == 429
        else:
            raise AssertionError("no error")


def test_endpoint_answers_upstream_error_with_its_status():
    from fastapi.testclient import TestClient

    async def failing_chat(*a, **kw):
        raise UpstreamError(429, "non-JSON answer (HTTP 202)")

    with mock.patch.multiple(app_module, check_key=lambda req: True, handle_chat=failing_chat):
        client = TestClient(app_module.app)
        r = client.post("/v1/chat/completions", json={"model": "v4.1flash", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 429
        assert r.json()["error"]["type"] == "rate_limit_error"
        r = client.post("/v1/messages", json={"model": "v4.1flash", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 429
        assert r.json()["type"] == "error"


def test_invalid_token_is_a_401_not_a_crash():
    # Seen live: every pool token at once answered HTTP 200
    # {"code":40003,"msg":"INVALID_TOKEN","data":null}; create_new_chat then
    # crashed on data["data"] being None and nothing rotated or quarantined.
    async def fake_post(*a, **kw):
        return FakeResp(200, "application/json", {"code": 40003, "msg": "INVALID_TOKEN", "data": None})

    with mock.patch.object(functions, "post_with_failover", fake_post):
        try:
            asyncio.run(functions.create_new_chat("tok"))
        except UpstreamError as e:
            assert e.status == 401
        else:
            raise AssertionError("no error")
