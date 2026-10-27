"""A token DeepSeek invalidated (quarantined on 401) comes back by itself when
its account has credentials in logins.json: the sweep logs in again, swaps
the token and returns it to ACTIVE. Without credentials, or on a failed login,
it stays quarantined and is not retried before the backoff."""

import asyncio
import json
import os
import tempfile
from unittest import mock

import functions
from test_upstream_challenge import FakeResp


def _pool(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr(functions, "_db", os.path.join(d, "pool.db"))
    functions._relogin_after.clear()
    functions.init_db()
    for alias in ("a", "b"):
        functions.add_token(f"old-{alias}", alias)
        functions.mark_quarantined(len(functions.get_tokens()), "test")
    with open(os.path.join(d, "logins.json"), "w") as f:
        json.dump({"a": {"email": "a@x", "password": "pw", "device_id": "dev"}}, f)


def test_relogin_restores_token_with_credentials(monkeypatch):
    _pool(monkeypatch)
    sent = []

    async def fake_post(path, **kw):
        sent.append((path, kw["json"]))
        return FakeResp(200, "application/json",
                        {"code": 0, "data": {"biz_code": 0, "biz_data": {"user": {"token": "new-a"}}}})

    with mock.patch.object(functions, "post_with_failover", fake_post):
        asyncio.run(functions.relogin_quarantined())
        asyncio.run(functions.relogin_quarantined())  # backoff: no second login
    assert sent == [("/api/v0/users/login", {"email": "a@x", "mobile": None, "password": "pw",
                                             "area_code": None, "device_id": "dev", "os": "android"})]
    toks = {t["alias"]: t for t in functions.get_tokens()}
    assert (toks["a"]["token"], toks["a"]["status"]) == ("new-a", "ACTIVE")
    assert (toks["b"]["token"], toks["b"]["status"]) == ("old-b", "QUARANTINED")


def test_failed_login_keeps_quarantine(monkeypatch):
    _pool(monkeypatch)

    async def fake_post(path, **kw):
        return FakeResp(200, "application/json",
                        {"code": 0, "data": {"biz_code": 11, "biz_msg": "RISK_DEVICE_DETECTED", "biz_data": None}})

    with mock.patch.object(functions, "post_with_failover", fake_post):
        asyncio.run(functions.relogin_quarantined())
    assert {t["alias"]: t["status"] for t in functions.get_tokens()} == {"a": "QUARANTINED", "b": "QUARANTINED"}
