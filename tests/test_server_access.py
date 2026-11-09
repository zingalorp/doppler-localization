"""The app answers only pages from this computer, whatever a web page elsewhere makes the browser send."""

import asyncio

import pytest

from doppler_loc.gui.server import OnlyThisComputer


def _status(host, origin=None, method="GET"):
    reached, sent = [], []

    async def inner(scope, receive, send):
        reached.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    headers = [(b"host", host.encode())] if host is not None else []
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    scope = {"type": "http", "method": method, "path": "/api/browse", "raw_path": b"/api/browse", "query_string": b"", "headers": headers}
    asyncio.run(OnlyThisComputer(inner)(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    assert bool(reached) == (status == 200)
    return status


@pytest.mark.parametrize("host, origin", [
    ("127.0.0.1:8765", None),
    ("127.0.0.1", None),
    ("localhost:8765", None),
    ("LOCALHOST:8765", None),
    ("127.0.0.1:8765", "http://127.0.0.1:8765"),
    ("localhost:8765", "http://localhost:8765"),
])
def test_this_computer_is_answered(host, origin):
    assert _status(host, origin, "POST" if origin else "GET") == 200


@pytest.mark.parametrize("host, origin", [
    ("evil.example", None),  # its domain pointed at 127.0.0.1
    ("evil.example:8765", None),
    ("127.0.0.1.evil.example:8765", None),
    ("localhost.evil.example", None),
    ("192.168.1.5:8765", None),
    ("[bad", None),
    (None, None),
    ("127.0.0.1:8765", "https://evil.example"),  # a page elsewhere asking directly
    ("127.0.0.1:8765", "http://127.0.0.1.evil.example"),
    ("127.0.0.1:8765", "null"),  # a sandboxed frame
])
def test_other_pages_are_refused(host, origin):
    assert _status(host, origin, "POST" if origin else "GET") == 403
