import json
import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

import main


class FakeEngine:
    def __init__(self, model_id="fake-model"):
        self.model_id = model_id
        self.last_context_info = {
            "compressed": False,
            "prompt_tokens": 5,
            "max_context": 8192,
            "budget": 7900,
            "strategy": "truncate",
            "dropped_messages": 0,
        }

    def infer(self, messages, max_tokens, temperature):
        # Simulate parse error pathway when special trigger is present
        if messages and isinstance(messages[0].get("content"), str) and "PARSE_ERR" in messages[0]["content"]:
            raise ValueError("Simulated parse error")
        # Return echo content for deterministic test
        parts = []
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
            elif isinstance(c, str):
                parts.append(c)
        txt = " ".join(parts) or "OK"
        # Simulate context accounting changing with request
        self.last_context_info = {
            "compressed": False,
            "prompt_tokens": max(1, len(txt.split())),
            "max_context": 8192,
            "budget": 7900,
            "strategy": "truncate",
            "dropped_messages": 0,
        }
        return f"OK: {txt}"

    def infer_stream(self, messages, max_tokens, temperature, cancel_event=None):
        # simple two-piece stream; respects cancel_event if set during streaming
        outputs = ["hello", " world"]
        for piece in outputs:
            if cancel_event is not None and cancel_event.is_set():
                break
            yield piece
            # tiny delay to allow cancel test to interleave
            time.sleep(0.01)

    def get_context_report(self):
        return {
            "compressionEnabled": True,
            "strategy": "truncate",
            "safetyMargin": 256,
            "modelMaxContext": 8192,
            "tokenizerModelMaxLength": 8192,
            "last": self.last_context_info,
        }


@contextmanager
def patched_engine():
    # Patch global engine so server does not load real model
    prev_engine = main._engine
    prev_err = main._engine_error
    fake = FakeEngine()
    main._engine = fake
    main._engine_error = None
    try:
        yield fake
    finally:
        main._engine = prev_engine
        main._engine_error = prev_err


def get_client():
    return TestClient(main.app)


def test_health_ready_and_context():
    with patched_engine():
        client = get_client()
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["modelReady"] is True
        assert body["modelId"] == "fake-model"
        # context block exists with required fields
        ctx = body["context"]
        assert ctx["compressionEnabled"] is True
        assert "last" in ctx
        assert isinstance(ctx["last"].get("prompt_tokens"), int)


def test_health_with_engine_error():
    # simulate model load error path
    prev_engine = main._engine
    prev_err = main._engine_error
    try:
        main._engine = None
        main._engine_error = "boom"
        client = get_client()
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["modelReady"] is False
        assert body["error"] == "boom"
    finally:
        main._engine = prev_engine
        main._engine_error = prev_err


def test_chat_non_stream_validation():
    with patched_engine():
        client = get_client()
        # missing messages should 400
        r = client.post("/v1/chat/completions", json={"messages": []})
        assert r.status_code == 400


def test_chat_non_stream_success_and_usage_context():
    with patched_engine():
        client = get_client()
        payload = {
            "messages": [{"role": "user", "content": "Hello Qwen"}],
            "max_tokens": 8,
            "temperature": 0.0,
        }
        r = client.post("/v1/chat/completions", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"].startswith("OK:")
        # usage prompt_tokens filled from engine.last_context_info
        assert body["usage"]["prompt_tokens"] >= 1
        # response includes context echo
        assert "context" in body
        assert "prompt_tokens" in body["context"]


def test_chat_non_stream_parse_error_to_400():
    with patched_engine():
        client = get_client()
        payload = {
            "messages": [{"role": "user", "content": "PARSE_ERR trigger"}],
            "max_tokens": 4,
        }
        r = client.post("/v1/chat/completions", json=payload)
        # ValueError in engine -> 400 per API contract
        assert r.status_code == 400


def read_sse_lines(resp):
    # Utility to parse event-stream into list of data payloads (including [DONE])
    lines = []
    buf = b""

    # Starlette TestClient (httpx) responses expose iter_bytes()/iter_raw(), not requests.iter_content().
    # Fall back to available iterator or to full content if streaming isn't supported.
    iterator = None
    for name in ("iter_bytes", "iter_raw", "iter_content"):
        it = getattr(resp, name, None)
        if callable(it):
            iterator = it
            break

    if iterator is None:
        data = getattr(resp, "content", b"")
        if isinstance(data, str):
            data = data.encode("utf-8", "ignore")
        buf = data
    else:
        for chunk in iterator():
            if not chunk:
                continue
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8", "ignore")
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                # keep original frame text for asserts
                lines.append(frame.decode("utf-8", errors="ignore"))

    # Drain any leftover
    if buf:
        lines.append(buf.decode("utf-8", errors="ignore"))
    return lines


def test_chat_stream_sse_flow_and_resume():
    with patched_engine():
        client = get_client()
        payload = {
            "session_id": "s1",
            "stream": True,
            "messages": [{"role": "user", "content": "stream please"}],
            "max_tokens": 8,
            "temperature": 0.2,
        }
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            assert resp.status_code == 200
            lines = read_sse_lines(resp)
        # Must contain role delta, content pieces, finish chunk, and [DONE]
        joined = "\n".join(lines)
        assert "delta" in joined
        assert "[DONE]" in joined

        # Resume from event index 0 should receive at least one subsequent event
        headers = {"Last-Event-ID": "s1:0"}
        with client.stream("POST", "/v1/chat/completions", headers=headers, json=payload) as resp2:
            assert resp2.status_code == 200
            lines2 = read_sse_lines(resp2)
        assert any("data:" in l for l in lines2)
        assert "[DONE]" in "\n".join(lines2)

        # Invalid Last-Event-ID format should not crash (covered by try/except)
        headers_bad = {"Last-Event-ID": "not-an-index"}
        with client.stream("POST", "/v1/chat/completions", headers=headers_bad, json=payload) as resp3:
            assert resp3.status_code == 200
            _ = read_sse_lines(resp3)  # just ensure no crash


def test_cancel_endpoint_stops_generation():
    with patched_engine():
        client = get_client()
        payload = {
            "session_id": "to-cancel",
            "stream": True,
            "messages": [{"role": "user", "content": "cancel me"}],
        }
        # Start streaming in background (client.stream keeps the connection open)
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            # Immediately cancel
            rc = client.post("/v1/cancel/to-cancel")
            assert rc.status_code == 200
            # Stream should end with [DONE] without hanging
            lines = read_sse_lines(resp)
            assert "[DONE]" in "\n".join(lines)


def test_cancel_unknown_session_is_ok():
    with patched_engine():
        client = get_client()
        rc = client.post("/v1/cancel/does-not-exist")
        # Endpoint returns ok regardless (idempotent, operationally safe)
        assert rc.status_code == 200


def test_edge_large_last_event_id_after_finish_yields_done():
    with patched_engine():
        client = get_client()
        payload = {
            "session_id": "done-session",
            "stream": True,
            "messages": [{"role": "user", "content": "edge"}],
        }
        # Complete a run
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            _ = read_sse_lines(resp)
        # Resume with huge index; should return DONE quickly
        headers = {"Last-Event-ID": "done-session:99999"}
        with client.stream("POST", "/v1/chat/completions", headers=headers, json=payload) as resp2:
            lines2 = read_sse_lines(resp2)
        assert "[DONE]" in "\n".join(lines2)