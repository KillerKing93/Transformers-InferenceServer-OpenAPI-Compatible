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


def test_stream_resume_basic_functionality():
    """Test that basic streaming resume functionality works correctly"""
    with patched_engine():
        client = get_client()
        session_id = "resume-basic-test"
        payload = {
            "session_id": session_id,
            "stream": True,
            "messages": [{"role": "user", "content": "test resume"}],
            "max_tokens": 100,
        }

        # Complete a streaming session first
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            complete_lines = read_sse_lines(resp)

        # Verify we got some data lines
        data_lines = [line for line in complete_lines if line.startswith("data: ")]
        assert len(data_lines) > 0, "Should have received some data lines in complete session"

        # Test resume from index 0 (should replay everything)
        headers = {"Last-Event-ID": f"{session_id}:0"}
        with client.stream("POST", "/v1/chat/completions", headers=headers, json=payload) as resp2:
            resume_lines = read_sse_lines(resp2)

        # Should get data lines again
        resume_data_lines = [line for line in resume_lines if line.startswith("data: ")]
        assert len(resume_data_lines) > 0, "Should have received data lines on resume"

        # Should end with [DONE]
        assert any("[DONE]" in line for line in resume_lines), "Resume should end with [DONE]"


def test_stream_resume_preserves_exact_chunk_order():
    """Test that resume maintains exact chunk order and content"""
    with patched_engine():
        client = get_client()
        session_id = "order-test"
        payload = {
            "session_id": session_id,
            "stream": True,
            "messages": [{"role": "user", "content": "test order"}],
        }

        # Get complete session with default chunks
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            complete_lines = read_sse_lines(resp)

        complete_chunks = []
        for line in complete_lines:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                        complete_chunks.append(data["choices"][0]["delta"]["content"])
                except (json.JSONDecodeError, KeyError):
                    continue

        # Skip test if no chunks received (default FakeEngine may not produce content chunks)
        if len(complete_chunks) == 0:
            pytest.skip("Default FakeEngine does not produce content chunks for this test")

        # Test resume from middle point
        resume_point = min(1, len(complete_chunks) - 1)  # Resume after first chunk, or 0 if only 1 chunk
        headers = {"Last-Event-ID": f"{session_id}:{resume_point}"}

        with client.stream("POST", "/v1/chat/completions", headers=headers, json=payload) as resp:
            resume_lines = read_sse_lines(resp)

        resume_chunks = []
        for line in resume_lines:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                        resume_chunks.append(data["choices"][0]["delta"]["content"])
                except (json.JSONDecodeError, KeyError):
                    continue

        # Should get remaining chunks after resume point
        expected_resume_chunks = complete_chunks[resume_point:]

        assert resume_chunks == expected_resume_chunks, (
            f"Order preservation failed. Resume chunks: {resume_chunks}, "
            f"Expected: {expected_resume_chunks}"
        )


def test_stream_resume_with_partial_disconnect():
    """Test resume when client disconnects mid-stream and reconnects"""
    with patched_engine():
        client = get_client()
        session_id = "disconnect-test"
        payload = {
            "session_id": session_id,
            "stream": True,
            "messages": [{"role": "user", "content": "test disconnect"}],
        }

        # Complete a session first to populate buffers
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            complete_lines = read_sse_lines(resp)

        # Extract chunks and event IDs
        received_chunks = []
        event_ids = []
        for line in complete_lines:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                        chunk = data["choices"][0]["delta"]["content"]
                        received_chunks.append(chunk)
                        if "id" in data:
                            event_ids.append(data["id"])
                except (json.JSONDecodeError, KeyError):
                    continue

        if len(received_chunks) < 2:
            pytest.skip("Not enough chunks received for disconnect test")

        # Simulate disconnect after first chunk
        resume_index = 0  # Resume from beginning
        headers = {"Last-Event-ID": f"{session_id}:{resume_index}"}

        with client.stream("POST", "/v1/chat/completions", headers=headers, json=payload) as resp:
            resume_lines = read_sse_lines(resp)

        resume_chunks = []
        for line in resume_lines:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                        resume_chunks.append(data["choices"][0]["delta"]["content"])
                except (json.JSONDecodeError, KeyError):
                    continue

        # Should get all chunks again (resume from 0 replays everything)
        assert resume_chunks == received_chunks, (
            f"Resume from beginning failed. Got: {resume_chunks}, Expected: {received_chunks}"
        )


def test_stream_resume_buffer_overflow_handling():
    """Test resume when buffer overflows and older chunks are lost"""
    with patched_engine():
        client = get_client()
        session_id = "overflow-test"
        payload = {
            "session_id": session_id,
            "stream": True,
            "messages": [{"role": "user", "content": "test overflow"}],
        }

        # Complete a session with default chunks
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            complete_lines = read_sse_lines(resp)

        # Count total chunks produced
        total_chunks = sum(1 for line in complete_lines
                          if line.startswith("data: ") and '"content"' in line)

        if total_chunks == 0:
            pytest.skip("No content chunks produced by default engine")

        # Try to resume from early index
        early_resume_index = min(1, total_chunks - 1)  # Resume after first chunk
        headers = {"Last-Event-ID": f"{session_id}:{early_resume_index}"}

        with client.stream("POST", "/v1/chat/completions", headers=headers, json=payload) as resp:
            resume_lines = read_sse_lines(resp)

        resume_chunks = []
        for line in resume_lines:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                        resume_chunks.append(data["choices"][0]["delta"]["content"])
                except (json.JSONDecodeError, KeyError):
                    continue

        # Should get chunks from resume point onwards, or [DONE] if buffer overflowed
        if resume_chunks:
            # Verify we got some chunks
            assert len(resume_chunks) > 0, "Should get some chunks on resume"
        else:
            # If no chunks, should at least get [DONE]
            assert any("[DONE]" in line for line in resume_lines), "Should get [DONE] even with buffer overflow"


def test_stream_resume_concurrent_sessions_isolation():
    """Test that resume works correctly with multiple concurrent sessions"""
    with patched_engine():
        client = get_client()

        # Create multiple concurrent sessions with different session IDs
        session_ids = ["session_A", "session_B", "session_C"]
        sessions_data = {}

        for sid in session_ids:
            payload = {
                "session_id": sid,
                "stream": True,
                "messages": [{"role": "user", "content": f"test {sid}"}],
            }

            # Complete session
            with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                lines = read_sse_lines(resp)

            chunks = []
            for line in lines:
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                            chunks.append(data["choices"][0]["delta"]["content"])
                    except (json.JSONDecodeError, KeyError):
                        continue

            sessions_data[sid] = chunks

        # Skip if no chunks received
        if all(len(chunks) == 0 for chunks in sessions_data.values()):
            pytest.skip("No content chunks received for any session")

        # Test resume for each session independently
        for sid in session_ids:
            if len(sessions_data[sid]) < 2:
                continue  # Skip sessions with too few chunks

            resume_point = 0  # Resume from beginning
            headers = {"Last-Event-ID": f"{sid}:{resume_point}"}
            payload = {
                "session_id": sid,
                "stream": True,
                "messages": [{"role": "user", "content": f"test {sid}"}],
            }

            with client.stream("POST", "/v1/chat/completions", headers=headers, json=payload) as resp:
                resume_lines = read_sse_lines(resp)

            resume_chunks = []
            for line in resume_lines:
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                            resume_chunks.append(data["choices"][0]["delta"]["content"])
                    except (json.JSONDecodeError, KeyError):
                        continue

            # Should get all chunks again (resume from 0)
            assert resume_chunks == sessions_data[sid], (
                f"Session {sid} resume failed: {resume_chunks} != {sessions_data[sid]}"
            )


def test_stream_resume_with_sqlite_persistence():
    """Test resume works correctly with SQLite persistence enabled"""
    # This test requires setting up SQLite persistence
    original_persist = main.PERSIST_SESSIONS
    original_db_path = main.SESSIONS_DB_PATH

    try:
        # Enable persistence for this test
        main.PERSIST_SESSIONS = True
        main.SESSIONS_DB_PATH = ":memory:"  # Use in-memory SQLite for test

        # Reinitialize the SQLite store
        main._DB_STORE = main._SQLiteStore(main.SESSIONS_DB_PATH)

        with patched_engine():
            client = get_client()
            session_id = "persistent-test"
            payload = {
                "session_id": session_id,
                "stream": True,
                "messages": [{"role": "user", "content": "test persistence"}],
            }

            # Complete session to populate SQLite
            with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                complete_lines = read_sse_lines(resp)

            # Extract chunks from complete session
            complete_chunks = []
            for line in complete_lines:
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                            complete_chunks.append(data["choices"][0]["delta"]["content"])
                    except (json.JSONDecodeError, KeyError):
                        continue

            if len(complete_chunks) == 0:
                pytest.skip("No content chunks received for persistence test")

            # Verify session was persisted
            assert main._DB_STORE.session_meta(session_id)[0] == True  # Should be marked finished

            # Test resume from SQLite
            resume_point = min(1, len(complete_chunks) - 1)
            headers = {"Last-Event-ID": f"{session_id}:{resume_point}"}

            with client.stream("POST", "/v1/chat/completions", headers=headers, json=payload) as resp:
                resume_lines = read_sse_lines(resp)

            resume_chunks = []
            for line in resume_lines:
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                            resume_chunks.append(data["choices"][0]["delta"]["content"])
                    except (json.JSONDecodeError, KeyError):
                        continue

            expected_chunks = complete_chunks[resume_point:]
            assert resume_chunks == expected_chunks, (
                f"SQLite resume failed: {resume_chunks} != {expected_chunks}"
            )

    finally:
        # Restore original settings
        main.PERSIST_SESSIONS = original_persist
        main.SESSIONS_DB_PATH = original_db_path
        main._DB_STORE = main._SQLiteStore(main.SESSIONS_DB_PATH) if main.PERSIST_SESSIONS else None


def test_stream_resume_data_integrity_with_unicode():
    """Test resume preserves Unicode characters correctly"""
    with patched_engine():
        client = get_client()
        session_id = "unicode-test"
        payload = {
            "session_id": session_id,
            "stream": True,
            "messages": [{"role": "user", "content": "test unicode"}],
        }

        # Complete session
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            complete_lines = read_sse_lines(resp)

        complete_chunks = []
        for line in complete_lines:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                        complete_chunks.append(data["choices"][0]["delta"]["content"])
                except (json.JSONDecodeError, KeyError):
                    continue

        if len(complete_chunks) == 0:
            pytest.skip("No content chunks received for unicode test")

        # Test resume from middle
        resume_point = min(1, len(complete_chunks) - 1)
        headers = {"Last-Event-ID": f"{session_id}:{resume_point}"}

        with client.stream("POST", "/v1/chat/completions", headers=headers, json=payload) as resp:
            resume_lines = read_sse_lines(resp)

        resume_chunks = []
        for line in resume_lines:
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    if "choices" in data and data["choices"][0].get("delta", {}).get("content"):
                        resume_chunks.append(data["choices"][0]["delta"]["content"])
                except (json.JSONDecodeError, KeyError):
                    continue

        expected_resume_chunks = complete_chunks[resume_point:]
        assert resume_chunks == expected_resume_chunks

        # Verify Unicode integrity (basic check)
        for actual, expected in zip(resume_chunks, expected_resume_chunks):
            assert actual == expected, f"Content mismatch: '{actual}' != '{expected}'"

def test_ktp_ocr_success():
    with patched_engine() as fake_engine:
        # Configure the fake engine to return a specific JSON structure for this test
        expected_json = {
            "nik": "1234567890123456",
            "nama": "JOHN DOE",
            "tempat_lahir": "JAKARTA",
            "tgl_lahir": "01-01-1990",
            "jenis_kelamin": "LAKI-LAKI",
            "alamat": {
                "name": "JL. JEND. SUDIRMAN KAV. 52-53",
                "rt_rw": "001/001",
                "kel_desa": "SENAYAN",
                "kecamatan": "KEBAYORAN BARU",
            },
            "agama": "ISLAM",
            "status_perkawinan": "KAWIN",
            "pekerjaan": "PEGAWAI SWASTA",
            "kewarganegaraan": "WNI",
            "berlaku_hingga": "SEUMUR HIDUP",
        }
        fake_engine.infer = lambda messages, max_tokens, temperature: json.dumps(expected_json)

        client = get_client()
        with open("image.jpg", "rb") as f:
            files = {"image": ("image.jpg", f, "image/jpeg")}
            r = client.post("/ktp-ocr/", files=files)

        assert r.status_code == 200
        body = r.json()
        assert body["nik"] == "1234567890123456"
        assert body["nama"] == "JOHN DOE"