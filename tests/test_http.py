"""Contract <-> test mapping for the exl3-serve HTTP surface.

Every bullet of the task spec's Contract section, plus the depth requirements,
mapped to the tests that pin it (>= 2 assertions per clause where meaningful):

| Contract clause | Tests |
|---|---|
| /props: model_path (absolute), chat_template, total_slots, default_generation_settings.n_ctx | test_props_fields |
| /props carries NO build_info; identity goes in the Server header | test_props_has_no_build_info, test_server_header_on_every_response |
| Server: exl3-serve/<v> exllamav3/<v or unknown> on every response | test_server_header_on_every_response |
| /health 200 {"status":"ok"} when loaded, 503 loading shape while loading | test_health_ready, test_health_loading_503 |
| /slots: id / is_processing / n_ctx / n_past (0 when free) | test_slots_idle, test_slots_busy_tracks_context |
| POST /apply-template -> {"prompt": ...} | test_apply_template, test_apply_template_rejects_bad_bodies |
| Final-chunk timings.predicted_ms / predicted_n from the engine's eos result, never wall clock | test_final_timings_from_engine_not_wall_clock, test_draft_timings_only_with_draft_stats |
| draft_n / draft_n_accepted only when the engine reports draft counters | test_draft_timings_only_with_draft_stats |
| Per-token provisional timings, replaced by engine figures on the final chunk; timings ALWAYS on the final chunk | test_timings_per_token_provisional_then_final, test_final_chunk_always_carries_timings |
| data: [DONE] always terminates a stream, including after a mid-stream error | test_stream_ends_with_done, test_midstream_error_then_done |
| Event loop never blocked by engine calls (/health during 50 ms/token answers < 100 ms) | test_health_during_generation_is_fast |
| Two concurrent streams on --parallel 2 both complete, isolated tokens | test_two_concurrent_streams_are_isolated |
| A third concurrent request waits and then completes (not rejected) | test_third_request_waits_for_free_slot |
| Think-split across token boundaries (tags one character per token) | test_think_split_char_by_char, test_stop_string_across_token_boundary (tag plumbing), test_think_split_unterminated |
| --no-think-split sends everything as content | test_no_think_split_flag |
| engine_exl3.py compiles; package imports without exllamav3 | test_engine_exl3_compiles, test_imports_without_exllamav3 |
| No new dependencies / no network in tests | by construction: stdlib + aiohttp test client on 127.0.0.1 (see report) |
| prompt_progress chunks (processed=0, then =total on first token; no choices) | test_prompt_progress_chunks |
| usage on final chunk with stream_options.include_usage | test_usage_on_final_chunk |
| max_tokens / n_predict alias honoured; "length" finish | test_length_finish, test_n_predict_alias |
| 3-class input defence: malformed JSON -> 400; 'prompt' -> 400 naming /completion; non-string content -> 400 | test_bad_bodies_rejected |
| Huge max_tokens clamped to --max-tokens-cap (default 4096) | test_max_tokens_clamped |
| temperature 0/absent -> greedy, else passed through | test_temperature_routing |
| stop strings truncate content and finish "stop" (incl. across token boundary) | test_stop_string_truncates, test_stop_string_across_token_boundary |
| Non-streaming chat.completion shape (message/usage/timings) | test_nonstream_shape, test_nonstream_think_split |
| model field ignored and echoed back | test_model_field_echoed |
| 503 with the loading error shape on completions while loading | test_health_loading_503 |
| Self-review defect classes: client disconnect (slot/job cleanup), stop string spanning tokens, unterminated think block | test_client_disconnect_frees_slot_and_cancels_job, test_stop_string_across_token_boundary, test_think_split_unterminated |
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import ClientSession, web

from exl3_serve import __version__
from exl3_serve.engine_fake import FakeEngine
from exl3_serve.server import Options, create_app

pytestmark = pytest.mark.asyncio

REPO_ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def serve(engine=None, loading_gate=None, **opts):
    """Start the real app on a random localhost port, yield a ClientSession.

    handler_cancellation=True so a client disconnect cancels the streaming
    handler (the cleanup path the disconnect test exercises).
    """
    if engine is None and loading_gate is None:
        engine = FakeEngine()
    if loading_gate is not None:
        app = create_app(Options(**opts),
                         loader=lambda: (loading_gate.wait(), FakeEngine())[1])
    else:
        app = create_app(Options(**opts), engine=engine)
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    async with ClientSession(base_url=f"http://127.0.0.1:{port}") as session:
        yield session
    await runner.cleanup()


async def read_sse(resp):
    """Parse an SSE body into a list of payloads; '[DONE]' kept as a string."""
    events = []
    while True:
        raw = await resp.content.readline()
        if not raw:
            break
        line = raw.decode().strip()
        if line.startswith("data: "):
            payload = line[len("data: "):]
            events.append(json.loads(payload) if payload != "[DONE]" else "[DONE]")
    return events


async def sse(client, body):
    resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status == 200
    return await read_sse(resp)


def content_of(events):
    return "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in events if isinstance(c, dict) and c.get("choices"))


def reasoning_of(events):
    return "".join(
        c["choices"][0]["delta"].get("reasoning_content", "")
        for c in events if isinstance(c, dict) and c.get("choices"))


def final_chunk(events):
    finals = [c for c in events if isinstance(c, dict) and c.get("choices")
              and c["choices"][0]["finish_reason"] is not None]
    assert finals, "no final chunk in stream"
    return finals[-1]


USER = [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------- /props

async def test_props_fields():
    async with serve() as client:
        r = await client.get("/props")
        assert r.status == 200
        p = await r.json()
        assert p["model_path"] == "/tmp/fake-model" and os.path.isabs(p["model_path"])
        assert isinstance(p["chat_template"], str) and p["chat_template"]
        assert p["total_slots"] == 2  # --parallel default
        assert p["default_generation_settings"]["n_ctx"] == 32768


async def test_props_has_no_build_info():
    async with serve() as client:
        p = await (await client.get("/props")).json()
        assert "build_info" not in p
        # the engine identity travels in the Server header instead
        r = await client.get("/props")
        assert r.headers["Server"] == f"exl3-serve/{__version__} exllamav3/unknown"


async def test_server_header_on_every_response():
    async with serve() as client:
        cases = [
            ("GET", "/props", {}),
            ("GET", "/health", {}),
            ("GET", "/slots", {}),
            ("POST", "/v1/chat/completions", {"json": {"messages": USER}}),
            ("POST", "/v1/chat/completions", {"json": {"bogus": 1}}),   # 400
            ("POST", "/v1/chat/completions", {"data": "{nope"}),        # 400
        ]
        for method, path, kw in cases:
            r = await client.request(method, path, **kw)
            server = r.headers.get("Server", "")
            assert server.startswith("exl3-serve/"), (path, server)
            assert " exllamav3/" in server, (path, server)


# ---------------------------------------------------------------- /health, /slots

async def test_health_ready():
    async with serve() as client:
        r = await client.get("/health")
        assert r.status == 200
        assert await r.json() == {"status": "ok"}


async def test_health_loading_503():
    gate = threading.Event()
    async with serve(loading_gate=gate) as client:
        r = await client.get("/health")
        assert r.status == 503
        assert await r.json() == {"error": {"message": "Loading model"}}
        r2 = await client.post("/v1/chat/completions", json={"messages": USER})
        assert r2.status == 503
        assert await r2.json() == {"error": {"message": "Loading model"}}
        gate.set()
        for _ in range(200):  # load runs in a thread; poll until ready
            if (await client.get("/health")).status == 200:
                break
            await asyncio.sleep(0.02)
        h = await client.get("/health")
        assert h.status == 200 and await h.json() == {"status": "ok"}


async def test_slots_idle():
    async with serve(parallel=3) as client:
        r = await client.get("/slots")
        assert r.status == 200
        s = await r.json()
        assert [x["id"] for x in s] == [0, 1, 2]
        assert all(x["is_processing"] is False and x["n_past"] == 0 for x in s)
        assert all(x["n_ctx"] == 32768 for x in s)


async def test_slots_busy_tracks_context():
    eng = FakeEngine(script="x" * 60, delay=0.02)
    async with serve(engine=eng) as client:
        task = asyncio.create_task(
            sse(client, {"messages": USER, "stream": True}))
        await asyncio.sleep(0.15)
        s = await (await client.get("/slots")).json()
        busy = [x for x in s if x["is_processing"]]
        assert len(busy) == 1
        assert busy[0]["n_past"] > 0
        await task
        s2 = await (await client.get("/slots")).json()
        assert all(x["is_processing"] is False and x["n_past"] == 0 for x in s2)


# ---------------------------------------------------------------- /apply-template

async def test_apply_template():
    async with serve() as client:
        r = await client.post("/apply-template", json={
            "messages": [{"role": "system", "content": "be brief"},
                         {"role": "user", "content": "hi"}]})
        assert r.status == 200
        prompt = (await r.json())["prompt"]
        assert "be brief" in prompt and "hi" in prompt
        assert "<|assistant|>" in prompt  # generation prompt appended
        r2 = await client.post("/apply-template", json={
            "messages": USER, "chat_template_kwargs": {"reasoning_effort": "low"}})
        assert r2.status == 200 and "prompt" in await r2.json()


async def test_apply_template_rejects_bad_bodies():
    async with serve() as client:
        r = await client.post("/apply-template", data="{not json")
        assert r.status == 400
        e = (await r.json())["error"]
        assert e["code"] == 400 and e["type"] == "invalid_request_error"
        r2 = await client.post("/apply-template", json={"prompt": "hi"})
        assert r2.status == 400
        assert "/completion" in (await r2.json())["error"]["message"]


# ---------------------------------------------------------------- non-streaming

async def test_nonstream_shape():
    async with serve() as client:
        r = await client.post("/v1/chat/completions",
                              json={"messages": [{"role": "user", "content": "toktape"}]})
        assert r.status == 200
        b = await r.json()
        assert b["object"] == "chat.completion" and b["id"].startswith("chatcmpl-")
        ch = b["choices"][0]
        assert ch["finish_reason"] == "stop"
        assert ch["message"]["role"] == "assistant"
        assert ch["message"]["content"] == "The fake model answers toktape."
        assert ch["message"]["reasoning_content"] == "The fake model thinks about toktape."
        assert b["model"] == "fake-model"  # alias defaults to the model dir basename
        assert b["usage"]["prompt_tokens"] == 24
        assert b["usage"]["completion_tokens"] > 0
        assert b["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}
        assert {"prompt_n", "prompt_ms", "prompt_per_second", "predicted_n",
                "predicted_ms", "predicted_per_second", "cache_n"} <= set(b["timings"])


async def test_nonstream_think_split():
    eng = FakeEngine(script="<think>why</think>because")
    async with serve(engine=eng) as client:
        b = await (await client.post("/v1/chat/completions",
                                   json={"messages": USER})).json()
        m = b["choices"][0]["message"]
        assert m["reasoning_content"] == "why"
        assert m["content"] == "because"
        assert "<think>" not in m["content"] and "</think>" not in m["content"]


async def test_model_field_echoed():
    async with serve() as client:
        b = await (await client.post("/v1/chat/completions",
                                   json={"messages": USER, "model": "whatever"})).json()
        assert b["model"] == "whatever"
        ev = await sse(client, {"messages": USER, "stream": True, "model": "whatever"})
        assert ev[0]["model"] == "whatever"


# ---------------------------------------------------------------- streaming shape

async def test_stream_ends_with_done():
    async with serve() as client:
        ev = await sse(client, {"messages": USER, "stream": True})
        assert ev[-1] == "[DONE]"
        final = final_chunk(ev)
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final["choices"][0]["delta"] == {}
        first = ev[0]
        assert first["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        assert first["object"] == "chat.completion.chunk"
        assert first["id"].startswith("chatcmpl-") and isinstance(first["id_slot"], int)


async def test_midstream_error_then_done():
    eng = FakeEngine(script="D" * 30, fail_on_token=3)
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True})
        assert ev[-1] == "[DONE]"
        errs = [c for c in ev if isinstance(c, dict) and "error" in c]
        assert len(errs) == 1
        assert errs[0]["error"]["type"] == "server_error"
    eng2 = FakeEngine(script="D" * 30, fail_on_token=3)
    async with serve(engine=eng2) as client:
        r = await client.post("/v1/chat/completions", json={"messages": USER})
        assert r.status == 500
        assert (await r.json())["error"]["type"] == "server_error"


async def test_health_during_generation_is_fast():
    eng = FakeEngine(script="E" * 80, delay=0.05)
    async with serve(engine=eng) as client:
        resp = await client.post("/v1/chat/completions",
                                 json={"messages": USER, "stream": True})
        assert resp.status == 200
        await resp.content.readline()  # wait for the first SSE event
        t0 = time.monotonic()
        h = await client.get("/health")
        dt = time.monotonic() - t0
        assert h.status == 200 and await h.json() == {"status": "ok"}
        assert dt < 0.1  # loop must stay responsive during 50 ms/token decode
        resp.close()


# ---------------------------------------------------------------- think split

async def test_think_split_char_by_char():
    # the fake engine emits one character per token, so both tags necessarily
    # arrive split across token boundaries
    eng = FakeEngine(script="<think>abc</think>def")
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True})
        assert reasoning_of(ev) == "abc"
        assert content_of(ev) == "def"
        assert "<think>" not in content_of(ev)
        assert "</think>" not in reasoning_of(ev) + content_of(ev)


async def test_think_split_unterminated():
    eng = FakeEngine(script="<think>never closed")
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True})
        final = final_chunk(ev)
        assert reasoning_of(ev) == "never closed"
        assert content_of(ev) == ""
        assert final["choices"][0]["finish_reason"] == "stop"


async def test_no_think_split_flag():
    eng = FakeEngine(script="<think>abc</think>def")
    async with serve(engine=eng, no_think_split=True) as client:
        ev = await sse(client, {"messages": USER, "stream": True})
        assert content_of(ev) == "<think>abc</think>def"
        assert reasoning_of(ev) == ""
        assert all("reasoning_content" not in c["choices"][0]["delta"]
                   for c in ev if isinstance(c, dict) and c.get("choices"))


async def test_think_split_ignores_leading_whitespace_prefix():
    eng = FakeEngine(script="  <think>r</think>c")
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True})
        assert reasoning_of(ev) == "r"
        assert content_of(ev) == "c"


# ---------------------------------------------------------------- timings

async def test_final_timings_from_engine_not_wall_clock():
    eng = FakeEngine(script="A" * 10)
    async with serve(engine=eng) as client:
        t0 = time.monotonic()
        ev = await sse(client, {"messages": USER, "stream": True,
                                "timings_per_token": True})
        elapsed = time.monotonic() - t0
        t = final_chunk(ev)["timings"]
        assert t["predicted_n"] == 10
        assert t["predicted_ms"] == pytest.approx(10 * 0.04 * 1000)  # 400, engine
        assert t["prompt_n"] == 24
        assert t["prompt_ms"] == pytest.approx(500.0)
        assert t["prompt_per_second"] == pytest.approx(48.0)
        assert t["predicted_per_second"] == pytest.approx(25.0)
        assert t["cache_n"] == 0
        assert "draft_n" not in t and "draft_n_accepted" not in t
        assert elapsed < 10 * 0.04  # wall time is nowhere near the engine figure


async def test_draft_timings_only_with_draft_stats():
    eng = FakeEngine(script="B" * 6, draft_stats=(3, 1))
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True,
                                "timings_per_token": True})
        t = final_chunk(ev)["timings"]
        assert t["draft_n"] == 4
        assert t["draft_n_accepted"] == 3
        provisional = [c for c in ev if isinstance(c, dict) and c.get("choices")
                       and "timings" in c and c["choices"][0]["finish_reason"] is None]
        assert provisional
        assert all("draft_n" not in c["timings"] for c in provisional)


async def test_timings_per_token_provisional_then_final():
    eng = FakeEngine(script="C" * 8, delay=0.01)
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True,
                                "timings_per_token": True})
        timed = [c for c in ev if isinstance(c, dict) and c.get("choices")
                 and "timings" in c]
        assert len(timed) == 9  # 8 content chunks + final
        provs, final = timed[:-1], timed[-1]
        assert all(p["timings"]["prompt_ms"] == 0.0 for p in provs)  # unknown until eos
        assert all(0.0 <= p["timings"]["predicted_ms"] < 200.0 for p in provs)  # wall clock
        assert final["timings"]["predicted_ms"] == pytest.approx(8 * 0.04 * 1000)
        assert final["timings"]["prompt_ms"] == pytest.approx(500.0)  # replaced


async def test_final_chunk_always_carries_timings():
    eng = FakeEngine(script="C" * 5)
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True})  # no per-token
        final = final_chunk(ev)
        assert "timings" in final
        assert final["timings"]["predicted_n"] == 5
        others = [c for c in ev if isinstance(c, dict) and c is not final]
        assert all("timings" not in c for c in others)


# ---------------------------------------------------------------- progress / usage

async def test_prompt_progress_chunks():
    async with serve() as client:
        ev = await sse(client, {"messages": [{"role": "user",
                                              "content": "hello progress world"}],
                                "stream": True, "return_progress": True})
        progs = [c for c in ev if isinstance(c, dict) and "prompt_progress" in c]
        assert len(progs) == 2
        assert all("choices" not in c for c in progs)
        # rendered prompt "<|user|>\nhello progress world\n<|assistant|>\n"
        # splits into 5 whitespace words
        assert progs[0]["prompt_progress"]["total"] == 5
        assert progs[0]["prompt_progress"]["processed"] == 0
        assert progs[0]["prompt_progress"]["cache"] == 0
        assert progs[1]["prompt_progress"]["processed"] == 5
        assert progs[1]["prompt_progress"]["time_ms"] >= progs[0]["prompt_progress"]["time_ms"]
        first_content = next(
            i for i, c in enumerate(ev)
            if isinstance(c, dict) and c.get("choices")
            and (c["choices"][0]["delta"].get("content")
                 or "reasoning_content" in c["choices"][0]["delta"]))
        assert ev.index(progs[0]) == 1  # right after the role chunk
        assert ev.index(progs[1]) < first_content


async def test_usage_on_final_chunk():
    eng = FakeEngine(script="F" * 7)
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True,
                                "stream_options": {"include_usage": True}})
        final = final_chunk(ev)
        u = final["usage"]
        assert u["prompt_tokens"] == 24 and u["completion_tokens"] == 7
        assert u["total_tokens"] == 31
        assert u["prompt_tokens_details"] == {"cached_tokens": 0}
        others = [c for c in ev if isinstance(c, dict) and c is not final]
        assert all("usage" not in c for c in others)
    eng2 = FakeEngine(script="F" * 7)
    async with serve(engine=eng2) as client:
        ev = await sse(client, {"messages": USER, "stream": True})
        assert all("usage" not in c for c in ev if isinstance(c, dict))


# ---------------------------------------------------------------- limits / clamps

async def test_length_finish():
    eng = FakeEngine(script="G" * 20)
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True, "max_tokens": 5,
                                "stream_options": {"include_usage": True}})
        final = final_chunk(ev)
        assert final["choices"][0]["finish_reason"] == "length"
        assert final["usage"]["completion_tokens"] == 5
        assert final["timings"]["predicted_n"] == 5


async def test_n_predict_alias():
    eng = FakeEngine(script="G" * 20)
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True, "n_predict": 3})
        assert content_of(ev) == "GGG"
        assert final_chunk(ev)["choices"][0]["finish_reason"] == "length"
    eng2 = FakeEngine(script="G" * 20)
    async with serve(engine=eng2) as client:
        # n_predict wins when both are present (llama-server semantics)
        ev = await sse(client, {"messages": USER, "stream": True,
                                "max_tokens": 5, "n_predict": 3})
        assert content_of(ev) == "GGG"


async def test_max_tokens_clamped():
    eng = FakeEngine(script="H" * 50)
    async with serve(engine=eng) as client:
        await sse(client, {"messages": USER, "stream": True, "max_tokens": 10 ** 9})
        assert eng.records[-1]["max_new_tokens"] == 4096  # --max-tokens-cap default
    eng2 = FakeEngine(script="H" * 50)
    async with serve(engine=eng2, max_tokens_cap=8) as client:
        await sse(client, {"messages": USER, "stream": True, "max_tokens": 10 ** 9})
        assert eng2.records[-1]["max_new_tokens"] == 8


async def test_temperature_routing():
    eng = FakeEngine()
    async with serve(engine=eng) as client:
        await client.post("/v1/chat/completions", json={"messages": USER})
        assert eng.records[-1]["temperature"] is None  # absent -> greedy
        await client.post("/v1/chat/completions",
                          json={"messages": USER, "temperature": 0})
        assert eng.records[-1]["temperature"] is None  # 0 -> greedy
        await client.post("/v1/chat/completions",
                          json={"messages": USER, "temperature": 0.7})
        assert eng.records[-1]["temperature"] == 0.7


# ---------------------------------------------------------------- stop strings

async def test_stop_string_truncates():
    eng = FakeEngine(script="keep this STOP drop this")
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True, "stop": ["STOP"]})
        assert content_of(ev) == "keep this "
        final = final_chunk(ev)
        assert final["choices"][0]["finish_reason"] == "stop"
        # the fake engine still reports figures after the cancel: 14 tokens
        # were generated ("keep this STOP" is 14 chars) before the match
        assert final["timings"]["predicted_n"] == 14
        assert ev[-1] == "[DONE]"


async def test_stop_string_across_token_boundary():
    # one char per token: "ST" and "OP" necessarily land in different deltas
    eng = FakeEngine(script="abcSTOPdef")
    async with serve(engine=eng) as client:
        ev = await sse(client, {"messages": USER, "stream": True, "stop": ["STOP"]})
        assert content_of(ev) == "abc"
        assert final_chunk(ev)["choices"][0]["finish_reason"] == "stop"
        # multiple stop strings: the earliest match wins
    eng2 = FakeEngine(script="one TWO three FOUR four")
    async with serve(engine=eng2) as client:
        ev = await sse(client, {"messages": USER, "stream": True,
                                "stop": ["FOUR", "TWO"]})
        assert content_of(ev) == "one "


# ---------------------------------------------------------------- 3-class defence

async def test_bad_bodies_rejected():
    async with serve() as client:
        url = "/v1/chat/completions"
        r = await client.post(url, data="{nope")  # corrupted: invalid JSON
        assert r.status == 400
        e = (await r.json())["error"]
        assert e["code"] == 400 and e["type"] == "invalid_request_error"
        r = await client.post(url, json={"prompt": "hi"})  # stale schema
        assert r.status == 400
        assert "/completion" in (await r.json())["error"]["message"]
        r = await client.post(url, json={"messages": [{"role": "user", "content": 123}]})
        assert r.status == 400  # malicious: non-string content
        assert "content" in (await r.json())["error"]["message"]
        r = await client.post(url, json={"messages": "hi"})
        assert r.status == 400
        r = await client.post(url, json={"messages": USER, "max_tokens": 0})
        assert r.status == 400
        r = await client.post(url, json={"messages": USER, "max_tokens": -5})
        assert r.status == 400
        r = await client.post(url, json={"messages": USER, "stream": "yes"})
        assert r.status == 400
        r = await client.post(url, json={"messages": USER, "temperature": "hot"})
        assert r.status == 400
        r = await client.post(url, json={"messages": USER, "stop": [1]})
        assert r.status == 400
        r = await client.post(url, json=[{"role": "user", "content": "x"}])
        assert r.status == 400


# ---------------------------------------------------------------- concurrency

async def test_two_concurrent_streams_are_isolated():
    eng = FakeEngine(delay=0.005)
    async with serve(engine=eng) as client:

        async def one(topic):
            ev = await sse(client, {"messages": [{"role": "user", "content": topic}],
                                    "stream": True})
            return topic, content_of(ev), ev[-1] == "[DONE]"

        (t1, c1, ok1), (t2, c2, ok2) = await asyncio.gather(one("alpha"), one("beta"))
        assert ok1 and ok2
        assert t1 in c1 and t2 not in c1
        assert t2 in c2 and t1 not in c2
        assert eng.peak_active == 2  # genuinely batched, not serialized


async def test_third_request_waits_for_free_slot():
    eng = FakeEngine(delay=0.01)
    async with serve(engine=eng, parallel=2) as client:
        done = []

        async def one(topic):
            ev = await sse(client, {"messages": [{"role": "user", "content": topic}],
                                    "stream": True})
            assert ev[-1] == "[DONE]"
            done.append(topic)

        await asyncio.gather(*[asyncio.create_task(one(t)) for t in ("a", "b", "c")])
        assert len(done) == 3        # the third waited and completed, not rejected
        assert eng.peak_active == 2  # never more than --parallel jobs in flight


# ---------------------------------------------------------------- self-review defect classes

async def test_client_disconnect_frees_slot_and_cancels_job():
    eng = FakeEngine(script="I" * 100, delay=0.02)
    async with serve(engine=eng) as client:
        resp = await client.post("/v1/chat/completions",
                                 json={"messages": USER, "stream": True})
        assert resp.status == 200
        await resp.content.readline()
        resp.close()
        for _ in range(250):  # cleanup is asynchronous
            if eng.num_remaining_jobs() == 0:
                break
            await asyncio.sleep(0.02)
        assert eng.num_remaining_jobs() == 0
        s = await (await client.get("/slots")).json()
        assert all(x["is_processing"] is False for x in s)
        ev = await sse(client, {"messages": [{"role": "user", "content": "after"}],
                                "stream": True})
        assert ev[-1] == "[DONE]"  # slot freed: the server still serves
        await asyncio.sleep(0.05)  # let the fire-and-forget cleanup settle


# ---------------------------------------------------------------- packaging

async def test_engine_exl3_compiles():
    import py_compile
    path = Path(__import__("exl3_serve").__file__).parent / "engine_exl3.py"
    py_compile.compile(str(path), doraise=True)
    assert path.exists()


async def test_imports_without_exllamav3():
    code = (
        "import sys, exl3_serve, exl3_serve.server, exl3_serve.cli, "
        "exl3_serve.engine_fake, exl3_serve.engine; "
        "assert 'exllamav3' not in sys.modules, 'exllamav3 leaked into imports'; "
        "print('IMPORT_OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=str(REPO_ROOT))
    assert r.returncode == 0, r.stderr
    assert "IMPORT_OK" in r.stdout
