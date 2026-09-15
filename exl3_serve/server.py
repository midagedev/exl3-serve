"""HTTP layer: an aiohttp app presenting the llama-server surface toktape expects.

Concurrency model (stated, per the contract): ONE engine, ONE dispatch task.
Requests are enqueued as engine jobs; a single asyncio task drives
``engine.iterate()`` one call at a time inside a one-worker thread pool
(``loop.run_in_executor``), so the event loop is never blocked by the engine,
and every engine call is serialized through that task -- cheap calls (submit,
cancel, render) run on the loop between iterations, blocking calls (iterate,
encode) run in the worker thread. Results are routed to per-request queues by
job identity. Slots limit concurrency to ``--parallel``; an arriving request
with no free slot waits (it is never rejected).
"""
from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from aiohttp import web

from . import __version__


# --------------------------------------------------------------------------
# think-split: token-level <think>...</think> handling
# --------------------------------------------------------------------------

class ThinkSplitter:
    """Splits generated tokens into reasoning (inside ``<think>...</think>``)
    and content. The tags are single tokens for this model class, so the
    split works per token and nothing is buffered: a token whose stripped
    text is exactly a tag switches mode and emits nothing, and a token with
    an embedded ``</think>`` emits one delta carrying both sides.

    The splitter starts in think mode when the rendered prompt ends with
    ``<think>`` (this model's template opens the think block itself, and the
    reply closes it later without ever emitting ``<think>``)."""

    START = "<think>"
    END = "</think>"

    def __init__(self, prompt: str = "") -> None:
        self.mode = "think" if prompt.rstrip().endswith(self.START) else "start"

    def feed_token(self, text: str) -> Optional[dict]:
        """One token's text -> its delta dict, or None when the token emits
        nothing (a pure tag token). Only non-empty sides are included."""
        stripped = text.strip()
        if stripped == self.START:
            self.mode = "think"
            return None
        if stripped == self.END:
            if self.mode == "think":
                self.mode = "content"
            return None  # a stray closer outside a think block emits nothing
        if self.mode == "think":
            if self.END in text:
                before, after = text.split(self.END, 1)
                self.mode = "content"
                delta: dict = {}
                if before:
                    delta["reasoning_content"] = before
                if after:
                    delta["content"] = after
                return delta or None
            return {"reasoning_content": text}
        if self.mode == "start":
            self.mode = "content"
        return {"content": text}


def timings_from_engine(res: dict) -> dict:
    """The timings object, from the engine's final (eos) result.

    llama-server counts only the newly prefilled tokens: ``prompt_n`` is
    ``prompt_tokens`` minus ``cached_tokens`` and ``prompt_per_second`` is
    over that. A missing ``cached_tokens`` counts as 0 in the arithmetic and
    is omitted from the output rather than zeroed.
    """
    cached = res.get("cached_tokens")
    prompt_n = (res.get("prompt_tokens") or 0) - (cached or 0)
    prompt_s = res.get("time_prefill") or 0.0
    pred_n = res.get("new_tokens") or 0
    pred_s = res.get("time_generate") or 0.0
    t = {
        "prompt_n": prompt_n,
        "prompt_ms": prompt_s * 1000.0,
        "prompt_per_second": (prompt_n / prompt_s) if prompt_s > 0 else 0.0,
        "predicted_n": pred_n,
        "predicted_ms": pred_s * 1000.0,
        "predicted_per_second": (pred_n / pred_s) if pred_s > 0 else 0.0,
    }
    if cached is not None:
        t["cache_n"] = cached
    accepted = res.get("accepted_draft_tokens")
    rejected = res.get("rejected_draft_tokens")
    if accepted is not None and rejected is not None:
        t["draft_n"] = accepted + rejected
        t["draft_n_accepted"] = accepted
    return t


# --------------------------------------------------------------------------
# dispatch: one serialized owner for the engine
# --------------------------------------------------------------------------

@dataclass
class GenRecord:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    job: Any = None


class Dispatcher:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.routes: dict[Any, GenRecord] = {}
        self._cmds: asyncio.Queue = asyncio.Queue()
        self._wake = asyncio.Event()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="exl3-engine")
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name="exl3-dispatcher")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._pool.shutdown(wait=False)

    async def _call(self, fn: Callable[[], Any]) -> Any:
        fut = asyncio.get_running_loop().create_future()
        self._cmds.put_nowait((fn, fut))
        self._wake.set()
        return await fut

    async def encode(self, text: str) -> list[int]:
        return await self._call(lambda: self.engine.encode(text))

    async def submit(self, rec: GenRecord, ids: list[int], max_new_tokens: int,
                     sampling: dict, stop: Optional[list[str]]) -> Any:
        def _do() -> Any:
            job = self.engine.submit(ids, max_new_tokens, sampling, stop)
            rec.job = job
            self.routes[job] = rec  # registered atomically with the enqueue
            return job
        return await self._call(_do)

    async def cancel(self, job: Any) -> None:
        if job is not None:
            await self._call(lambda: self.engine.cancel(job))

    async def purge(self, job: Any) -> None:
        if job is not None:
            await self._call(lambda: self.routes.pop(job, None))

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            self._wake.clear()
            while not self._cmds.empty():
                fn, fut = self._cmds.get_nowait()
                try:
                    fut.set_result(fn())
                except Exception as exc:  # surfaced to the awaiting handler
                    fut.set_exception(exc)
            if self.engine.num_remaining_jobs() > 0:
                try:
                    results = await loop.run_in_executor(self._pool, self.engine.iterate)
                except Exception as exc:  # engine died mid-iteration
                    self._fail_all(exc)
                    continue
                for res in results or []:
                    self._route(res)
                await asyncio.sleep(0)
            else:
                await self._wake.wait()

    def _route(self, res: dict) -> None:
        job = res.get("job")
        rec = self.routes.get(job) if job is not None else None
        if rec is None:
            return
        rec.queue.put_nowait(res)
        if res.get("eos"):
            self.routes.pop(job, None)

    def _fail_all(self, exc: Exception) -> None:
        for job, rec in list(self.routes.items()):
            rec.queue.put_nowait(
                {"job": job, "engine_error": f"{type(exc).__name__}: {exc}"})
        self.routes.clear()


# --------------------------------------------------------------------------
# app state and options
# --------------------------------------------------------------------------

@dataclass
class Options:
    parallel: int = 2
    alias: Optional[str] = None
    reasoning_effort: str = "high"
    no_think_split: bool = False
    max_tokens_cap: int = 4096


@dataclass
class Slot:
    id: int
    n_ctx: int
    busy: bool = False
    n_past: int = 0


def _error(status: int, message: str, type_: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"code": status, "message": message, "type": type_}}, status=status)


def _loading_response(st: "AppState") -> web.Response:
    if st.load_error:
        return web.json_response(
            {"error": {"message": f"Model load failed: {st.load_error}"}}, status=503)
    return web.json_response({"error": {"message": "Loading model"}}, status=503)


class AppState:
    def __init__(self, options: Options, engine: Any = None,
                 loader: Optional[Callable[[], Any]] = None) -> None:
        self.options = options
        self.engine = engine
        self.loader = loader
        self.load_error: Optional[str] = None
        self.dispatcher: Optional[Dispatcher] = None
        self.slots: list[Slot] = []
        self.slot_sem: Optional[asyncio.Semaphore] = None
        self.alias = options.alias or "exl3"
        self.model_path = ""
        self.chat_template = ""
        self.exl3_version: Optional[str] = None
        self.engine_block: Optional[dict] = None
        self.n_ctx = 0
        self._id_counter = itertools.count(1)
        self._load_task: Optional[asyncio.Task] = None

    # -- lifecycle ----------------------------------------------------------

    async def on_startup(self, app: web.Application) -> None:
        if self.engine is not None:
            self._ready(self.engine)
        elif self.loader is not None:
            self._load_task = asyncio.get_running_loop().create_task(
                self._load(), name="exl3-load")

    async def _load(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            engine = await loop.run_in_executor(None, self.loader)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            return
        self._ready(engine)

    def _ready(self, engine: Any) -> None:
        self.engine = engine
        self.dispatcher = Dispatcher(engine)
        self.dispatcher.start()
        props = engine.props
        self.model_path = props.get("model_path", "")
        self.chat_template = props.get("chat_template") or ""
        self.exl3_version = props.get("exllamav3_version")
        self.engine_block = props.get("engine")
        try:
            self.n_ctx = int(props.get("n_ctx") or 0)
        except (TypeError, ValueError):
            self.n_ctx = 0
        self.slots = [Slot(i, self.n_ctx) for i in range(self.options.parallel)]
        self.slot_sem = asyncio.Semaphore(self.options.parallel)
        if self.options.alias:
            self.alias = self.options.alias
        elif self.model_path:
            self.alias = os.path.basename(self.model_path.rstrip("/")) or "exl3"

    async def on_cleanup(self, app: web.Application) -> None:
        if self._load_task is not None:
            self._load_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._load_task
        if self.dispatcher is not None:
            await self.dispatcher.stop()

    # -- slots ----------------------------------------------------------------

    async def acquire_slot(self) -> Slot:
        await self.slot_sem.acquire()
        for s in self.slots:
            if not s.busy:
                s.busy = True
                s.n_past = 0
                return s
        raise RuntimeError("no free slot despite semaphore")  # unreachable

    def release_slot(self, slot: Slot) -> None:
        slot.busy = False
        slot.n_past = 0
        self.slot_sem.release()

    def next_id(self) -> str:
        return f"chatcmpl-{next(self._id_counter)}"

    # -- simple endpoints -------------------------------------------------------

    async def handle_props(self, request: web.Request) -> web.Response:
        if self.engine is None:
            return _loading_response(self)
        body = {
            # no build_info on purpose: toktape would mislabel us "llama-server
            # bNNNN"; the engine identifies itself in the Server header
            "model_path": self.model_path,
            "chat_template": self.chat_template,
            "total_slots": self.options.parallel,
            "default_generation_settings": {"n_ctx": self.n_ctx},
        }
        if self.engine_block is not None:
            body["engine"] = self.engine_block
        return web.json_response(body)

    async def handle_health(self, request: web.Request) -> web.Response:
        if self.engine is None:
            return _loading_response(self)
        return web.json_response({"status": "ok"})

    async def handle_slots(self, request: web.Request) -> web.Response:
        if self.engine is None:
            return _loading_response(self)
        return web.json_response([
            {"id": s.id, "is_processing": s.busy, "n_ctx": s.n_ctx, "n_past": s.n_past}
            for s in self.slots
        ])

    async def handle_apply_template(self, request: web.Request) -> web.Response:
        if self.engine is None:
            return _loading_response(self)
        try:
            body = await request.json()
        except Exception:
            return _error(400, "Request body is not valid JSON.")
        if not isinstance(body, dict):
            return _error(400, "Request body must be a JSON object.")
        messages, err = _validate_messages(body)
        if err:
            return _error(400, err)
        ctk = body.get("chat_template_kwargs")
        if ctk is not None and not isinstance(ctk, dict):
            return _error(400, "'chat_template_kwargs' must be an object.")
        kw = {
            "messages": messages,
            "add_generation_prompt": True,
            "reasoning_effort": (ctk or {}).get("reasoning_effort") or self.options.reasoning_effort,
            "tools": None,
        }
        kw.update(ctk or {})
        kw["messages"] = messages          # not overridable by the client
        kw["add_generation_prompt"] = True
        prompt = self.engine.render_chat(**kw)
        return web.json_response({"prompt": prompt})

    # -- chat completions ---------------------------------------------------------

    async def handle_chat(self, request: web.Request) -> web.StreamResponse:
        if self.engine is None:
            return _loading_response(self)
        try:
            body = await request.json()
        except Exception:
            return _error(400, "Request body is not valid JSON.")
        params, err = _parse_chat_request(body, self.options)
        if err:
            return _error(400, err)
        slot = await self.acquire_slot()
        try:
            return await self._run_chat(request, slot, params)
        finally:
            self.release_slot(slot)

    async def _run_chat(self, request: web.Request, slot: Slot, params: dict) -> web.StreamResponse:
        ctk = params["chat_template_kwargs"]
        render_kw = {
            "messages": params["messages"],
            "add_generation_prompt": True,
            "reasoning_effort": ctk.get("reasoning_effort") or self.options.reasoning_effort,
            "tools": None,
        }
        render_kw.update(ctk)
        render_kw["messages"] = params["messages"]  # not overridable by the client
        render_kw["add_generation_prompt"] = True
        prompt = self.engine.render_chat(**render_kw)
        ids = await self.dispatcher.encode(prompt)
        gen = _Generation(self, slot, params, ids, prompt)
        if params["stream"]:
            return await gen.stream(request)
        return await gen.collect()


# --------------------------------------------------------------------------
# request validation (3-class input defence lives here)
# --------------------------------------------------------------------------

def _validate_messages(body: dict) -> tuple[Optional[list], Optional[str]]:
    if "messages" not in body:
        if "prompt" in body:
            return None, ("This endpoint takes chat 'messages'; the '/completion'-style "
                          "'prompt' field is not supported.")
        return None, "'messages' is required."
    messages = body["messages"]
    if not isinstance(messages, list) or not messages:
        return None, "'messages' must be a non-empty list."
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return None, f"messages[{i}] must be an object."
        role = m.get("role")
        if not isinstance(role, str) or not role:
            return None, f"messages[{i}].role must be a non-empty string."
        content = m.get("content")
        if content is not None and not isinstance(content, str):
            return None, f"messages[{i}].content must be a string or null."
    return messages, None


def _parse_chat_request(body: dict, options: Options) -> tuple[Optional[dict], Optional[str]]:
    if not isinstance(body, dict):
        return None, "Request body must be a JSON object."
    messages, err = _validate_messages(body)
    if err:
        return None, err
    p: dict = {"messages": messages}

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        return None, "'stream' must be a boolean."
    p["stream"] = stream

    # n_predict is llama-server's alias for max_tokens and wins when both are given
    raw_max = body.get("n_predict") if "n_predict" in body else body.get("max_tokens", 512)
    if raw_max is None:
        raw_max = 512
    if isinstance(raw_max, bool) or not isinstance(raw_max, int):
        return None, "'max_tokens'/'n_predict' must be an integer."
    if raw_max < 1:
        return None, "'max_tokens'/'n_predict' must be >= 1."
    p["max_tokens"] = min(raw_max, options.max_tokens_cap)  # malicious-size clamp

    temperature = body.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            return None, "'temperature' must be a number."
        if temperature < 0:
            return None, "'temperature' must be >= 0."
        temperature = float(temperature)
    p["temperature"] = temperature or None  # absent or 0 -> greedy

    top_p = body.get("top_p")
    if top_p is not None:
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
            return None, "'top_p' must be a number."
        top_p = float(top_p)
        if not (0.0 < top_p <= 1.0):
            return None, "'top_p' must be in (0, 1]."
    p["top_p"] = top_p

    top_k = body.get("top_k")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            return None, "'top_k' must be an integer."
        if top_k < 0:
            return None, "'top_k' must be >= 0."
    p["top_k"] = top_k

    min_p = body.get("min_p")
    if min_p is not None:
        if isinstance(min_p, bool) or not isinstance(min_p, (int, float)):
            return None, "'min_p' must be a number."
        min_p = float(min_p)
        if not (0.0 <= min_p <= 1.0):
            return None, "'min_p' must be in [0, 1]."
    p["min_p"] = min_p

    stop = body.get("stop")
    if stop is not None and isinstance(stop, str):
        stop = [stop]
    if stop is not None and (not isinstance(stop, list)
                             or not all(isinstance(s, str) for s in stop)):
        return None, "'stop' must be a string or a list of strings."
    p["stop"] = stop or []

    for key in ("timings_per_token", "return_progress"):
        v = body.get(key, False)
        if not isinstance(v, bool):
            return None, f"'{key}' must be a boolean."
        p[key] = v

    so = body.get("stream_options")
    if so is not None:
        if not isinstance(so, dict):
            return None, "'stream_options' must be an object."
        iu = so.get("include_usage", False)
        if not isinstance(iu, bool):
            return None, "'stream_options.include_usage' must be a boolean."
        p["include_usage"] = iu
    else:
        p["include_usage"] = False

    ctk = body.get("chat_template_kwargs")
    if ctk is not None and not isinstance(ctk, dict):
        return None, "'chat_template_kwargs' must be an object."
    p["chat_template_kwargs"] = ctk or {}

    p["model"] = body.get("model") or None  # ignored, echoed back
    return p, None


# --------------------------------------------------------------------------
# one chat request's lifetime
# --------------------------------------------------------------------------

class _Generation:
    def __init__(self, st: AppState, slot: Slot, params: dict, ids: list[int],
                 prompt: str) -> None:
        self.st = st
        self.slot = slot
        self.params = params
        self.ids = ids
        self.rec = GenRecord()
        self.splitter: Optional[ThinkSplitter] = (
            None if st.options.no_think_split else ThinkSplitter(prompt))
        self.stop_strings: list[str] = params["stop"]
        self.max_tokens = params["max_tokens"]
        self._max_stop_len = max((len(s) for s in params["stop"]), default=0)
        self.raw = ""
        self.held: list[str] = []  # whole tokens held back for stop matching
        self.released_end = 0      # chars of self.raw already emitted
        self.stopped = False       # a stop string matched
        self._last_cache = 0       # last engine-reported cached prompt length
        self.emitted = 0
        self.first_token_at: Optional[float] = None
        self.started_at = time.monotonic()
        self.submitted_at: Optional[float] = None  # when the job reached the engine
        self.figures: Optional[dict] = None  # the engine's final result
        self.engine_error: Optional[str] = None
        self.model = params["model"] or st.alias
        self.id = st.next_id()
        self.created = int(time.time())
        self.reasoning_parts: list[str] = []
        self.content_parts: list[str] = []

    # -- token plumbing --------------------------------------------------------

    def _stop_match(self) -> int:
        best = -1
        for s in self.stop_strings:
            p = self.raw.find(s)
            if p >= 0 and (best < 0 or p < best):
                best = p
        return best

    def _token_deltas(self, tok: str) -> list[dict]:
        """One token -> at most one delta dict (carrying both think sides
        when the token straddles the closing tag)."""
        if self.splitter is None:
            return [{"content": tok}] if tok else []
        delta = self.splitter.feed_token(tok)
        return [delta] if delta else []

    def _process_token(self, text: str) -> list[dict]:
        """Feed one token's text; return the deltas to emit.

        The stop-string hold-back releases **whole tokens**: a token stays
        queued until no stop string starting inside it can still complete
        (its end is followed by max_stop_len - 1 characters of raw text).
        When a stop string matches, everything before it is emitted -- the
        queued whole tokens, then the straddling token's prefix as one chunk
        -- and the rest is dropped.
        """
        if self.stopped or not text:
            return []
        deltas: list[dict] = []
        self.raw += text
        if self.stop_strings:
            pos = self._stop_match()
            if pos >= 0:
                self.stopped = True
                prefix = self.raw[:pos]
                while self.held and self.released_end + len(self.held[0]) <= len(prefix):
                    tok = self.held.pop(0)
                    self.released_end += len(tok)
                    deltas.extend(self._token_deltas(tok))
                rem = prefix[self.released_end:]
                if rem:
                    deltas.extend(self._token_deltas(rem))
                return deltas
        self.held.append(text)
        while self.held and (self._max_stop_len <= 1
                             or self.released_end + len(self.held[0]) - 1
                             + self._max_stop_len <= len(self.raw)):
            tok = self.held.pop(0)
            self.released_end += len(tok)
            deltas.extend(self._token_deltas(tok))
        return deltas

    def _flush_held(self) -> list[dict]:
        """Emit the queued whole tokens at finish time (no stop matched).

        After a stop match the straddling token is still queued -- only its
        pre-stop prefix was emitted -- and must stay dropped, not flushed.
        """
        if self.stopped:
            return []
        deltas: list[dict] = []
        while self.held:
            tok = self.held.pop(0)
            self.released_end += len(tok)
            deltas.extend(self._token_deltas(tok))
        return deltas

    # -- timings / usage ---------------------------------------------------------

    def _provisional_timings(self) -> dict:
        n = self.emitted
        per_s = 0.0
        ms = 0.0
        if n and self.first_token_at is not None:
            sec = max(time.monotonic() - self.first_token_at, 1e-9)
            ms = sec * 1000.0
            per_s = n / sec
        # The engine reports its own prefill time only in the eos result, so
        # until then -- and for good on a cancelled job -- the prefill figure is
        # this server's wall clock. It is measured from the moment the job
        # reached the engine, not from when the request arrived: a recorder
        # subtracts this from its TTFT to show the wait before prefill, so a
        # figure that contained that wait would be counted twice (measured
        # 2026-09-15; leaving it 0 was worse, it said a 274-token prefill took
        # no time).
        prompt_ms = 0.0
        prompt_per_s = 0.0
        prompt_n = len(self.ids)
        if self.first_token_at is not None and self.submitted_at is not None:
            prompt_ms = max(self.first_token_at - self.submitted_at, 0.0) * 1000.0
            if prompt_ms > 0.0:
                prompt_per_s = prompt_n / (prompt_ms / 1000.0)
        return {
            "prompt_n": prompt_n,       # measured at encode time
            "prompt_ms": prompt_ms,
            "prompt_per_second": prompt_per_s,
            "predicted_n": n,
            "predicted_ms": ms,
            "predicted_per_second": per_s,
        }

    def _final_timings(self) -> dict:
        if self.figures is not None:
            return timings_from_engine(self.figures)
        # stop-cancel path where the engine never produced a final result
        # (exllamav3's cancel); documented in README Limitations
        return self._provisional_timings()

    def _usage(self) -> dict:
        prompt_n = (self.figures or {}).get("prompt_tokens")
        if prompt_n is None:
            prompt_n = len(self.ids)
        completion_n = (self.figures or {}).get("new_tokens")
        if completion_n is None:
            completion_n = self.emitted
        usage = {
            "prompt_tokens": prompt_n,
            "completion_tokens": completion_n,
            "total_tokens": prompt_n + completion_n,
        }
        cached = (self.figures or {}).get("cached_tokens")
        if cached is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        return usage

    def _finish_reason(self) -> str:
        if self.stopped:
            return "stop"
        if (self.figures or {}).get("eos_reason") == "max_new_tokens":
            return "length"
        if self.figures is not None:
            return "stop"
        # no final result arrived (cancel path): fall back to the cap
        if self.emitted >= self.max_tokens:
            return "length"
        return "stop"

    def _chunk(self, delta: Optional[dict] = None, finish_reason: Optional[str] = None,
               extra: Optional[dict] = None) -> dict:
        obj = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "id_slot": self.slot.id,
            "choices": [{"index": 0, "delta": delta if delta is not None else {},
                         "finish_reason": finish_reason}],
        }
        if extra:
            obj.update(extra)
        return obj

    # -- the shared generation loop ---------------------------------------------

    async def _drive(self, emit: Callable) -> None:
        st = self.st
        p = self.params
        if p["return_progress"]:
            await emit("progress", 0, 0)  # the submission chunk
        sampling = {"temperature": p["temperature"], "top_p": p.get("top_p"),
                    "top_k": p.get("top_k"), "min_p": p.get("min_p")}
        job = await st.dispatcher.submit(
            self.rec, self.ids, self.max_tokens, sampling, p["stop"])
        self.submitted_at = time.monotonic()
        try:
            while True:
                res = await self.rec.queue.get()
                if "engine_error" in res:
                    self.engine_error = res["engine_error"]
                    return
                if (p["return_progress"] and self.first_token_at is None
                        and isinstance(res.get("curr_progress"), int)
                        and isinstance(res.get("max_progress"), int)):
                    total = len(self.ids)
                    cache = max(0, total - res["max_progress"])
                    self._last_cache = cache
                    await emit("progress", cache, cache + res["curr_progress"])
                text = res.get("text") or ""
                if text:
                    self.emitted += 1
                    self.slot.n_past = len(self.ids) + self.emitted
                    if self.first_token_at is None:
                        self.first_token_at = time.monotonic()
                        if p["return_progress"]:
                            # prefill is done once tokens flow: final chunk
                            await emit("progress", self._last_cache, len(self.ids))
                    for delta in self._process_token(text):
                        await emit("delta", delta)
                    if self.stopped:
                        if res.get("eos"):
                            self.figures = res
                            return
                        await st.dispatcher.cancel(job)
                        # the fake engine still reports final figures after a
                        # cancel; take them when they come
                        deadline = time.monotonic() + 1.0
                        while self.figures is None:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            try:
                                res2 = await asyncio.wait_for(
                                    self.rec.queue.get(), remaining)
                            except asyncio.TimeoutError:
                                break
                            if res2.get("eos"):
                                self.figures = res2
                        return
                if res.get("eos"):
                    self.figures = res
                    return
        finally:
            if self.figures is None and self.rec.job is not None:
                self._cleanup_bg()

    def _cleanup_bg(self) -> None:
        # fire-and-forget: this must also work from inside a cancelled handler
        d = self.st.dispatcher
        job = self.rec.job

        async def _c() -> None:
            try:
                await asyncio.wait_for(d.cancel(job), 1.0)
                await asyncio.wait_for(d.purge(job), 1.0)
            except Exception:
                pass

        try:
            asyncio.get_running_loop().create_task(_c(), name="exl3-cleanup")
        except RuntimeError:
            pass

    # -- streaming response ---------------------------------------------------

    async def stream(self, request: web.Request) -> web.StreamResponse:
        p = self.params
        resp = web.StreamResponse(headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await resp.prepare(request)

        async def send(obj: dict) -> None:
            await resp.write(b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n")

        async def emit(kind: str, *args: Any) -> None:
            if kind == "progress":
                cache, processed = args
                await send({
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "id_slot": self.slot.id,
                    "prompt_progress": {
                        "total": len(self.ids),
                        "cache": cache,
                        "processed": processed,
                        "time_ms": (time.monotonic() - self.started_at) * 1000.0,
                    },
                })
            else:  # "delta": exactly one chunk per emitted token
                (delta,) = args
                extra = ({"timings": self._provisional_timings()}
                         if p["timings_per_token"] else None)
                await send(self._chunk(delta=delta, extra=extra))

        await send(self._chunk(delta={"role": "assistant", "content": ""}))
        try:
            await self._drive(emit)
            if self.engine_error is not None:
                await send({"error": {"code": 500, "message": self.engine_error,
                                      "type": "server_error"}})
                await resp.write(b"data: [DONE]\n\n")
                return resp
            for delta in self._flush_held():
                extra = ({"timings": self._provisional_timings()}
                         if p["timings_per_token"] else None)
                await send(self._chunk(delta=delta, extra=extra))
            final = self._chunk(finish_reason=self._finish_reason(),
                                extra={"timings": self._final_timings()})
            if p["include_usage"]:
                final["usage"] = self._usage()
            await send(final)
            await resp.write(b"data: [DONE]\n\n")
        except asyncio.CancelledError:
            # never swallowed; _drive's finally owns the fire-and-forget
            # cleanup for the cancelled-handler configuration
            raise
        except ConnectionResetError:
            # the client went away mid-stream (production runs without
            # handler_cancellation, so nobody cancels this handler for us).
            # Cancel the job through the dispatcher -- the same call the
            # stop-sequence path makes -- and stop writing to the dead
            # socket: one log line, no traceback. The slot is released by
            # handle_chat's finally, exactly like a normal completion.
            job = self.rec.job
            if self.figures is None and job is not None:
                with contextlib.suppress(Exception):
                    await self.st.dispatcher.cancel(job)
                    await self.st.dispatcher.purge(job)
                print(f"exl3-serve: client gone after {self.emitted} tokens, "
                      "job cancelled", flush=True)
        return resp

    # -- blocking response ------------------------------------------------------

    async def collect(self) -> web.Response:
        async def emit(kind: str, *args: Any) -> None:
            if kind == "progress":
                return  # prompt_progress is a streaming-only feature
            (delta,) = args
            if "reasoning_content" in delta:
                self.reasoning_parts.append(delta["reasoning_content"])
            if "content" in delta:
                self.content_parts.append(delta["content"])

        await self._drive(emit)
        if self.engine_error is not None:
            return _error(500, self.engine_error, "server_error")
        for delta in self._flush_held():
            if "reasoning_content" in delta:
                self.reasoning_parts.append(delta["reasoning_content"])
            if "content" in delta:
                self.content_parts.append(delta["content"])
        message: dict = {"role": "assistant", "content": "".join(self.content_parts)}
        if self.reasoning_parts:
            message["reasoning_content"] = "".join(self.reasoning_parts)
        obj = {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": self._finish_reason()}],
            "usage": self._usage(),
            "timings": self._final_timings(),
        }
        return web.json_response(obj)


# --------------------------------------------------------------------------
# app assembly
# --------------------------------------------------------------------------

STATE_KEY = web.AppKey("exl3_state", AppState)


async def _server_header(request: web.Request, response: web.StreamResponse) -> None:
    st = request.app[STATE_KEY]
    response.headers["Server"] = f"exl3-serve/{__version__} exllamav3/{st.exl3_version or 'unknown'}"


def create_app(options: Optional[Options] = None, engine: Any = None,
               loader: Optional[Callable[[], Any]] = None) -> web.Application:
    app = web.Application()
    st = AppState(options or Options(), engine=engine, loader=loader)
    app[STATE_KEY] = st
    app.on_response_prepare.append(_server_header)
    app.on_startup.append(st.on_startup)
    app.on_cleanup.append(st.on_cleanup)
    app.router.add_get("/props", st.handle_props)
    app.router.add_get("/health", st.handle_health)
    app.router.add_get("/slots", st.handle_slots)
    app.router.add_post("/apply-template", st.handle_apply_template)
    app.router.add_post("/v1/chat/completions", st.handle_chat)
    return app
