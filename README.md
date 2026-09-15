# exl3-serve

A llama-server-compatible HTTP front for [ExLlamaV3](https://github.com/turboderp-org/exllamav3).

ExLlamaV3 ships no server. This serves one EXL3 model with the surface
llama.cpp's `llama-server` exposes — `/props`, `/health`, `/slots`,
`/v1/chat/completions` (SSE) with the `timings` object filled from
exllamav3's own job results — so tooling written against llama-server,
[toktape](https://github.com/midagedev/toktape) first, drives an EXL3 model
unchanged and records server-timed numbers.

Why Python: exllamav3 is a Python library with CUDA extensions and no C API or
process protocol, so the model has to live in a Python process. This module
is that process plus about two hundred lines of HTTP; anything larger belongs
outside it.

Status: first cut. See `docs/` once it exists.

## Usage

    python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
    .venv/bin/exl3-serve -m /path/to/model --host 0.0.0.0 --port 8080 --parallel 2

Every exllamav3 `model_init` flag is accepted verbatim (`-gs`, `-mcs`, `-mcl`,
`-mct`, `-cs`, `-mtp`, ...), exactly as in `reference/exl3-bench3.py`. The
server flags are:

- `--parallel N` — max concurrent streams (slots), default 2. Requests beyond
  that wait for a slot; they are never rejected.
- `--alias NAME` — model name reported to clients (default: model dir basename).
- `--reasoning-effort` — server default for the chat-template kwarg (default
  `high`); requests override it via `chat_template_kwargs`.
- `--draft-n N` — MTP draft tokens per step. Defaults to **1** when `-mtp` is
  given (depth 1 measured best with CPU-offloaded experts on this project's
  target model). `--dyn-draft` enables dynamic draft tokens on the Generator.
- `--no-think-split` — send everything as `content`, no `reasoning_content`.
- `--max-tokens-cap` — clamp for request `max_tokens`/`n_predict` (default 4096).

No GPU? `exl3-serve --fake -m /tmp/fake-model` serves a deterministic fake
engine implementing the same surface — that is what the test suite drives and
how the HTTP layer is smoke-tested offline.

## The toktape-facing surface

- `GET /props` — `model_path`, `chat_template` (raw contents of
  `<model_dir>/chat_template.jinja`), `total_slots`, and
  `default_generation_settings.n_ctx` (cache size in tokens). **No
  `build_info`**, on purpose: its presence makes toktape label the engine
  "llama-server bNNNN". The engine identifies itself in the
  `Server: exl3-serve/<version> exllamav3/<version or "unknown">` header
  instead, on every response.
- `GET /health` — `{"status":"ok"}` when loaded; 503 with the loading error
  shape while loading.
- `GET /slots` — one entry per slot: `id`, `is_processing`, `n_ctx`, `n_past`
  (tokens in that slot's context while busy, 0 when free).
- `POST /apply-template` — `{"messages":[...]}` → `{"prompt": "<rendered>"}`.
- `POST /v1/chat/completions` — honoured fields: `messages`, `stream`,
  `max_tokens` (default 512) with `n_predict` as an alias that wins when both
  are given, `temperature` (0 or absent → greedy; otherwise
  `exllamav3.DefaultSampler(temperature=…)` when this install has it, else
  greedy), `model` (ignored, echoed back), `timings_per_token`,
  `return_progress`, `stream_options.include_usage`, `chat_template_kwargs`
  (merged into the render kwargs; `reasoning_effort` comes from here or the
  server default), `stop` (list of strings).
- Streaming: `text/event-stream`, `data: <json>\n\n` events terminated by
  `data: [DONE]\n\n` — including after a mid-stream error (error chunk, then
  `[DONE]`). First chunk carries `delta: {"role":"assistant","content":""}`;
  chunks carry `id_slot`; the final chunk carries `finish_reason`
  (`"stop"`/`"length"`), empty `delta`, `timings`, and (with
  `include_usage`) `usage`.
- Thinking: text inside a leading `<think>…</think>` streams as
  `delta.reasoning_content`, the rest as `delta.content`; tags arriving split
  across token boundaries are handled by buffering. An unterminated think
  block flushes as reasoning at finish.
- `timings` object (on the final chunk always; on content chunks too when
  `timings_per_token`): `prompt_n`, `prompt_ms`, `prompt_per_second`,
  `predicted_n`, `predicted_ms`, `predicted_per_second`, `cache_n`, plus
  `draft_n`/`draft_n_accepted` only when the engine reports draft counters.
  Final-chunk figures come from the engine's own eos result — never from wall
  clock. Per-token chunks carry provisional wall-clock values until then.
- Malformed bodies get 400
  `{"error":{"code":400,"message":"...","type":"invalid_request_error"}}`;
  a `/completion`-style `prompt` field is rejected with a message naming
  `/completion` as unsupported.

## Limitations (first cut)

- No prefix cache: `cache_n` is always 0, `cached_tokens` is 0, and a freed
  slot reports `n_past: 0`.
- exllamav3 reports prefill only at completion, so `return_progress` sends
  exactly two prompt-progress chunks: `processed=0` before generation, and
  `processed=total` when the first token arrives.
- When a `stop` string cancels a job and the engine produces no final result
  (exllamav3's cancel path), the final chunk falls back to the provisional
  wall-clock timings.
- `messages[*].content` must be a string or null — no multimodal parts.
- One engine thread: prompt encoding waits for in-flight decode steps.
- While the model is loading, `/props`, `/slots` and `/v1/chat/completions`
  return 503 with the loading error shape.
- Huge `max_tokens` is clamped to `--max-tokens-cap` (4096 by default) rather
  than rejected.

