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

Status: it serves, and it has been recorded. On a two-card workstation
(RTX A6000 48 GB + RTX 3090 24 GB, Threadripper PRO 5975WX, 252 GB DDR4-3600)
it served GLM-5.3-Flash EXL3 4.05 bpw — 154 GiB, per-expert placement with
`-mcs 185`, the MTP layer as its own draft — and toktape attached to it and
recorded 22.0 tok/s decode with the draft 96 % accepted, naming the engine
`exllamav3 1.5.0` rather than a llama-server build number. The numbers, the
tape and the clip are in that machine's log:
[rig-log 2026-09-15](https://github.com/midagedev/rig-log/blob/main/log/2026-09-15-exl3-serve-and-tabbyapi-timings.md).

## Usage

    python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
    .venv/bin/exl3-serve -m /path/to/model --host 0.0.0.0 --port 8080 --parallel 2

Every exllamav3 `model_init` flag is accepted verbatim (`-gs`, `-mcs`, `-mcl`,
`-mct`, `-cs`, `-mtp`, ...), exactly as in `reference/exl3-bench3.py`. The
server flags are:

- `--parallel N` — max concurrent streams (slots), default 2. Requests beyond
  that wait for a slot; they are never rejected. exllamav3 sizes every cache
  from `-ambs` (default 1) and the Generator clamps its batch to the cache's
  slot count, so `load` raises `-ambs` to at least N before `model_init` and
  then asserts the served batch: if it is still smaller the load fails with
  both numbers rather than serving N slots that decode one at a time. Two real
  slots need more VRAM than one — on a split with little headroom the load can
  fail where `--parallel 1` fits, and a smaller `-cs` is the lever.
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
  `<model_dir>/chat_template.jinja`), `total_slots`,
  `default_generation_settings.n_ctx` (cache size in tokens), and an `engine`
  object (below). **No `build_info`**, on purpose: its presence makes toktape
  label the engine "llama-server bNNNN". The engine identifies itself in the
  `Server: exl3-serve/<version> exllamav3/<version or "unknown">` header
  instead, on every response — the exllamav3 version there is
  `exllamav3.version.__version__` (e.g. `1.5.0`), not the wheel metadata.
- `GET /health` — `{"status":"ok"}` when loaded; 503 with the loading error
  shape while loading.
- `GET /slots` — one entry per slot: `id`, `is_processing`, `n_ctx`, `n_past`
  (tokens in that slot's context while busy, 0 when free).
- `POST /apply-template` — `{"messages":[...]}` → `{"prompt": "<rendered>"}`.
- `POST /v1/chat/completions` — honoured fields: `messages`, `stream`,
  `max_tokens` (default 512) with `n_predict` as an alias that wins when both
  are given, `temperature` (0 or absent → `GreedySampler()`; otherwise
  `ComboSampler(temperature=…, top_p=…, top_k=…, min_p=…)`), `top_p` in
  (0, 1] (falls back to `generation_config.json`'s `top_p` when the request
  omits it, else the sampler's own default), `top_k` ≥ 0, `min_p` in [0, 1],
  `model` (ignored, echoed back), `timings_per_token`, `return_progress`,
  `stream_options.include_usage`, `chat_template_kwargs` (merged into the
  render kwargs; `reasoning_effort` comes from here or the server default),
  `stop` (list of strings). Stop conditions are the union of
  `generation_config.json`'s `eos_token_id` (int or list) and the tokenizer's
  eos, passed as the Job's stop tokens.
- Streaming: `text/event-stream`, `data: <json>\n\n` events terminated by
  `data: [DONE]\n\n` — including after a mid-stream error (error chunk, then
  `[DONE]`). First chunk carries `delta: {"role":"assistant","content":""}`;
  chunks carry `id_slot`; the final chunk carries `finish_reason` (`"stop"`,
  or `"length"` when the engine's `eos_reason` is `"max_new_tokens"`), empty
  `delta`, `timings`, and (with `include_usage`) `usage`. **One chunk per
  generated token**: every engine result with text is emitted as exactly one
  delta — never two tokens merged, never one token split, never an empty-text
  delta.
- Stop strings hold back **whole tokens**: a token is released only once no
  stop string starting inside it can still complete. On a match everything
  before it is emitted (queued tokens whole, then the straddling token's
  prefix as one final chunk) and the rest is dropped.
- Thinking: token-level `<think>…</think>` split into `delta.reasoning_content`
  / `delta.content`. The splitter starts in think mode when the rendered
  prompt ends with `<think>` (this model's template opens the block itself).
  A token that is exactly a tag switches mode and emits nothing; a token
  straddling `</think>` emits one delta carrying both sides. An unterminated
  think block flushes as reasoning at finish. `--no-think-split` sends every
  token as `content`, tags included.
- `timings` object (on the final chunk always; on content chunks too when
  `timings_per_token`): `prompt_n` (prompt tokens **minus** cached),
  `prompt_ms`, `prompt_per_second` (over `prompt_n`), `predicted_n`,
  `predicted_ms`, `predicted_per_second`, `cache_n` (the engine's
  `cached_tokens`; omitted when the engine does not report it), plus
  `draft_n`/`draft_n_accepted` only when the engine reports draft counters.
  Final-chunk figures come from the engine's own eos result — never from wall
  clock. Per-token chunks carry provisional wall-clock values until then.
  `usage.prompt_tokens` stays the **total** (cached + new); the split travels
  in `usage.prompt_tokens_details.cached_tokens`.
- `prompt_progress` chunks (with `return_progress`): one at submission
  (`processed: 0`), one for every engine result carrying
  `curr_progress`/`max_progress` (`cache = total − max_progress`,
  `processed = cache + curr_progress`, `time_ms` since request start), and a
  final one when the first token arrives (`processed = total`).
- Malformed bodies get 400
  `{"error":{"code":400,"message":"...","type":"invalid_request_error"}}`;
  a `/completion`-style `prompt` field is rejected with a message naming
  `/completion` as unsupported.

## The `/props` engine block

`engine` states what is actually running, measured where a number can be
measured and **omitted** where it cannot (never a 0 or a guess; a measured 0
is printed as 0):

    "engine": {
      "name": "exllamav3",
      "version": "1.5.0",                    // exllamav3.version.__version__
      "args": ["-m", "/path", "-gs", "44,21"],  // sys.argv[1:], verbatim
      "model": {
        "format": "exl3", "arch": "...ForCausalGeneration",
        "quant": "EXL3 4.05 bpw · head 6.0",  // bits as written in the file
        "bytes": 188666605996, "files": 31,   // top-level non-hidden files
        "params": 313326811966,               // params, EXL3 quads included
        "n_layers": 81, "n_experts": 384, "n_experts_used": 8,
        "ctx_train": 131072,
        "active_bytes_per_token": 10979084996  // dense tensors + 1 embed row
      },                                      //   + top_k experts/layer
      "placement": {                          // runtime walk, once at load
        "devices": [
          {"device": "GPU0", "bytes": …, "classes": {"attention": …, …}},
          {"device": "CPU",  "bytes": …, "classes": {"experts": …}}
        ],
        "vram_kv_bytes": …
      }
    }

Tensor classes follow the recorder's classification: `embeddings`, `output`
(`lm_head`/`shared_head`), `experts` (routed experts **and** the router
`.mlp.gate.` and `.mlp.shared_experts.` — counted as experts, not ffn), `ffn`
(other `.mlp.`), `attention` (`.self_attn.`), `other`. Per-device `bytes`
equals the sum of its classes; classes and devices with nothing measured are
omitted; `placement` disappears entirely when nothing was measured. A CPU
device whose CPU-worker expert sizes are not uniform across a layer's experts
reports `{"device": "CPU"}` with the numbers omitted rather than guessed.
There are no per-device `active_bytes` or layer counts — those are
whole-model properties.

The disk-side computations (safetensors header walk, param count, active
bytes) live in `exl3_serve/engine_info.py` — pure Python, no exllamav3/torch
imports, unit-tested against a synthetic model directory with hand-computed
numbers. The runtime walk (module tree, storages, `stc.get_tensor_sizes`,
cache bytes) happens once at load in `engine_exl3.py`. `CUDA_DEVICE_ORDER=PCI_BUS_ID`
is set at package import, before torch loads, so `GPU<n>` matches
nvidia-smi's numbering.

## Limitations (second cut)

- No prefix cache is configured, so `cached_tokens` is 0 in practice — but a
  nonzero engine report flows through `prompt_n`/`cache_n`/`usage` unchanged,
  and a missing report is omitted rather than zeroed. A freed slot reports
  `n_past: 0`.
- When a `stop` string cancels a job and the engine produces no final result
  (exllamav3's cancel path), the final chunk falls back to the provisional
  wall-clock timings.
- `messages[*].content` must be a string or null — no multimodal parts.
- One engine thread: prompt encoding waits for in-flight decode steps.
- While the model is loading, `/props`, `/slots` and `/v1/chat/completions`
  return 503 with the loading error shape.
- Huge `max_tokens` is clamped to `--max-tokens-cap` (4096 by default) rather
  than rejected.
- Concurrency is for latency, not throughput, when the experts live in host
  RAM: two streams each read their own experts, so the bytes a step moves
  double with the tokens it produces. Measured on the workstation above, two
  streams decoded 11.1 tok/s each against 22.0 for one, while the second
  request's wait fell from 12.3 s to 4.6 s.

