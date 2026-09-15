# The surface, in detail

exl3-serve presents the part of `llama-server`'s HTTP surface a recorder or a
chat client needs, and nothing else. This file is the exact contract; the
README is the short version.

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
