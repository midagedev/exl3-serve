# exl3-serve

**A llama-server-compatible HTTP front for [ExLlamaV3](https://github.com/turboderp-org/exllamav3).**

[![check](https://github.com/midagedev/exl3-serve/actions/workflows/ci.yml/badge.svg)](https://github.com/midagedev/exl3-serve/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ExLlamaV3 ships no server, so nothing written against `llama-server` can drive
an EXL3 model — including the recorders people use to publish numbers. This
serves one EXL3 model on that surface: `/props`, `/health`, `/slots` and
`/v1/chat/completions` (SSE) with llama-server's `timings` object filled from
exllamav3's own job results.

Which means a run through it records like any other:

```text
┌──────────────────────────────────────────────────────────────────────┐
│ toktape v0.2.2               20260915-220345-glm-5-3-flash-exl3-4-05 │
├──────────────────────────────────────────────────────────────────────┤
│ MODEL    GLM-5.3-Flash-exl3-4.05 · EXL3 4.05 bpw · head 6.0          │
│          153.8 GiB                                                   │
│ ENGINE   exllamav3 1.5.0 · linux 6.8.0-139-generic · workstation     │
│ RIG      RTX A6000 48G · RTX 3090 24G                                │
│          AMD Ryzen Threadripper PRO 5975WX 32-Cores                  │
│          252 GB DDR4-3600                                            │
├──────────────────────────────────────────────────────────────────────┤
│ Decode        23.5 tok/s aggregate · 11.5 tok/s each                 │
│ Prefill       46.4 tok/s aggregate · ? tok/s each                    │
│               274 prompt tokens · TTFT p50 11652 ms                  │
│ Context       8192 (274 in / 214 out)                                │
│ Prefix cache  0% hit (0/274) · warm                                  │
│ Sampling      greedy (temp 0) · chat                                 │
│ Streams       2 × 11.5 tok/s = 23.5 tok/s aggregate                  │
│               TTFT p50 11652 ms p95 11809 ms · slots busy max 2      │
├──────────────────────────────────────────────────────────────────────┤
│ MEMORY   GPU0 [█████████░] 44.0/48.0 GiB                             │
│          GPU1 [████████░░] 19.8/24.0 GiB                             │
│          weights 57.9 | kv 0.1 | compute ? GiB                       │
│          Host placed 92.5 GiB                                        │
│          Host RSS 99.2 GiB (file 1.2 / anon 97.6)                    │
│          Page faults 0.0 maj/token (0 during decode)                 │
├──────────────────────────────────────────────────────────────────────┤
│ HOST     GPU0 68°C 122 W · GPU1 50°C 145 W · throttled: no           │
│          contended: no                                               │
│          conditions changed: k10temp Tctl 53 → 67 °C                 │
├──────────────────────────────────────────────────────────────────────┤
│ FLAGS    -gs 44,21 -mcs 185 -mct 32 -cs 8192 -mtp --parallel 2       │
├──────────────────────────────────────────────────────────────────────┤
│ ! 5 caveats — the client measured 11.7 tok/s where the server        │
│   reported 11.5 tok/s, over the 2% tolerance · recorded · recorded · │
│   conditions_changed · run_cut_by_clock                              │
├──────────────────────────────────────────────────────────────────────┤
│                toktape · github.com/midagedev/toktape                │
└──────────────────────────────────────────────────────────────────────┘
```

That card is [toktape](https://github.com/midagedev/toktape) attached to
exl3-serve over `/props`, on a two-card workstation serving GLM-5.3-Flash at
4.05 bpw with 185 of every layer's 288 experts in host RAM. Nothing about it
is exl3-specific except what the engine reported.

## Run it

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/exl3-serve -m /path/to/exl3-model --port 8080 --parallel 2
```

Every exllamav3 `model_init` flag is accepted verbatim (`-gs`, `-mcs`, `-mcl`,
`-mct`, `-cs`, `-mtp`, …). Without a GPU, `--fake -m /tmp/fake-model` serves a
deterministic fake engine — the same surface, which is what the test suite
drives.

| flag | |
|---|---|
| `--parallel N` | concurrent streams, default 2. Requests past N wait for a slot, never rejected |
| `--alias NAME` | model name reported to clients (default: the model directory's name) |
| `--reasoning-effort` | server default for the chat-template kwarg; requests override it |
| `--draft-n N` | MTP draft tokens per step, default 1 when `-mtp` is given. `--dyn-draft` for dynamic |
| `--no-think-split` | send everything as `content`, no `reasoning_content` |
| `--max-tokens-cap` | clamp for request `max_tokens`/`n_predict`, default 4096 |

`--parallel` is the one flag with a trap in it, and it is exllamav3's: caches
are sized from `-ambs` (default 1) and the Generator clamps its batch to the
cache's slot count, so asking for two slots quietly gets you one that
serializes. `load` raises `-ambs` to match and then asserts the served batch,
failing the load with both numbers rather than serving N slots that decode one
at a time. Two real slots also cost VRAM: on a tight split the load can fail
where `--parallel 1` fits, and a smaller `-cs` is the lever.

## What it serves

- **`/props`** — `model_path`, `chat_template`, `total_slots`, `n_ctx`, and an
  `engine` object: what is running, the model's shape, and where every tensor
  actually sits, measured at load. No `build_info`, on purpose.
- **`/health`**, **`/slots`**, **`/apply-template`** — loading state, per-slot
  `n_past`, and the rendered prompt.
- **`/v1/chat/completions`** — streamed or not, one SSE chunk per generated
  token, `timings` from the engine's own figures rather than a wall clock,
  `prompt_progress` during prefill, `<think>` split into `reasoning_content`,
  stop strings that hold back whole tokens.

The exact contract — every honoured field, the `engine` block's schema, the
tensor classes, the streaming and `timings` semantics — is
[`docs/surface.md`](docs/surface.md).

## Measured

One workstation (RTX A6000 48 GB + RTX 3090 24 GB, Threadripper PRO 5975WX,
252 GB DDR4-3600), GLM-5.3-Flash EXL3 4.05 bpw, 154 GiB, `-gs 44,21 -mcs 185
-mct 32 -mtp`, recorded by toktape v0.2.2:

| streams | prompt | per stream | aggregate | draft accepted |
|---:|---:|---:|---:|---:|
| 1 | 32 | 21.96 tok/s | 22.1 tok/s | 96 % |
| 2 | 274 each | 11.5 tok/s | 23.5 tok/s | 92 % |

Two streams deliver about 5 % more in total, not double: decode is the read of
98.1 GB of host-RAM experts, and two tokens routed to different experts double
the bytes along with the tokens. Concurrency here buys both clients being
answered at once — the second one's wait for its first token fell from 12.3 s
to 4.6 s — rather than throughput. The full write-up, with the tapes, is in
[rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-15-exl3-serve-and-tabbyapi-timings.md).

## Limitations

- No prefix cache is configured, so `cached_tokens` is 0 in practice; a nonzero
  engine report flows through untouched, and a missing one is omitted, never
  zeroed.
- One engine thread: prompt encoding waits behind in-flight decode steps.
- `messages[*].content` must be a string or null — no multimodal parts.
- A cancelled job — a `stop` string, or a client that goes away — never gets
  the engine's own eos figures, so that chunk's `timings` are this server's
  wall clock: prefill is measured from the moment the job reached the engine
  to the first token (not from when the request arrived, so a recorder
  subtracting it from TTFT does not count the queue twice), decode from the
  first token to the last. A finished job always carries the engine's own
  figures instead.
- While the model loads, everything answers 503 with the reason in it.

## Why Python

exllamav3 is a Python library with CUDA extensions, no C API and no process
protocol, so the model has to live in a Python process. This is that process
plus about two hundred lines of HTTP; anything larger belongs outside it.

MIT.
