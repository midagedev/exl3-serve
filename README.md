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
