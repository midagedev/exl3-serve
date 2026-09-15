"""The engine seam: everything the HTTP layer needs from a model backend.

The real backend (engine_exl3) needs a GPU with exllamav3 installed; the fake
(engine_fake) is deterministic and runs anywhere, which is what the test suite
drives end to end. Both satisfy this Protocol, so the HTTP layer never knows
which one it is serving.

Result dicts mirror the exllamav3 `Generator.iterate()` job results documented
in reference/exl3-bench3.py: `text` and `eos` on every step, the timing figures
(`new_tokens`, `prompt_tokens`, `time_prefill`, `time_generate`, and the draft
counters when a draft model is active) on the final result, plus `job` so a
batched dispatch loop can route each result to its request.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, TypedDict, runtime_checkable


class EngineJobResult(TypedDict, total=False):
    text: str
    eos: bool
    new_tokens: int
    prompt_tokens: int
    time_prefill: float
    time_generate: float
    accepted_draft_tokens: Optional[int]
    rejected_draft_tokens: Optional[int]
    job: Any


@runtime_checkable
class Engine(Protocol):
    def render_chat(self, messages: list[dict], **kwargs) -> str: ...
    def encode(self, text: str) -> list[int]: ...
    def submit(self, ids: list[int], max_new_tokens: int,
               temperature: Optional[float], stop: Optional[list[str]]) -> Any: ...
    def cancel(self, job: Any) -> None: ...
    def iterate(self) -> list[EngineJobResult]: ...
    def num_remaining_jobs(self) -> int: ...

    props: dict  # model_path, chat_template, n_ctx, exllamav3_version, alias
