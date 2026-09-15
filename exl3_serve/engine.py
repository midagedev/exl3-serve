"""The engine seam: everything the HTTP layer needs from a model backend.

The real backend (engine_exl3) needs a GPU with exllamav3 installed; the fake
(engine_fake) is deterministic and runs anywhere, which is what the test suite
drives end to end. Both satisfy this Protocol, so the HTTP layer never knows
which one it is serving.

Result dicts mirror the measured exllamav3 1.5.0 ``Generator.iterate()``
job results (rig-log probe 2026-09-15): one result per generated token.
Every result carries ``job``, ``stage``, ``serial``, ``eos``. The two prefill
results carry no text; the second carries ``curr_progress``/``max_progress``.
Token results carry ``text`` (complete characters) and a one-element
``token_ids``. The final result still carries that step's text and adds
``eos_reason`` plus the figures: ``new_tokens``, ``prompt_tokens``,
``cached_tokens``, ``cached_pages``, ``time_enqueued``, ``time_prefill``,
``time_generate``, the accepted/rejected draft counters when a draft model
is active, and ``full_completion``.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, TypedDict, runtime_checkable


class EngineJobResult(TypedDict, total=False):
    text: str
    token_ids: list
    eos: bool
    eos_reason: str
    stage: str
    serial: int
    curr_progress: int
    max_progress: int
    new_tokens: int
    prompt_tokens: int
    cached_tokens: int
    cached_pages: int
    time_enqueued: float
    time_prefill: float
    time_generate: float
    accepted_draft_tokens: Optional[int]
    rejected_draft_tokens: Optional[int]
    full_completion: str
    job: Any


@runtime_checkable
class Engine(Protocol):
    def render_chat(self, messages: list[dict], **kwargs) -> str: ...
    def encode(self, text: str) -> list[int]: ...
    def submit(self, ids: list[int], max_new_tokens: int,
               sampling: Optional[dict], stop: Optional[list[str]]) -> Any: ...
    def cancel(self, job: Any) -> None: ...
    def iterate(self) -> list[EngineJobResult]: ...
    def num_remaining_jobs(self) -> int: ...

    # model_path, chat_template, n_ctx, exllamav3_version, alias, and engine
    # (the /props engine block) when the engine can state one
    props: dict
