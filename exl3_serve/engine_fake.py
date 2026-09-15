"""Deterministic fake engine: the whole HTTP layer is testable without a GPU.

It mirrors the measured exllamav3 1.5.0 ``Generator.iterate()`` result shape
(rig-log probe 2026-09-15): per job, two text-less prefill results first (the
second carrying ``curr_progress``/``max_progress``), then **one result per
token** with ``text`` and one-element ``token_ids``, the final one riding the
last token together with ``eos``, ``eos_reason`` and the timing figures.
Scripted tokens are word-sized (``<think>``/``</think>`` are single tokens),
so token-boundary behaviour is exercised the way the real tokenizer produces
it.

Behaviour contracts the test suite relies on:

- ``render_chat`` concatenates the messages chatml-ish and ends with
  ``<|assistant|><think>`` when constructed with ``think_prompt=True``
  (default) -- like the target model's template -- so the reply opens inside
  a think block. The default script accordingly starts with reasoning and
  contains ``</think>`` but never ``<think>``.
- ``encode`` maps each whitespace-separated word to a stable fake token id,
  so the engine can decode a prompt back to its words; ``submit`` derives
  the scripted reply from the prompt's user message unless ``script`` was
  pinned -- distinct prompts get distinct scripts (stream isolation tests).
- ``iterate`` advances every active job one step per call (prefill, prefill
  with progress, then one token per call), sleeping ``delay`` seconds once
  per call to emulate decode latency.
- ``cached_tokens`` (default 0) and ``draft_stats`` ride the final result;
  ``prefill_progress`` overrides the default (60% of the prompt, whole
  prompt) progress figures.
- ``cancel`` makes the next ``iterate`` produce a figures-only final result.
- ``fail_on_token`` injects an exception mid-generation, for the error paths.
- ``props["engine"]`` is a fixed engine block (name, version, args from the
  constructor or ``sys.argv[1:]``, a model block computed by engine_info
  when the model dir exists, no placement). The ``exllamav3_version`` prop
  stays None: no exllamav3 is installed here, so the Server header keeps
  saying ``unknown`` -- the block's version is a scripted fixture, not a
  measurement.
"""
from __future__ import annotations

import os
import re
import sys
import time
from typing import Any, Optional

from . import engine_info

# A built-in minimal template so `--fake` needs no files on disk.
MINIMAL_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{ '<|' + message['role'] + '|>\\n' + message['content'] }}\\n"
    "{%- endfor %}{{ '<|assistant|>\\n' }}"
)

_TAG_RE = re.compile(r"(<think>|</think>)")
_WORD_RE = re.compile(r"\s*\S+|\s+")


def tokenize(text: str) -> list[str]:
    """Word-sized tokens with the think tags as single tokens; leading
    spaces stay attached to their word (how the real detokenizer emits)."""
    out: list[str] = []
    for piece in _TAG_RE.split(text):
        if piece in ("<think>", "</think>"):
            out.append(piece)
        else:
            out.extend(_WORD_RE.findall(piece))
    return [t for t in out if t]


class FakeJob:
    __slots__ = ("tokens", "index", "cancelled", "final_sent", "phase",
                 "hit_cap", "prompt_len")

    def __init__(self, tokens: list[str], hit_cap: bool, prompt_len: int) -> None:
        self.tokens = tokens
        self.index = 0
        self.cancelled = False
        self.final_sent = False
        self.phase = 0  # 0/1: the two prefill steps, 2: decoding
        self.hit_cap = hit_cap
        self.prompt_len = prompt_len


class FakeEngine:
    def __init__(self, model_dir: str = "/tmp/fake-model", script: Optional[str] = None,
                 delay: float = 0.0, n_ctx: int = 32768,
                 draft_stats: Optional[tuple[int, int]] = None,
                 fail_on_token: Optional[int] = None,
                 cached_tokens: int = 0, think_prompt: bool = True,
                 prefill_progress: Optional[tuple[int, int]] = None,
                 engine_version: str = "1.5.0",
                 engine_args: Optional[list[str]] = None,
                 engine_draft: Optional[dict] = None) -> None:
        self.model_dir = model_dir
        self.script = script            # pinned script; None -> derive from the prompt
        self.delay = delay              # seconds slept once per iterate() call
        self.n_ctx = n_ctx
        self.think_prompt = think_prompt
        self.prefill_progress = prefill_progress
        self.fail_on_token = fail_on_token
        # Figures reported on the eos result (spec'd example values).
        self.prompt_tokens = 24
        self.cached_tokens = cached_tokens
        self.time_prefill = 0.5
        self.sec_per_token = 0.04
        self.draft_stats = draft_stats  # (accepted, rejected) or None
        # Observability for tests.
        self.jobs: list[FakeJob] = []
        self.records: list[dict] = []   # one dict per submit() call
        self.peak_active = 0
        self._vocab: dict[str, int] = {}
        self._rev: dict[int, str] = {}
        engine = engine_info.build_engine_block(
            "exllamav3", engine_version,
            engine_args if engine_args is not None else engine_info.engine_args(sys.argv[1:]),
            model_dir=model_dir)
        if engine_draft is not None:
            engine["draft"] = dict(engine_draft)
        self.props: dict = {
            "model_path": os.path.abspath(model_dir),
            "chat_template": MINIMAL_CHAT_TEMPLATE,
            "n_ctx": n_ctx,
            "exllamav3_version": None,
            "alias": None,
            "engine": engine,
        }

    # -- Engine protocol --------------------------------------------------

    def render_chat(self, messages: list[dict], **kwargs: Any) -> str:
        out = []
        for m in messages:
            out.append(f"<|{m.get('role', 'user')}|>\n{m.get('content') or ''}\n")
        # the target model's template ends '<|assistant|><think>'
        out.append("<|assistant|><think>" if self.think_prompt else "<|assistant|>\n")
        return "".join(out)

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            tid = self._vocab.get(word)
            if tid is None:
                tid = 1000 + len(self._vocab)
                self._vocab[word] = tid
                self._rev[tid] = word
            ids.append(tid)
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._rev.get(i, "?") for i in ids)

    def submit(self, ids: list[int], max_new_tokens: int,
               sampling: Optional[dict], stop: Optional[list[str]]) -> Any:
        self.records.append({
            "ids_len": len(ids),
            "max_new_tokens": max_new_tokens,
            "sampling": dict(sampling or {}),
            "stop": list(stop or []),
        })
        text = self.script if self.script is not None else self._script_text(ids)
        all_tokens = tokenize(text)
        hit_cap = max_new_tokens is not None and len(all_tokens) > max_new_tokens
        tokens = all_tokens[:max_new_tokens] if hit_cap else all_tokens
        job = FakeJob(tokens, hit_cap, len(ids))
        self.jobs.append(job)
        self.peak_active = max(self.peak_active, len(self.jobs))
        return job

    def cancel(self, job: Any) -> None:
        job.cancelled = True

    def iterate(self) -> list[dict]:
        if self.delay:
            time.sleep(self.delay)
        results = []
        for job in list(self.jobs):
            if (self.fail_on_token is not None and job.phase == 2
                    and job.index == self.fail_on_token):
                self.jobs = []  # engine "crashed": nothing stays runnable
                raise RuntimeError(f"fake engine injected failure at token {self.fail_on_token}")
            if job.cancelled:
                results.append(self._final(job, "cancelled"))
                continue
            if job.phase == 0:
                job.phase = 1
                results.append({"stage": "prefill", "serial": 0,
                                "eos": False, "job": job})
            elif job.phase == 1:
                job.phase = 2
                curr, mx = self._progress_for(job)
                results.append({"stage": "prefill", "serial": 0, "eos": False,
                                "job": job, "curr_progress": curr,
                                "max_progress": mx})
            elif job.index < len(job.tokens):
                token = job.tokens[job.index]
                job.index += 1
                res = {"stage": "streaming", "serial": job.index, "eos": False,
                       "job": job, "text": token, "token_ids": [5000 + job.index]}
                if job.index == len(job.tokens):
                    # last token rides on the final result, like exllamav3
                    res.update(self._figures(job.index,
                                             "max_new_tokens" if job.hit_cap else "eos"))
                    res["eos"] = True
                    res["full_completion"] = "".join(job.tokens)
                    job.final_sent = True
                results.append(res)
            elif not job.final_sent:  # max_new_tokens == 0
                results.append(self._final(job, "eos"))
        self.jobs = [j for j in self.jobs if not j.final_sent]
        return results

    def num_remaining_jobs(self) -> int:
        return len(self.jobs)

    # -- internals ----------------------------------------------------------

    def _script_text(self, ids: list[int]) -> str:
        words = self.decode(ids)
        m = re.search(r"<\|user\|>\s*(.*?)\s*<\|assistant\|>", words)
        topic = (m.group(1) if m else "this").strip() or "this"
        if self.think_prompt:
            # the prompt already opened the think block
            return (f"The fake model thinks about {topic}.</think>"
                    f"The fake model answers {topic}.")
        return (f"<think>The fake model thinks about {topic}.</think>"
                f"The fake model answers {topic}.")

    def _progress_for(self, job: FakeJob) -> tuple[int, int]:
        if self.prefill_progress is not None:
            return self.prefill_progress
        mx = max(job.prompt_len, 1)
        return mx - max(1, mx // 5), mx

    def _figures(self, emitted: int, eos_reason: Optional[str] = None) -> dict:
        fig = {
            "new_tokens": emitted,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cached_pages": 0,
            "time_enqueued": 0.0005,
            "time_prefill": self.time_prefill,
            "time_generate": emitted * self.sec_per_token,
        }
        if eos_reason is not None:
            fig["eos_reason"] = eos_reason
        if self.draft_stats is not None:
            fig["accepted_draft_tokens"], fig["rejected_draft_tokens"] = self.draft_stats
        return fig

    def _final(self, job: FakeJob, eos_reason: str) -> dict:
        job.final_sent = True
        res = {"stage": "streaming", "serial": job.index, "text": "",
               "token_ids": [], "eos": True, "job": job,
               "full_completion": "".join(job.tokens[:job.index])}
        res.update(self._figures(job.index, eos_reason))
        return res
