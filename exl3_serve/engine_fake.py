"""Deterministic fake engine: the whole HTTP layer is testable without a GPU.

Behaviour contracts the test suite relies on:

- ``render_chat`` concatenates the messages in a chatml-ish shape (and always
  ends with the assistant generation prompt).
- ``encode`` maps each whitespace-separated word to a stable fake token id and
  remembers the mapping, so the engine can decode a prompt back to its words.
- ``submit`` derives the scripted reply from the prompt's user message unless
  ``script`` was pinned at construction -- distinct prompts therefore get
  distinct scripts, which is what the stream-isolation tests rely on.
- ``iterate`` emits **one character per call per active job** (so ``<think>``
  tags necessarily arrive split across token boundaries, like real
  detokenized deltas), sleeping ``delay`` seconds once per call to emulate
  decode latency, and the last token rides on the final result together with
  the fixed timing figures, mirroring the exllamav3 Job result shape.
- ``cancel`` makes the next ``iterate`` produce a final result with figures
  for the tokens emitted so far (what the real engine does on its good days).
- ``fail_on_token`` injects an exception mid-generation, for the error paths.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

# A built-in minimal template so `--fake` needs no files on disk.
MINIMAL_CHAT_TEMPLATE = (
    "{%- for message in messages %}"
    "{{ '<|' + message['role'] + '|>\\n' + message['content'] }}\\n"
    "{%- endfor %}{{ '<|assistant|>\\n' }}"
)


class FakeJob:
    __slots__ = ("tokens", "index", "cancelled", "final_sent")

    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.index = 0
        self.cancelled = False
        self.final_sent = False


class FakeEngine:
    def __init__(self, model_dir: str = "/tmp/fake-model", script: Optional[str] = None,
                 delay: float = 0.0, n_ctx: int = 32768,
                 draft_stats: Optional[tuple[int, int]] = None,
                 fail_on_token: Optional[int] = None) -> None:
        self.model_dir = model_dir
        self.script = script            # pinned script; None -> derive from the prompt
        self.delay = delay              # seconds slept once per iterate() call
        self.n_ctx = n_ctx
        # Figures reported on the eos result (spec'd example values).
        self.prompt_tokens = 24
        self.time_prefill = 0.5
        self.sec_per_token = 0.04
        self.draft_stats = draft_stats  # (accepted, rejected) or None
        self.fail_on_token = fail_on_token
        # Observability for tests.
        self.jobs: list[FakeJob] = []
        self.records: list[dict] = []   # one dict per submit() call
        self.peak_active = 0
        self._vocab: dict[str, int] = {}
        self._rev: dict[int, str] = {}
        self.props: dict = {
            "model_path": os.path.abspath(model_dir),
            "chat_template": MINIMAL_CHAT_TEMPLATE,
            "n_ctx": n_ctx,
            "exllamav3_version": None,
            "alias": None,
        }

    # -- Engine protocol --------------------------------------------------

    def render_chat(self, messages: list[dict], **kwargs: Any) -> str:
        out = []
        for m in messages:
            out.append(f"<|{m.get('role', 'user')}|>\n{m.get('content') or ''}\n")
        out.append("<|assistant|>\n")
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
               temperature: Optional[float], stop: Optional[list[str]]) -> Any:
        self.records.append({
            "ids_len": len(ids),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "stop": list(stop or []),
        })
        job = FakeJob(self._script_for(ids, max_new_tokens))
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
            if self.fail_on_token is not None and job.index == self.fail_on_token:
                self.jobs = []  # engine "crashed": nothing stays runnable
                raise RuntimeError(f"fake engine injected failure at token {self.fail_on_token}")
            if job.cancelled:
                results.append(self._final(job))
                continue
            if job.index < len(job.tokens):
                token = job.tokens[job.index]
                job.index += 1
                res = {"text": token, "eos": False, "job": job}
                if job.index == len(job.tokens):
                    # last token rides on the final result, like exllamav3
                    res.update(self._figures(job.index))
                    res["eos"] = True
                    job.final_sent = True
                results.append(res)
            elif not job.final_sent:  # max_new_tokens == 0
                results.append(self._final(job))
        self.jobs = [j for j in self.jobs if not j.final_sent]
        return results

    def num_remaining_jobs(self) -> int:
        return len(self.jobs)

    # -- internals ----------------------------------------------------------

    def _script_for(self, ids: list[int], max_new_tokens: Optional[int]) -> list[str]:
        if self.script is not None:
            text = self.script
        else:
            words = self.decode(ids)
            m = re.search(r"<\|user\|>\s*(.*?)\s*<\|assistant\|>", words)
            topic = (m.group(1) if m else "this").strip() or "this"
            text = (f"<think>The fake model thinks about {topic}.</think>"
                    f"The fake model answers {topic}.")
        tokens = list(text)
        if max_new_tokens is not None:
            tokens = tokens[:max_new_tokens]
        return tokens

    def _figures(self, emitted: int) -> dict:
        fig = {
            "new_tokens": emitted,
            "prompt_tokens": self.prompt_tokens,
            "time_prefill": self.time_prefill,
            "time_generate": emitted * self.sec_per_token,
        }
        if self.draft_stats is not None:
            fig["accepted_draft_tokens"], fig["rejected_draft_tokens"] = self.draft_stats
        return fig

    def _final(self, job: FakeJob) -> dict:
        job.final_sent = True
        res = {"text": "", "eos": True, "job": job}
        res.update(self._figures(job.index))
        return res
