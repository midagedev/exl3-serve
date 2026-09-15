"""Real engine: wraps exllamav3 exactly as reference/exl3-bench3.py drives it.

exllamav3 is imported lazily (inside functions), so this module -- and the
whole package -- imports on machines without it; that is how the test suite
and `--fake` smoke runs work. Nothing here is exercised by the unit tests:
it is checked by `python -m py_compile` here and validated by the lead on the
GPU box.

Two API assumptions, both taken from the reference bench (exllamav3 1.5.0):

- iterate() result dicts can be attributed to their job. We read ``res["job"]``
  when present; otherwise we fall back to "only one job in flight" and refuse
  to batch rather than misroute tokens to the wrong stream.
- the final result of a job (eos, cap or cancel) carries the timing figures
  the bench prints: ``new_tokens``, ``prompt_tokens``, ``time_prefill``,
  ``time_generate``, and the accepted/rejected draft counters. When a job is
  cancelled mid-flight no final result may arrive -- the HTTP layer then falls
  back to provisional timings and says so (README, Limitations).
"""
from __future__ import annotations

import os
from typing import Any, Optional

import jinja2


def _default_sampler_available() -> bool:
    try:
        from exllamav3 import DefaultSampler  # noqa: F401
        return True
    except Exception:
        return False


class Exl3Engine:
    def __init__(self, model_dir: str, model: Any, cache: Any, tokenizer: Any,
                 gen: Any, template: Any, chat_template_raw: str,
                 version: Optional[str], n_ctx: int,
                 default_sampler_available: bool) -> None:
        self.model_dir = model_dir
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.gen = gen
        self.template = template
        self.default_sampler_available = default_sampler_available
        self._inflight: set = set()
        self.props: dict = {
            "model_path": os.path.abspath(model_dir),
            "chat_template": chat_template_raw,
            "n_ctx": n_ctx,
            "exllamav3_version": version,
            "alias": None,
        }

    @classmethod
    def load(cls, args: Any) -> "Exl3Engine":
        # Lazy on purpose: importing exllamav3 must stay optional.
        from exllamav3 import Generator, model_init
        from importlib.metadata import version as _pkg_version

        r = model_init.init(args, progress=False, quiet=True)
        model, config, cache, tokenizer = r[:4]
        draft_model = draft_config = draft_cache = None
        if len(r) > 4:
            draft_model, draft_config, draft_cache = (list(r[4:7]) + [None, None, None])[:3]
        gen = Generator(
            model=model, cache=cache, tokenizer=tokenizer,
            draft_model=draft_model, draft_cache=draft_cache,
            max_batch_size=getattr(args, "parallel", 2) or 2,
            num_draft_tokens=getattr(args, "draft_n", None),
            dynamic_draft_tokens=bool(getattr(args, "dyn_draft", False)),
        )
        tpl_path = os.path.join(args.model_dir, "chat_template.jinja")
        if not os.path.isfile(tpl_path):
            raise FileNotFoundError(f"chat template not found: {tpl_path}")
        chat_template_raw = open(tpl_path, encoding="utf-8").read()
        template = jinja2.Environment(
            extensions=["jinja2.ext.loopcontrols"]).from_string(chat_template_raw)
        try:
            version = _pkg_version("exllamav3")
        except Exception:
            version = None
        n_ctx = cls._cache_tokens(cache, config, args)
        default_sampler = _default_sampler_available()
        if not default_sampler:
            print("exl3-serve: exllamav3.DefaultSampler not available; "
                  "non-zero temperatures fall back to greedy", flush=True)
        return cls(args.model_dir, model, cache, tokenizer, gen, template,
                   chat_template_raw, version, n_ctx, default_sampler)

    @staticmethod
    def _cache_tokens(cache: Any, config: Any, args: Any) -> int:
        # Best effort across exllamav3 versions; the -cs flag default is the
        # final fallback (model_init.add_args(..., default_cache_size=32768)).
        for obj, attrs in ((cache, ("max_num_tokens", "num_tokens", "max_seq_len")),
                           (config, ("max_seq_len",))):
            for attr in attrs:
                v = getattr(obj, attr, None)
                if isinstance(v, int) and v > 0:
                    return v
        v = getattr(args, "cache_size", None)
        return v if isinstance(v, int) and v > 0 else 32768

    # -- Engine protocol ----------------------------------------------------

    def render_chat(self, messages: list[dict], **kwargs: Any) -> str:
        kw = {"tools": None}
        kw.update(kwargs)
        return self.template.render(messages=messages, **kw)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, encode_special_tokens=True)

    def submit(self, ids: list[int], max_new_tokens: int,
               temperature: Optional[float], stop: Optional[list[str]]) -> Any:
        from exllamav3 import GreedySampler, Job
        sampler = GreedySampler() if not temperature else self._sampler(temperature)
        job = Job(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            sampler=sampler,
            stop_conditions=[self.tokenizer.eos_token_id],
        )
        self._inflight.add(job)
        self.gen.enqueue(job)
        return job

    def _sampler(self, temperature: float) -> Any:
        # exllamav3.DefaultSampler(temperature=...) when this install has it,
        # else greedy (the fallback is reported at load time).
        from exllamav3 import GreedySampler
        if self.default_sampler_available:
            from exllamav3 import DefaultSampler
            try:
                return DefaultSampler(temperature=temperature)
            except TypeError:
                pass
        return GreedySampler()

    def cancel(self, job: Any) -> None:
        self._inflight.discard(job)
        try:
            self.gen.cancel(job)
        except Exception as exc:
            print(f"exl3-serve: gen.cancel failed: {exc}", flush=True)

    def iterate(self) -> list[dict]:
        if not self.gen.num_remaining_jobs():
            return []
        out = []
        for res in (self.gen.iterate() or []):
            r = dict(res)
            job = r.get("job")
            if job is None:
                if len(self._inflight) == 1:
                    r["job"] = job = next(iter(self._inflight))
                else:
                    raise RuntimeError(
                        "exllamav3 iterate() results carry no job identity and "
                        f"{len(self._inflight)} jobs are in flight; cannot dispatch batch")
            if r.get("eos"):
                self._inflight.discard(job)
            out.append(r)
        return out

    def num_remaining_jobs(self) -> int:
        return self.gen.num_remaining_jobs()
