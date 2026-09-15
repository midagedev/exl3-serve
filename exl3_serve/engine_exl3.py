"""Real engine: wraps exllamav3 exactly as reference/exl3-bench3.py drives it.

exllamav3 is imported lazily (inside functions), so this module -- and the
whole package -- imports on machines without it; that is how the test suite
and `--fake` smoke runs work. The runtime walk and the sampler wiring are
checked by `python -m py_compile` here and validated by the lead on the GPU
box; the pure helpers they use (engine_info, stop_token_ids) are unit-tested
on any machine.

API facts taken from the 2026-09-15 probe of exllamav3 1.5.0 (rig-log
tools/exl3/exl3-probe.py):

- ``Generator.iterate()`` yields one result per generated token, even with
  the MTP draft on: two text-less prefill results first (the second carrying
  ``curr_progress``/``max_progress``), then one ``text``/``token_ids`` result
  per token, the last one carrying ``eos`` plus the timing figures.
- ``Generator.__init__`` takes ``num_draft_tokens``/``dynamic_draft_tokens``;
  ``Generator.cancel(job)`` exists.
- the client-facing version is ``exllamav3.version.__version__`` ("1.5.0");
  the wheel's importlib.metadata string carries a local-version suffix
  ("1.5.0+cu128.torch2.10.0") and must not be used.
- ``GreedySampler()`` and ``ComboSampler(temperature=..., top_p=...,
  top_k=..., min_p=...)``; ``DefaultSampler`` takes no arguments.
- iterating the model yields its top modules, each with a ``modules`` list
  and ``get_tensors()``; ``config.stc.get_tensor_sizes`` gives per-expert
  byte sizes; ``cache.layers`` holds the KV tensors.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

import jinja2

from . import engine_info


def stop_token_ids(gen_cfg: Optional[dict],
                   tokenizer_eos: Optional[int]) -> set[int]:
    """Stop conditions for a Job: the union of generation_config.json's
    ``eos_token_id`` (int or list) and the tokenizer's eos id. The target
    model ends turns with any of three ids while the tokenizer knows one."""
    ids: set[int] = set()
    if isinstance(gen_cfg, dict):
        v = gen_cfg.get("eos_token_id")
        if isinstance(v, bool):
            v = None
        if isinstance(v, int):
            ids.add(v)
        elif isinstance(v, (list, tuple)):
            ids.update(x for x in v if isinstance(x, int) and not isinstance(x, bool))
    if tokenizer_eos is not None:
        ids.add(int(tokenizer_eos))
    return ids


def _tensors_of(module: Any) -> list:
    try:
        ts = module.get_tensors()
    except Exception:
        return []
    if isinstance(ts, dict):
        return list(ts.values())
    return list(ts or [])


def _walk_roots(root: Any):
    """Every module of a model, exactly once: the top modules come from
    iterating the model, the rest from each module's ``modules`` list."""
    seen: set[int] = set()
    stack = list(root)
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        yield m
        stack.extend(getattr(m, "modules", []) or [])


def _tensor_key(t: Any) -> tuple:
    """Identity of a tensor's bytes: start address, byte length, device. Two
    views of the same bytes count once; two tensors sharing one storage do not
    collapse into each other."""
    return (t.data_ptr(), t.numel() * t.element_size(), str(t.device))


def _measure_weights(model: Any, draft_model: Any) -> dict[str, dict[str, int]]:
    """Per-device, per-class byte counts of the resident weights (main model
    and MTP model): each tensor's own bytes (numel x element size), deduped by
    (data_ptr, nbytes, device).

    Not the storage size: the loader packs weights into shared 128 MiB chunks,
    so ``untyped_storage().nbytes()`` gave a whole chunk to whichever tensor the
    walk met first -- an 8 KB norm weight was credited 0.125 GiB and the class
    split followed walk order (2026-09-15 class probe). Chunk slack is not a
    weight and is not counted."""
    import torch

    by: dict[str, dict[str, int]] = {}
    for root in (model, draft_model):
        if root is None:
            continue
        seen: set[tuple] = set()
        for m in _walk_roots(root):
            key = getattr(m, "key", "") or ""
            for t in _tensors_of(m):
                if not isinstance(t, torch.Tensor):
                    continue
                tk = _tensor_key(t)
                if tk in seen:
                    continue
                seen.add(tk)
                d = by.setdefault(str(t.device), {})
                # A sparse MoE module's own tensors (key "...layers.N.mlp") are expert
                # data -- the fused/per-expert buffers it keeps -- not dense ffn: the
                # 2026-09-15 probe found ~10 GB under that key on the two cards
                # against 1.2 GB of dense ffn + shared + router on disk.
                cls = ("experts" if hasattr(m, "num_experts")
                       else engine_info.classify_tensor(key))
                d[cls] = d.get(cls, 0) + tk[1]
    return by


def _cpu_expert_bytes(model: Any, config: Any) -> Optional[int]:
    """Bytes of the experts living in the CPU worker process: for every MoE
    module, (num_experts - len(ups)) x per-expert bytes. None when any
    layer's experts differ in byte size -- which experts sit on the CPU
    moves at run time, so only a uniform size makes the number exact."""
    total = 0
    for m in _walk_roots(model):
        if not (hasattr(m, "num_experts") and hasattr(m, "ups")):
            continue
        n_gpu = len(m.ups)
        if m.num_experts <= n_gpu:
            continue
        try:
            sizes = {sum(config.stc.get_tensor_sizes(f"{m.key}.experts.{e}"))
                     for e in range(m.num_experts)}
        except Exception:
            return None
        if len(sizes) != 1:
            return None
        total += (m.num_experts - n_gpu) * sizes.pop()
    return total


def _cache_bytes(caches: tuple) -> Optional[int]:
    """Bytes of the cache and draft-cache tensors on cuda:* devices, each
    tensor's own bytes, deduped like the weights."""
    import torch

    total = 0
    for cache in caches:
        layers = getattr(cache, "layers", None)
        if layers is None:
            continue
        items = layers.values() if isinstance(layers, dict) else layers
        seen: set[tuple] = set()
        for layer in items:
            for t in _tensors_of(layer):
                if not isinstance(t, torch.Tensor):
                    continue
                tk = _tensor_key(t)
                if tk in seen:
                    continue
                seen.add(tk)
                if str(t.device).startswith("cuda"):
                    total += tk[1]
    return total


class Exl3Engine:
    def __init__(self, model_dir: str, model: Any, cache: Any, tokenizer: Any,
                 gen: Any, template: Any, chat_template_raw: str,
                 version: Optional[str], n_ctx: int,
                 gen_cfg: Optional[dict], stop_ids: set[int],
                 engine_block: dict) -> None:
        self.model_dir = model_dir
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.gen = gen
        self.template = template
        self.gen_cfg = gen_cfg if isinstance(gen_cfg, dict) else {}
        self.stop_ids = stop_ids
        self._inflight: set = set()
        self.props: dict = {
            "model_path": os.path.abspath(model_dir),
            "chat_template": chat_template_raw,
            "n_ctx": n_ctx,
            "exllamav3_version": version,
            "alias": None,
            "engine": engine_block,
        }

    @classmethod
    def load(cls, args: Any) -> "Exl3Engine":
        # Lazy on purpose: importing exllamav3 must stay optional.
        from exllamav3 import Generator, model_init
        try:
            from exllamav3.version import __version__ as version
        except Exception:
            version = None

        r = model_init.init(args, progress=False, quiet=True)
        model, config, cache, tokenizer = r[:4]
        draft_model = draft_cache = None
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
        n_ctx = cls._cache_tokens(cache, config, args)

        try:
            with open(os.path.join(args.model_dir, "generation_config.json"),
                      encoding="utf-8") as f:
                gen_cfg = json.load(f)
        except (OSError, ValueError):
            gen_cfg = None
        stop_ids = stop_token_ids(gen_cfg, tokenizer.eos_token_id)

        engine_block = engine_info.build_engine_block(
            "exllamav3", version, engine_info.engine_args(sys.argv[1:]),
            model_dir=args.model_dir)
        if draft_model is not None:
            draft = engine_info.draft_block(
                bool(getattr(args, "mtp", False)), getattr(args, "draft_model_dir", None),
                getattr(args, "draft_n", None))  # what the Generator above was given
            if draft is not None:
                engine_block["draft"] = draft
        placement = cls._measure_placement(model, draft_model, config,
                                           cache, draft_cache)
        if placement is not None:
            engine_block["placement"] = placement

        return cls(args.model_dir, model, cache, tokenizer, gen, template,
                   chat_template_raw, version, n_ctx, gen_cfg, stop_ids,
                   engine_block)

    @staticmethod
    def _measure_placement(model: Any, draft_model: Any, config: Any,
                           cache: Any, draft_cache: Any) -> Optional[dict]:
        by = _measure_weights(model, draft_model)
        cpu_exact = True
        cpu_experts = _cpu_expert_bytes(model, config)
        if cpu_experts is None:
            cpu_exact = False
        elif cpu_experts:
            d = by.setdefault("cpu", {})
            d["experts"] = d.get("experts", 0) + cpu_experts
        vram_kv = _cache_bytes((cache, draft_cache))
        return engine_info.finalize_placement(
            by, vram_kv_bytes=vram_kv, cpu_numbers_exact=cpu_exact)

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
               sampling: Optional[dict], stop: Optional[list[str]]) -> Any:
        from exllamav3 import GreedySampler, Job
        s = sampling or {}
        temperature = s.get("temperature")
        if not temperature:
            sampler = GreedySampler()
        else:
            from exllamav3 import ComboSampler
            kw: dict[str, Any] = {"temperature": temperature}
            top_p = s.get("top_p")
            if top_p is None:
                top_p = self.gen_cfg.get("top_p")
            if isinstance(top_p, (int, float)) and not isinstance(top_p, bool):
                kw["top_p"] = top_p
            for k in ("top_k", "min_p"):
                v = s.get(k)
                if v is not None:
                    kw[k] = v
            sampler = ComboSampler(**kw)
        stops = sorted(self.stop_ids)
        if not stops and self.tokenizer.eos_token_id is not None:
            stops = [self.tokenizer.eos_token_id]
        job = Job(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            sampler=sampler,
            stop_conditions=stops or None,
        )
        self._inflight.add(job)
        self.gen.enqueue(job)
        return job

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
