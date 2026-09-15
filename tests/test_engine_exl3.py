"""Real-engine contract tests that need no exllamav3: the load-time batch
assertion (defect 1) and the encode/submit token plumbing (defect 2).

exllamav3 is not installed here (nor on CI), so each test installs stubs in
``sys.modules`` that mirror the measured exllamav3 1.5.0 behaviour verbatim:

- ``model_init.init`` builds every cache with
  ``max_batch_size = args.autosplit_max_batch_size`` -- the ``-ambs`` flag
  whose argparse default is 1 (exllamav3/model_init.py:75, read at :292).
- ``cache.num_slots`` is that ``max_batch_size`` (exllamav3/cache/cache.py:161).
- ``Generator.__init__`` clamps ``self.max_batch_size = min(self.max_batch_size,
  cache.num_slots)`` (exllamav3/generator/generator.py:233).
- ``tokenizer.encode`` returns a ``[1, N]`` long tensor
  (exllamav3/tokenizer/tokenizer.py:420), so ``len()`` of it is 1.
- ``Job.__init__`` takes ``input_ids: torch.Tensor | list[torch.Tensor]``,
  shape ``[1, seq_len]``, CPU (exllamav3/generator/job.py:49, :146, :206).

The stubs reenact the defect chain the GPU box measured; monkeypatch.setitem
restores ``sys.modules`` after each test.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from exl3_serve.engine_exl3 import Exl3Engine


# ---------------------------------------------------------------- stubs

class StubCache:
    """cache.py:161 -- self.num_slots = max_batch_size."""

    def __init__(self, max_batch_size: int) -> None:
        self.num_slots = max_batch_size
        self.max_num_tokens = 4096
        self.layers = None


class StubGenerator:
    """generator.py:233 -- the generator clamps to the cache's slot count."""

    def __init__(self, model=None, cache=None, tokenizer=None, draft_model=None,
                 draft_cache=None, max_batch_size=1, num_draft_tokens=None,
                 dynamic_draft_tokens=False) -> None:
        self.max_batch_size = min(max_batch_size, cache.num_slots)


class StubTokenizer:
    eos_token_id = 7


class StubTokenTensor:
    """What tokenizer.encode returns: a [1, N] tensor whose len() is 1."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def tolist(self) -> list[list[int]]:
        return [list(self._ids)]

    def __len__(self) -> int:
        return 1  # the batch dimension -- the defect


def _install_exllamav3(monkeypatch, mi_init) -> None:
    ex = types.ModuleType("exllamav3")
    ex.Generator = StubGenerator
    ex.model_init = SimpleNamespace(init=mi_init)
    monkeypatch.setitem(sys.modules, "exllamav3", ex)
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    monkeypatch.setitem(sys.modules, "torch", torch)


def _model_dir(tmp_path) -> str:
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "chat_template.jinja").write_text(
        "{%- for m in messages %}{{ m['content'] }}{% endfor %}", encoding="utf-8")
    return str(d)


def _load_with_stubs(monkeypatch, tmp_path, parallel=2, ambs=None,
                     ambs_attr="autosplit_max_batch_size"):
    """Exl3Engine.load() against the stubs; returns (engine, what model_init
    saw as the cache's max_batch_size). ``parallel=None`` omits the attribute
    (the getattr(args, "parallel", 2) default path). ``ambs_attr`` is the
    attribute the stub model_init reads -- pass a different name to model a
    future exllamav3 that renamed ``-ambs`` so the server's raise does not
    take and the cache stays at the one-slot default."""
    seen = {}

    def mi_init(args, progress=False, quiet=True):
        # model_init.py:292 -- the cache is built from args.<ambs flag>
        seen["ambs"] = getattr(args, ambs_attr, 1)
        return ([], SimpleNamespace(max_seq_len=4096),
                StubCache(seen["ambs"]), StubTokenizer())

    _install_exllamav3(monkeypatch, mi_init)
    overrides = {} if ambs is None else {"autosplit_max_batch_size": ambs}
    if parallel is not None:
        overrides["parallel"] = parallel
    args = SimpleNamespace(model_dir=_model_dir(tmp_path), cache_size=1024,
                           draft_n=None, dyn_draft=False, **overrides)
    return Exl3Engine.load(args), seen


# ---------------------------------------------------------------- defect 1
# --parallel N must produce N served slots, not a cache-sized clamp

def test_load_raises_when_generator_clamped_below_parallel(monkeypatch, tmp_path):
    # The assertion's reason for being: the cache stays at the one-slot
    # default whenever the server's -ambs raise does not take (here: the stub
    # model_init reads a renamed flag), the generator clamps to it, and load
    # must fail loudly instead of serving two serialized streams.
    with pytest.raises(RuntimeError, match=r"--parallel 2 but the generator "
                                           r"clamped to 1 \(cache slots 1\); raise -ambs"):
        _load_with_stubs(monkeypatch, tmp_path, parallel=2,
                         ambs_attr="autosplit_max_batch_size_v2")


def test_load_sizes_cache_for_parallel(monkeypatch, tmp_path):
    eng, seen = _load_with_stubs(monkeypatch, tmp_path, parallel=2)
    assert seen["ambs"] == 2             # model_init got the raised -ambs
    assert eng.gen.max_batch_size == 2   # nothing clamped


def test_load_keeps_larger_user_ambs(monkeypatch, tmp_path):
    # a larger -ambs the user typed is never shrunk to --parallel
    eng, seen = _load_with_stubs(monkeypatch, tmp_path, parallel=2, ambs=4)
    assert seen["ambs"] == 4
    assert eng.gen.max_batch_size == 2   # min(2, 4): still serves --parallel


def test_load_defaults_parallel_when_attribute_absent(monkeypatch, tmp_path):
    # getattr(args, "parallel", 2) or 2 keeps working on a bare namespace
    eng, seen = _load_with_stubs(monkeypatch, tmp_path, parallel=None)
    assert seen["ambs"] == 2
    assert eng.gen.max_batch_size == 2


# ---------------------------------------------------------------- defect 2
# encode() returns a flat list[int]; submit() converts for Job

def _engine_with(tokenizer, gen=None) -> Exl3Engine:
    return Exl3Engine(
        model_dir="m", model=None, cache=None, tokenizer=tokenizer,
        gen=gen or SimpleNamespace(enqueue=lambda job: None),
        template=None, chat_template_raw="", version=None, n_ctx=8,
        gen_cfg={}, stop_ids=set(), engine_block={})


def test_encode_returns_flat_token_list():
    tok = SimpleNamespace(
        encode=lambda text, encode_special_tokens: StubTokenTensor([5, 6, 7, 8]))
    ids = _engine_with(tok).encode("prompt with four words")
    assert isinstance(ids, list)
    assert ids == [5, 6, 7, 8]  # N tokens, not the [1, N] tensor's len() of 1


def test_submit_passes_job_a_1_by_n_long_tensor(monkeypatch):
    made = {}
    torch = types.ModuleType("torch")
    torch.long = "long-dtype"

    def _tensor(data, dtype=None):
        made["call"] = (data, dtype)
        return ("tensor", data, dtype)

    torch.tensor = _tensor
    jobs = []
    ex = types.ModuleType("exllamav3")
    ex.GreedySampler = lambda: "greedy"

    class Job:
        def __init__(self, **kw):
            jobs.append(kw)

    ex.Job = Job
    monkeypatch.setitem(sys.modules, "exllamav3", ex)
    monkeypatch.setitem(sys.modules, "torch", torch)
    tok = SimpleNamespace(eos_token_id=7,
                          encode=lambda *a, **k: StubTokenTensor([5, 6, 7]))
    _engine_with(tok).submit([5, 6, 7], 4, None, [])
    # Job takes a [1, N] CPU tensor (job.py:49/:206); the conversion happens
    # in submit, in one place, from the flat list
    assert made["call"] == ([[5, 6, 7]], "long-dtype")
    assert jobs[0]["input_ids"] == ("tensor", [[5, 6, 7]], "long-dtype")
