"""Pure measurements for the ``/props`` ``engine`` block.

Nothing here imports exllamav3 or torch: safetensors headers are read with
``struct`` + ``json``, configs with ``json``, and the placement helpers only
transform numbers the runtime walk in ``engine_exl3`` produced. That keeps the
whole block unit-testable on a machine without a GPU.

The recorder's contract, enforced throughout: **every value is what the engine
reports or what is on disk.** A key that cannot be measured is omitted, never
filled with 0 or a guess; a 0 that was actually measured is printed as 0.

Tensor classes are the recorder's classification, shared verbatim by the disk
walk and the runtime walk so the two engines' cards agree: the router
(``.mlp.gate.*``) and the shared expert (``.mlp.shared_experts.*``) count as
``experts``, not ``ffn``.
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
from typing import Any, Optional

# EXL3 quantized-linear tensor quads; the linear's parameter count is
# len(suh) x len(svh) (the trellis/mul1 sides carry no parameters).
_QUANT_SUFFIXES = ("suh", "svh", "trellis", "mul1")

_MOE_RE = re.compile(r"\.mlp\.(?:experts|shared_experts|gate)\b")
# a routed expert's tensors sit under experts.<e> directly or one level deeper
# (experts.<e>.<proj>.<quad>, the EXL3 layout measured on GLM-5.3-Flash)
_ROUTED_EXPERT_RE = re.compile(r"\.mlp\.experts\.(\d+)(?:\.|$)")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def classify_tensor(key: str) -> str:
    """Class of a tensor (or module) key: embeddings / output / experts /
    ffn / attention / other. Tolerates both full tensor keys
    (``...mlp.gate.weight``) and module keys (``...mlp.gate``)."""
    if "embed_tokens" in key:
        return "embeddings"
    if key.startswith("lm_head") or "shared_head" in key:
        return "output"
    if _MOE_RE.search(key):
        return "experts"
    if ".mlp" in key:
        return "ffn"
    if ".self_attn" in key:
        return "attention"
    return "other"


# --------------------------------------------------------------------------
# safetensors headers (disk only; no safetensors import)
# --------------------------------------------------------------------------

def read_safetensors_header(path: str) -> dict:
    """The tensor entries of a safetensors file: name -> {dtype, shape,
    data_offsets}. ``__metadata__`` is dropped; offsets stay raw."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise ValueError(f"{path}: too short for a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        header = json.loads(f.read(n))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def load_tensors(model_dir: str) -> dict[str, dict]:
    """All tensors of the directory's non-hidden ``.safetensors`` files,
    merged: name -> {"shape": [...], "nbytes": <on-disk byte size>}."""
    out: dict[str, dict] = {}
    for name in sorted(os.listdir(model_dir)):
        if name.startswith(".") or not name.endswith(".safetensors"):
            continue
        for k, meta in read_safetensors_header(os.path.join(model_dir, name)).items():
            if k in out:
                continue
            offsets = meta.get("data_offsets") or [0, 0]
            out[k] = {"shape": list(meta.get("shape") or []),
                      "nbytes": int(offsets[1]) - int(offsets[0])}
    return out


def disk_stats(model_dir: str) -> tuple[int, int]:
    """(files, bytes) of the non-hidden top-level regular files."""
    files = total = 0
    for entry in os.scandir(model_dir):
        if entry.name.startswith(".") or not entry.is_file():
            continue
        files += 1
        total += entry.stat().st_size
    return files, total


# --------------------------------------------------------------------------
# config files
# --------------------------------------------------------------------------

def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _text_config(cfg: Optional[dict]) -> dict:
    if isinstance(cfg, dict):
        tc = cfg.get("text_config")
        if isinstance(tc, dict):
            return tc
        return cfg
    return {}


def _int_or_none(v: Any) -> Optional[int]:
    if isinstance(v, int) and not isinstance(v, bool):
        return v
    return None


def quant_string(model_dir: str) -> Optional[str]:
    """``EXL3 <bits> bpw · head <head_bits>`` from quantization_config.json,
    bits printed exactly as written in the file (4.05 stays ``4.05``); the
    head part is omitted when the file carries no ``head_bits``."""
    path = os.path.join(model_dir, "quantization_config.json")
    cfg = _read_json(path)
    if not isinstance(cfg, dict):
        return None
    bits: Optional[str] = None
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        m = re.search(r'"bits"\s*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)', raw)
        if m:
            bits = m.group(1)
    except OSError:
        pass
    if bits is None:
        v = cfg.get("bits")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            bits = "%g" % v
    if bits is None:
        return None
    head = cfg.get("head_bits")
    if isinstance(head, (int, float)) and not isinstance(head, bool):
        return f"EXL3 {bits} bpw · head {head:.1f}"
    return f"EXL3 {bits} bpw"


# --------------------------------------------------------------------------
# parameter count and active bytes per token (main text model only)
# --------------------------------------------------------------------------

def _groups(tensors: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """name -> members by last dotted component (the quant quads and the
    plain weights of one module land in the same group)."""
    groups: dict[str, dict[str, dict]] = {}
    for name, meta in tensors.items():
        prefix, _, suffix = name.rpartition(".")
        groups.setdefault(prefix, {})[suffix] = meta
    return groups


def _is_main_text(key: str, n_layers: Optional[int]) -> bool:
    """False for vision tensors and for the MTP layer(s): layer index at or
    above num_hidden_layers (the disk layout stores the MTP layer as
    ``layers.<n_layers>.*`` in mtp.safetensors)."""
    if ".visual." in key:
        return False
    if n_layers is not None:
        m = _LAYER_RE.search(key)
        if m and int(m.group(1)) >= n_layers:
            return False
    return True


def count_params(tensors: dict[str, dict], n_layers: Optional[int] = None) -> int:
    """Parameters of the main text model from the safetensors headers:
    quantized linear quads count as len(suh) x len(svh), every other tensor
    as the product of its shape."""
    total = 0
    for prefix, members in _groups(tensors).items():
        if not _is_main_text(prefix, n_layers):
            continue
        suh, svh = members.get("suh"), members.get("svh")
        if suh and svh:
            s1 = (suh["shape"] or [1])[0]
            s2 = (svh["shape"] or [1])[0]
            total += s1 * s2
        else:
            for meta in members.values():
                total += math.prod(meta["shape"]) if meta["shape"] else 1
    return total


def active_bytes_per_token(tensors: dict[str, dict], n_layers: Optional[int] = None,
                           vocab_size: Optional[int] = None,
                           top_k: Optional[int] = None) -> Optional[int | float]:
    """Bytes touched per decoded token, from disk: every non-routed-expert
    text tensor in full, one embedding row (embed bytes / vocab), and
    top_k x per-expert bytes for each MoE layer. None when the weighting
    inputs (vocab, top_k) are not on disk."""
    if not tensors or not vocab_size or vocab_size <= 0 or not top_k:
        return None
    total: Any = 0
    moe_experts: dict[str, dict[int, int]] = {}
    for prefix, members in _groups(tensors).items():
        if not _is_main_text(prefix, n_layers):
            continue
        nbytes = sum(m["nbytes"] for m in members.values())
        m = _ROUTED_EXPERT_RE.search(prefix)
        if m:
            parent = prefix[:m.start() + len(".mlp.experts")]
            per = moe_experts.setdefault(parent, {})
            e = int(m.group(1))
            per[e] = per.get(e, 0) + nbytes  # every projection of the expert
            continue
        if classify_tensor(prefix) == "embeddings":
            row, rem = divmod(nbytes, vocab_size)
            total += row if rem == 0 else nbytes / vocab_size
        else:
            total += nbytes
    for per_expert in moe_experts.values():
        total += top_k * per_expert[min(per_expert)]
    return total


# --------------------------------------------------------------------------
# block assembly
# --------------------------------------------------------------------------

def model_block(model_dir: str) -> Optional[dict]:
    """The ``engine.model`` object for a model directory, or None when the
    directory does not exist. Unmeasurable keys are omitted, never zeroed."""
    if not os.path.isdir(model_dir):
        return None
    block: dict = {"format": "exl3"}
    cfg = _read_json(os.path.join(model_dir, "config.json"))
    tc = _text_config(cfg)
    if isinstance(cfg, dict):
        archs = cfg.get("architectures")
        if isinstance(archs, list) and archs:
            block["arch"] = str(archs[0])
    quant = quant_string(model_dir)
    if quant is not None:
        block["quant"] = quant
    files, nbytes = disk_stats(model_dir)
    block["files"] = files
    block["bytes"] = nbytes
    tensors = load_tensors(model_dir)
    n_layers = _int_or_none(tc.get("num_hidden_layers"))
    if tensors:
        block["params"] = count_params(tensors, n_layers)
    for key, names in (("n_layers", ("num_hidden_layers",)),
                       ("n_experts", ("n_routed_experts", "num_experts")),
                       ("n_experts_used", ("num_experts_per_tok",)),
                       ("ctx_train", ("max_position_embeddings",))):
        for name in names:
            v = _int_or_none(tc.get(name))
            if v is not None:
                block[key] = v
                break
    active = active_bytes_per_token(
        tensors, n_layers,
        _int_or_none(tc.get("vocab_size")), block.get("n_experts_used"))
    if active is not None:
        block["active_bytes_per_token"] = active
    return block


def finalize_placement(by_device: dict[str, dict[str, int]],
                       vram_kv_bytes: Optional[int] = None,
                       cpu_numbers_exact: bool = True) -> Optional[dict]:
    """The ``engine.placement`` object from a raw runtime walk:
    ``{"cuda:0": {"experts": n, ...}, "cpu": {...}}``.

    Classes with 0 bytes are omitted, devices with no classes are omitted,
    and each device's ``bytes`` is the sum of its classes (the invariant the
    recorder checks). ``cpu_numbers_exact=False`` omits the CPU device's
    ``bytes`` and ``classes`` entirely: the CPU-worker expert split moves at
    run time, so only a uniform expert size makes the number exact.
    """
    devices: list[dict] = []
    for dev in sorted(by_device):
        classes = by_device[dev] or {}
        name = "CPU" if dev in ("cpu", "CPU") else re.sub(r"^cuda:", "GPU", dev)
        if not cpu_numbers_exact and name == "CPU":
            devices.append({"device": name})
            continue
        pruned = {c: b for c, b in sorted(classes.items()) if b > 0}
        if not pruned:
            continue
        devices.append({"device": name, "bytes": sum(pruned.values()),
                        "classes": pruned})
    if not devices and not vram_kv_bytes:
        return None
    out: dict = {"devices": devices}
    if vram_kv_bytes is not None:
        out["vram_kv_bytes"] = vram_kv_bytes
    return out


# Flags that do not change what the engine does: where it listens, and the model
# path (the card has its own model row). Both "--flag value" and "--flag=value".
_NON_BEHAVIOURAL_FLAGS = {"--host", "--port", "-m", "--model_dir", "--model"}


def engine_args(argv: list[str]) -> list[str]:
    """argv for the card's FLAGS row: every behaviour-changing flag verbatim, in
    order; host, port and the model path dropped with their values."""
    out: list[str] = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        name = a.split("=", 1)[0]
        if name in _NON_BEHAVIOURAL_FLAGS:
            skip = "=" not in a
            continue
        out.append(a)
    return out


def draft_block(mtp: bool, draft_model_dir: Optional[str],
                n_max: Optional[int]) -> Optional[dict]:
    """``engine.draft``: ``model`` is "mtp" for the model's own MTP head, else the
    draft model directory's name; ``n_max`` is the most tokens drafted per step as
    the server configured the generator (omitted when it left the engine default).
    None when no draft is configured."""
    if mtp:
        block: dict = {"model": "mtp"}
    elif draft_model_dir:
        block = {"model": os.path.basename(os.path.normpath(draft_model_dir))}
    else:
        return None
    if isinstance(n_max, int) and not isinstance(n_max, bool) and n_max > 0:
        block["n_max"] = n_max
    return block


def build_engine_block(name: str, version: Optional[str], args: Any,
                       model_dir: Optional[str] = None) -> dict:
    """The ``/props`` ``engine`` object: identity plus the disk-measured
    model block (omitted when the directory does not exist). The runtime
    walk attaches ``placement`` to the returned dict."""
    block: dict = {"name": name}
    if version:
        block["version"] = version
    block["args"] = [str(a) for a in (args or [])]
    if model_dir:
        model = model_block(model_dir)
        if model is not None:
            block["model"] = model
    return block
