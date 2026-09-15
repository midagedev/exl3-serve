"""engine_info: exact, hand-computed numbers on a synthetic model directory.

The synthetic dir has real safetensors framing (8-byte little-endian header
length + JSON header + zero data), a config.json with a text_config, a
quantization_config.json, a generation_config.json, a hidden .lock file, one
MTP-layer tensor and one vision tensor -- everything the disk walk must
exclude or include. All expected numbers below are computed by hand.
"""
from __future__ import annotations

import json
import os
import struct

from exl3_serve import engine_info

DTYPE_SIZE = {"F32": 4, "F16": 2, "U8": 1}


def write_safetensors(path: str, tensors: list[tuple[str, str, list[int]]]) -> int:
    """Write a minimal but real safetensors file; returns its byte size."""
    header: dict = {}
    data = b""
    for name, dtype, shape in tensors:
        n = DTYPE_SIZE[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [len(data), len(data) + n]}
        data += b"\0" * n
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(data)
    return 8 + len(hb) + len(data)


def _write_json(path: str, obj) -> int:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return os.path.getsize(path)


# The synthetic model: n_layers 2 (a dense layer and a MoE layer), 3 routed
# experts of which 2 are active per token, vocab 100. Hand-computed:
#   params = 800 embed + 8 norm + 256 attn + 128 ffn + 24 router + 3 bias
#            + 64 shared + 3x64 experts + 800 lm_head = 2275
#   active = 32 embed row (3200 B / vocab 100) + 32 norm + 48 attn + 32 ffn
#            + 108 router+bias + 24 shared + 2x24 experts + 116 lm_head = 440
MAIN_TENSORS: list[tuple[str, str, list[int]]] = [
    ("model.language_model.embed_tokens.weight", "F32", [100, 8]),
    ("model.language_model.layers.0.input_layernorm.weight", "F32", [8]),
    ("model.language_model.layers.0.self_attn.q_proj.suh", "U8", [32]),
    ("model.language_model.layers.0.self_attn.q_proj.svh", "U8", [8]),
    ("model.language_model.layers.0.self_attn.q_proj.trellis", "U8", [4]),
    ("model.language_model.layers.0.self_attn.q_proj.mul1", "F16", [2]),
    ("model.language_model.layers.0.mlp.up_proj.suh", "U8", [16]),
    ("model.language_model.layers.0.mlp.up_proj.svh", "U8", [8]),
    ("model.language_model.layers.0.mlp.up_proj.trellis", "U8", [4]),
    ("model.language_model.layers.0.mlp.up_proj.mul1", "F16", [2]),
    ("model.language_model.layers.1.mlp.gate.weight", "F32", [3, 8]),
    ("model.language_model.layers.1.mlp.gate.e_score_correction_bias", "F32", [3]),
    ("model.language_model.layers.1.mlp.shared_experts.up_proj.suh", "U8", [8]),
    ("model.language_model.layers.1.mlp.shared_experts.up_proj.svh", "U8", [8]),
    ("model.language_model.layers.1.mlp.shared_experts.up_proj.trellis", "U8", [4]),
    ("model.language_model.layers.1.mlp.shared_experts.up_proj.mul1", "F16", [2]),
    ("lm_head.suh", "U8", [100]),
    ("lm_head.svh", "U8", [8]),
    ("lm_head.trellis", "U8", [4]),
    ("lm_head.mul1", "F16", [2]),
    ("model.visual.patch_emb.weight", "F32", [4, 4]),  # vision: excluded
]
for _e in range(3):  # the routed experts of the MoE layer
    for _s, _dt, _sh in (("suh", "U8", [8]), ("svh", "U8", [8]),
                         ("trellis", "U8", [4]), ("mul1", "F16", [2])):
        MAIN_TENSORS.append(
            (f"model.language_model.layers.1.mlp.experts.{_e}.{_s}", _dt, _sh))

# the MTP layer lives at layers.<n_layers> in mtp.safetensors: excluded
MTP_TENSORS = [
    ("model.language_model.layers.2.mlp.experts.0.suh", "U8", [8]),
    ("model.language_model.layers.2.mlp.experts.0.svh", "U8", [8]),
    ("model.language_model.layers.2.mlp.experts.0.trellis", "U8", [4]),
    ("model.language_model.layers.2.mlp.experts.0.mul1", "F16", [2]),
]


def build_synthetic_model(model_dir: str) -> dict:
    """Create the synthetic model directory; returns the expected block."""
    os.makedirs(model_dir, exist_ok=True)
    main_size = write_safetensors(os.path.join(model_dir, "model-00001-of-00001.safetensors"),
                                  MAIN_TENSORS)
    mtp_size = write_safetensors(os.path.join(model_dir, "mtp.safetensors"),
                                 MTP_TENSORS)
    cfg_size = _write_json(os.path.join(model_dir, "config.json"), {
        "architectures": ["SynthForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 2,
            "n_routed_experts": 3,
            "num_experts_per_tok": 2,
            "max_position_embeddings": 4096,
            "vocab_size": 100,
        },
    })
    quant_size = _write_json(os.path.join(model_dir, "quantization_config.json"),
                             {"bits": 4.05, "head_bits": 6, "codebook": "mul1"})
    gen_size = _write_json(os.path.join(model_dir, "generation_config.json"),
                           {"eos_token_id": [5, 6], "top_p": 0.9})
    with open(os.path.join(model_dir, ".lock"), "wb") as f:  # hidden: excluded
        f.write(b"lockfile")
    return {
        "format": "exl3",
        "arch": "SynthForConditionalGeneration",
        "quant": "EXL3 4.05 bpw · head 6.0",
        "bytes": main_size + mtp_size + cfg_size + quant_size + gen_size,
        "files": 5,
        "params": 2275,
        "n_layers": 2,
        "n_experts": 3,
        "n_experts_used": 2,
        "ctx_train": 4096,
        "active_bytes_per_token": 440,
    }


def test_model_block_exact_numbers(tmp_path):
    expected = build_synthetic_model(str(tmp_path / "model"))
    block = engine_info.model_block(str(tmp_path / "model"))
    assert block == expected


def test_build_engine_block_with_model(tmp_path):
    model_dir = str(tmp_path / "model")
    expected = build_synthetic_model(model_dir)
    engine = engine_info.build_engine_block(
        "exllamav3", "1.5.0", ["-m", model_dir, "-gs", "44,21"], model_dir=model_dir)
    assert engine == {"name": "exllamav3", "version": "1.5.0",
                      "args": ["-m", model_dir, "-gs", "44,21"],
                      "model": expected}


def test_build_engine_block_without_model_dir(tmp_path):
    engine = engine_info.build_engine_block(
        "exllamav3", "1.5.0", ["--fake"], model_dir=str(tmp_path / "nope"))
    assert engine == {"name": "exllamav3", "version": "1.5.0",
                      "args": ["--fake"]}  # no "model" key: nothing measured


def test_bits_printed_as_written(tmp_path):
    d = tmp_path / "q"
    d.mkdir()
    (d / "quantization_config.json").write_text(
        '{"bits": 4.50, "codebook": "mul1"}', encoding="utf-8")
    assert engine_info.quant_string(str(d)) == "EXL3 4.50 bpw"  # no head_bits
    (d / "quantization_config.json").write_text(
        '{"bits": 4, "head_bits": 8}', encoding="utf-8")
    assert engine_info.quant_string(str(d)) == "EXL3 4 bpw · head 8.0"
    assert engine_info.quant_string(str(tmp_path / "missing")) is None


def test_class_mapping_recorder_classification():
    c = engine_info.classify_tensor
    # the router and the shared expert are "experts", not "ffn"
    assert c("model.x.layers.1.mlp.gate.weight") == "experts"
    assert c("model.x.layers.1.mlp.gate.e_score_correction_bias") == "experts"
    assert c("model.x.layers.1.mlp.shared_experts.up_proj.suh") == "experts"
    assert c("model.x.layers.1.mlp.experts.0.suh") == "experts"
    assert c("model.x.layers.1.mlp.experts.0") == "experts"
    # module-key forms (no trailing component) classify identically
    assert c("model.x.layers.1.mlp.gate") == "experts"
    assert c("model.x.layers.1.mlp.shared_experts") == "experts"
    # dense feed-forward of the first dense layers
    assert c("model.x.layers.0.mlp.up_proj.suh") == "ffn"
    assert c("model.x.layers.0.mlp.gate_proj.weight") == "ffn"  # not the router
    assert c("model.x.layers.0.self_attn.q_proj.suh") == "attention"
    assert c("model.x.embed_tokens.weight") == "embeddings"
    assert c("lm_head.suh") == "output"
    assert c("model.x.shared_head.norm.weight") == "output"
    assert c("model.x.layers.0.input_layernorm.weight") == "other"
    assert c("hc_expand.0") == "other"


def test_placement_invariant_and_zero_pruning():
    by = {"cuda:0": {"experts": 100, "attention": 50, "ffn": 0},
          "cpu": {"embeddings": 30},
          "cuda:1": {}}  # no bytes: the device is omitted
    p = engine_info.finalize_placement(by, vram_kv_bytes=200)
    assert p == {"devices": [
        {"device": "CPU", "bytes": 30, "classes": {"embeddings": 30}},
        {"device": "GPU0", "bytes": 150, "classes": {"attention": 50, "experts": 100}},
    ], "vram_kv_bytes": 200}
    for dev in p["devices"]:
        assert dev["bytes"] == sum(dev["classes"].values())
        assert all(v > 0 for v in dev["classes"].values())


def test_placement_cpu_numbers_omitted_when_not_exact():
    by = {"cuda:0": {"attention": 10}, "cpu": {"embeddings": 30}}
    p = engine_info.finalize_placement(by, vram_kv_bytes=0, cpu_numbers_exact=False)
    cpu, gpu = p["devices"]  # "cpu" sorts before "cuda:0"
    assert cpu == {"device": "CPU"}  # bytes and classes omitted, not guessed
    assert gpu == {"device": "GPU0", "bytes": 10, "classes": {"attention": 10}}
    assert p["vram_kv_bytes"] == 0  # a measured zero is printed


def test_placement_omitted_when_nothing_measurable():
    assert engine_info.finalize_placement({}) is None
    assert engine_info.finalize_placement({"cuda:1": {}}, vram_kv_bytes=None) is None


def test_no_numeric_zero_anywhere_in_engine_block(tmp_path):
    model_dir = str(tmp_path / "model")
    expected = build_synthetic_model(model_dir)
    engine = engine_info.build_engine_block(
        "exllamav3", "1.5.0", ["-m", model_dir], model_dir=model_dir)

    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)
        else:
            yield node

    values = list(walk(engine))
    assert values  # the block is populated
    assert all(v != 0 for v in values if isinstance(v, (int, float)))
    assert engine["model"] == expected


def test_active_bytes_and_params_with_projection_level_expert_keys(tmp_path):
    """The real EXL3 layout names routed experts ``experts.<e>.<proj>.<quad>``
    (measured on GLM-5.3-Flash 4.05 bpw), one level deeper than the fixture
    above. Every projection of an expert must aggregate into that expert, and
    only top_k experts per layer count toward active bytes."""
    d = tmp_path / "proj-level"
    d.mkdir()

    def quad(p):  # 8 + 4 + 16 + 2x2 = 32 bytes on disk, params 8 x 4 = 32
        return [(f"{p}.suh", "U8", [8]), (f"{p}.svh", "U8", [4]),
                (f"{p}.trellis", "U8", [16]), (f"{p}.mul1", "F16", [2])]

    tensors = [("model.language_model.embed_tokens.weight", "F32", [10, 4])]  # 160 B, 16 B a row
    for e in range(4):
        for proj in ("up_proj", "down_proj"):
            tensors += quad(f"model.language_model.layers.0.mlp.experts.{e}.{proj}")
    tensors += [("model.language_model.layers.0.mlp.gate.weight", "F32", [4, 4])]  # 64 B, 16 params
    tensors += quad("lm_head")
    write_safetensors(str(d / "model.safetensors"), tensors)
    t = engine_info.load_tensors(str(d))
    # one embedding row + router + lm_head in full + top_k=2 experts x (up 32 + down 32)
    assert engine_info.active_bytes_per_token(t, 1, 10, 2) == 16 + 64 + 32 + 2 * 64
    # embed 40 + 4 experts x 2 projections x 32 + router 16 + lm_head 32
    assert engine_info.count_params(t, 1) == 40 + 8 * 32 + 16 + 32


def test_engine_args_drop_non_behavioural_flags():
    """The card's FLAGS row carries what changes the engine's behaviour, in argv
    order; host, port and the model path (which has its own row) are dropped,
    in both ``--flag value`` and ``--flag=value`` forms."""
    argv = ["-m", "/models/X", "-gs", "44,21", "--host", "127.0.0.1", "--port=8089",
            "-mcs", "185", "-mct", "32", "-cs", "32768", "-mtp", "--model_dir=/models/X",
            "--draft-n", "1", "--parallel", "2"]
    assert engine_info.engine_args(argv) == [
        "-gs", "44,21", "-mcs", "185", "-mct", "32", "-cs", "32768", "-mtp",
        "--draft-n", "1", "--parallel", "2"]


def test_draft_block():
    """engine.draft fills the card's Draft row: the model verbatim ("mtp" for the
    model's own MTP head, else the draft model directory's name) and n_max, the
    most tokens drafted per step, omitted when the server does not set it."""
    assert engine_info.draft_block(True, None, 1) == {"model": "mtp", "n_max": 1}
    assert engine_info.draft_block(False, "/models/Some-Draft-exl3/", 4) == {"model": "Some-Draft-exl3", "n_max": 4}
    assert engine_info.draft_block(True, None, None) == {"model": "mtp"}
    assert engine_info.draft_block(False, None, 3) is None
