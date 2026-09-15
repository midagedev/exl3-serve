"""Command line: ``exl3-serve -m <model_dir> [--fake] [options]``.

Every exllamav3 ``model_init`` flag (``-gs``, ``-mcs``, ``-mcl``, ``-mct``,
``-cs``, ``-mtp``, ...) is accepted verbatim: when exllamav3 is importable we
let ``model_init.add_args`` extend the parser exactly like
reference/exl3-bench3.py does -- which is also what provides ``-m`` itself.
Without exllamav3 only ``--fake`` is useful, and we provide a fallback ``-m``
so the fake smoke server runs anywhere (including this GPU-less Mac).
"""
from __future__ import annotations

import argparse
import sys

from . import __version__


def _model_init():
    try:
        from exllamav3 import model_init
    except Exception:
        return None
    return model_init


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="exl3-serve",
        description="llama-server-compatible HTTP front for one ExLlamaV3 model",
    )
    p.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080, help="listen port (default: 8080)")
    p.add_argument("--parallel", type=int, default=2,
                   help="max concurrent streams / slots (default: 2)")
    p.add_argument("--alias", default=None,
                   help="model name reported to clients (default: model dir basename)")
    p.add_argument("--reasoning-effort", default="high",
                   help="server default reasoning_effort passed to the chat template (default: high)")
    p.add_argument("--draft-n", type=int, default=None,
                   help="MTP draft tokens per step; defaults to 1 when -mtp is given "
                        "(depth 1 measured best with CPU-offloaded experts)")
    p.add_argument("--dyn-draft", action="store_true",
                   help="enable dynamic draft tokens on the Generator")
    p.add_argument("--no-think-split", action="store_true",
                   help="do not split <think>...</think> into reasoning_content; "
                        "send everything as content")
    p.add_argument("--max-tokens-cap", type=int, default=4096,
                   help="clamp for request max_tokens/n_predict (default: 4096)")
    p.add_argument("--fake", action="store_true",
                   help="serve a deterministic fake engine (no exllamav3, no GPU); "
                        "for smoke tests")
    p.add_argument("--version", action="version", version=f"exl3-serve {__version__}")
    mi = _model_init()
    if mi is None:
        p.add_argument("-m", "--model", dest="model_dir", default=None,
                       help="model directory (exllamav3 flag pass-through unavailable: "
                            "exllamav3 is not installed)")
    else:
        # provides -m and every exllamav3 tuning flag, verbatim
        mi.add_args(p, cache=True, add_draft_model_args=True, default_cache_size=32768)
    return p


def main(argv=None) -> int:
    from aiohttp import web

    args = build_parser().parse_args(argv)
    if args.draft_n is None and getattr(args, "mtp", None):
        args.draft_n = 1

    from .server import Options, create_app
    options = Options(
        parallel=args.parallel,
        alias=args.alias,
        reasoning_effort=args.reasoning_effort,
        no_think_split=args.no_think_split,
        max_tokens_cap=args.max_tokens_cap,
    )
    if args.fake:
        from .engine_fake import FakeEngine
        engine = FakeEngine(model_dir=args.model_dir or "/tmp/fake-model")
        app = create_app(options, engine=engine)
    else:
        if _model_init() is None:
            print("exl3-serve: exllamav3 is not installed; "
                  "rerun with --fake for the offline smoke server", file=sys.stderr)
            return 2
        if not args.model_dir:
            print("exl3-serve: -m <model_dir> is required", file=sys.stderr)
            return 2
        from .engine_exl3 import Exl3Engine
        app = create_app(options, loader=lambda: Exl3Engine.load(args))
    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
