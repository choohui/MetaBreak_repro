"""Orchestrator — build data, calibrate + evaluate each defense per model, report.

Per model the victim is loaded once; stage 02 returns the in-memory prepared
defenses, which stage 03 consumes in the same process. Use ``--smoke`` for a
model-free run on the mock model.

Examples
--------
    python experiments_defense/run_all.py --smoke
    python experiments_defense/run_all.py --models llama \
        --model_path llama=/models/Llama-3.1-8B-Instruct \
        --guard_model /models/Llama-Guard-3-8B --n_gsm8k 50
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import add_common_args, config_from_args, ExpConfig  # noqa: E402


def load_stage(num: str):
    path = next(HERE.glob(f"stages/{num}_*.py"))
    spec = importlib.util.spec_from_file_location(f"stage_{num}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_model(cfg: ExpConfig, model: str):
    if cfg.smoke:
        from core.mock import build_mock_loaded_model
        return build_mock_loaded_model()
    from core.model import load_model
    if model not in cfg.model_paths:
        raise SystemExit(f"no --model_path for {model!r} (pass --model_path {model}=/path/to/checkpoint)")
    return load_model(cfg.model_paths[model], model, cfg.dtype, cfg.device)


def free_model(*objs) -> None:
    """Drop references and reclaim GPU memory so the next model can load.

    Single-GPU multi-model runs are sequential: the victim model (and any guard
    model held inside a defense) MUST be freed before the next victim loads, or
    the second ``from_pretrained`` OOMs."""
    import gc
    for o in objs:
        m = getattr(o, "model", None)
        if m is not None and hasattr(m, "to"):
            try:
                m.to("cpu")
            except Exception:
                pass
        del o
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run(cfg: ExpConfig) -> dict:
    s01, s02, s03, s04 = (load_stage(n) for n in ("01", "02", "03", "04"))
    print(f"[run_all] {cfg.slug()} :: models={cfg.models} defenses={cfg.defenses} smoke={cfg.smoke}")
    s01.run(cfg)
    done, failed = [], {}
    for model in cfg.models:
        print(f"[run_all] === model: {model} ===")
        lm = None
        defenses = {}
        try:
            lm = get_model(cfg, model)
            _, defenses = s02.run(cfg, lm, model)
            s03.run(cfg, lm, model, defenses)
            done.append(model)
        except Exception as e:  # one model failing shouldn't sink the rest
            failed[model] = repr(e)
            print(f"[run_all] !! model {model} FAILED: {e!r}")
        finally:
            free_model(lm, *defenses.values())   # release victim + any guard model
    out = s04.run(cfg)                            # report over whatever finished
    out["done"], out["failed"] = done, failed
    if failed:
        print(f"[run_all] completed={done}  failed={list(failed)}")
    print(f"[run_all] report -> {out['report_path']}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="experiments_defense — §4 defense comparison")
    add_common_args(p)
    cfg = config_from_args(p.parse_args())
    run(cfg)


if __name__ == "__main__":
    main()
