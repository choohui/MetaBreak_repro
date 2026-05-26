"""Download README-specified MetaBreak models from Hugging Face.

The script reads a Hugging Face token from `tokenlist.txt` without printing it,
downloads the README attack model, and optionally downloads Llama Guard 3.
It writes the resolved local snapshot paths to JSON for downstream runners.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


README_MODEL_REPO = "meta-llama/Llama-3.1-8B-Instruct"
README_GUARD_REPO = "meta-llama/Llama-Guard-3-8B"


def read_token(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if token:
            return token
    raise RuntimeError(f"No non-empty Hugging Face token found in {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--token_file", default="tokenlist.txt")
    p.add_argument("--output", default="experiments_yeonseok/results/readme_model_paths.json")
    p.add_argument("--model_repo", default=README_MODEL_REPO)
    p.add_argument("--guard_repo", default=README_GUARD_REPO)
    p.add_argument("--with_guard", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = read_token(Path(args.token_file))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[download] model repo: {args.model_repo}")
    model_path = snapshot_download(repo_id=args.model_repo, token=token)
    result = {
        "model_repo": args.model_repo,
        "model_path": model_path,
        "guard_repo": None,
        "guard_path": None,
    }

    if args.with_guard:
        print(f"[download] guard repo: {args.guard_repo}")
        guard_path = snapshot_download(repo_id=args.guard_repo, token=token)
        result["guard_repo"] = args.guard_repo
        result["guard_path"] = guard_path

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[download] wrote paths to {out_path}")


if __name__ == "__main__":
    main()
