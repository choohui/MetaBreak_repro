from __future__ import annotations


def apply_mimicry(text: str, special_strs: list[str], repl_strs: list[str]) -> str:
    pairs = sorted(zip(special_strs, repl_strs), key=lambda x: len(x[0]), reverse=True)
    out = text
    for src, dst in pairs:
        out = out.replace(src, dst)
    return out


def assistant_header_from_replacement(repl: dict) -> str:
    header = repl.get("assistant_header")
    if header:
        return str(header)
    strs = list(repl.get("target_token_strs", []))
    if len(strs) >= 3:
        return f"{strs[0]}{strs[1]}assistant{strs[2]}\n\n"
    return "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def mimicked_header(repl: dict) -> str:
    return apply_mimicry(
        assistant_header_from_replacement(repl),
        list(repl.get("target_token_strs", [])),
        list(repl.get("best_triple_decoded", [])),
    )


