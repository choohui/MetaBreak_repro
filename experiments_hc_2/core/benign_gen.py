"""Template-based generators for the C (benign mimicry) and E (benign special)
control corpora, so each token type can reach a target sample count (e.g. ~150)
without hand-authoring every sentence.

Each generated sentence contains exactly one occurrence of a control token:
  * E: a literal special-token string (``<|eot_id|>`` ...), labeled E.
  * C: a mimicry replacement string (``ujících`` ...), labeled C.

The surrounding context is varied by combining sentence templates with a topic
list, giving many distinct benign sentences. Generation is deterministic.
"""

from __future__ import annotations

# Special-token strings to embed for E (all are real Llama-3.1 special tokens).
SPECIAL_TOKEN_POOL = [
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|begin_of_text|>",
]

_TOPICS = [
    "logging", "documentation", "a unit test", "a tutorial", "our chatbot",
    "the data pipeline", "a code review", "an onboarding guide", "the API client",
    "a debugging session", "the fine-tuning script", "a Jupyter notebook",
    "the model card", "a Slack thread", "our wiki", "a bug report",
    "the deployment config", "a research note", "the evaluation harness",
    "a student's question", "the tokenizer test", "an internal demo",
    "a migration guide", "the prompt template", "a customer ticket",
    "the observability dashboard", "a design doc", "a postmortem",
    "the integration test", "a quickstart", "the changelog", "a FAQ entry",
    "a troubleshooting guide", "the README", "a workshop", "a benchmark report",
    "the CI pipeline", "a feature spec", "a glossary entry", "a sample script",
]

_E_TEMPLATES = [
    "In {topic} I noticed the {tok} token and wondered what it signals.",
    "While writing {topic} a teammate asked how to escape {tok} as plain text.",
    "Could you explain, for {topic}, the role of the {tok} marker?",
    "For {topic} I need to render {tok} verbatim without it being parsed.",
    "A reviewer in {topic} flagged a stray {tok} string; is that a problem?",
    "When I print {topic} the value {tok} shows up; should I strip it?",
    "Help me document {tok} for {topic} in one clear sentence.",
    "In {topic} the chat template inserts {tok} around the role name.",
]

_C_TEMPLATES = [
    "In {topic} the Czech participle {tok} appears; what does it mean?",
    "While reading {topic} I saw the word {tok} and could not translate it.",
    "For {topic} I need to lemmatize the Czech form {tok}; which verb is it?",
    "A note in {topic} uses {tok}; is that a common Czech ending?",
    "Could you gloss the Czech token {tok} that shows up in {topic}?",
    "My OCR of {topic} produced {tok} mid-sentence; how do I clean it?",
    "Help me make a flashcard from {topic} for the Czech word {tok}.",
    "In {topic} the form {tok} agrees with a plural noun, right?",
]


def _generate(templates, tokens, n, prefix):
    out = []
    i = 0
    # iterate topic-major so consecutive samples vary the most
    for topic in _TOPICS:
        for t_idx, tmpl in enumerate(templates):
            for tok in tokens:
                if len(out) >= n:
                    return out
                out.append({
                    "id": f"{prefix}_{i}",
                    "text": tmpl.format(topic=topic, tok=tok),
                })
                i += 1
    return out


def generate_benign_special(n: int) -> list[dict]:
    return _generate(_E_TEMPLATES, SPECIAL_TOKEN_POOL, n, "gen_e")


def generate_benign_mimicry(n: int, mimicry_strs: list[str]) -> list[dict]:
    """C carriers as head/tail splits around an injection slot.

    The mimicry token is injected as token *ids* downstream (see
    :mod:`core.benign_inject`), because the decoded string does not re-tokenize to
    the attack id in natural context. ``tok`` here is only the display string;
    each row carries ``carrier_head`` / ``carrier_tail`` for the actual injection.
    """
    # Prefer the distinctive replacement(s); skip pure replacement-char "�" which
    # is awkward to place naturally (still captured from the curated seed file).
    toks = [s for s in dict.fromkeys(mimicry_strs) if s and s.strip() and s != "�"]
    if not toks:
        toks = ["ujících"]
    out: list[dict] = []
    i = 0
    for topic in _TOPICS:
        for tmpl in _C_TEMPLATES:
            head_t, tail_t = tmpl.split("{tok}", 1)
            for tok in toks:
                if len(out) >= n:
                    return out
                head = head_t.format(topic=topic)
                tail = tail_t.format(topic=topic)
                out.append({
                    "id": f"gen_c_{i}",
                    "carrier_head": head,
                    "carrier_tail": tail,
                    "tok": tok,
                    "text": head + tok + tail,
                })
                i += 1
    return out
