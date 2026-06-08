"""experiments_hc_7 — causal activation-steering along the cos_to_attack direction.

Turns hc_4_claude's CORRELATIONAL detection finding (cos_to_attack scalar +
threshold separates attack tokens) into a CAUSAL test: does adding/subtracting
that same linear direction to the residual stream during generation change the
attack success rate (ASR), and at what utility cost?

The package CONSUMES hc_4_claude artifacts (fitted cos_to_attack centroids, the
saved train/held-out split, prompts, baseline ASR) so the steering vector is
bit-identical to the validated detector and evaluation stays on the same
held-out prompts. Generation stages (03-06) need the real victim model; vector
construction (00) and analysis (07-08) are model-free.
"""
