"""One-shot NLI smoke check.

Loads :class:`~luq.nli.NLIModel` on CPU, asserts the label positions match what the LUQ
math assumes, and verifies the entailment signal is oriented correctly (entailed pair
scores high, contradictory pair scores low). Run before trusting anything downstream:

    uv run python -m luq.sanity
"""

from __future__ import annotations

from luq.nli import NLIModel

# (premise, hypothesis, expectation) triples spanning both ends of the scale.
CASES: list[tuple[str, str, str]] = [
    ("A man is playing a guitar on stage.", "A person is making music.", "high"),
    ("The cat sat quietly on the mat.", "There is no animal in the room.", "low"),
    ("Paris is the capital of France.", "The capital of France is Paris.", "high"),
    ("She was born in 1990 in Berlin.", "She was born in 1965 in Tokyo.", "low"),
]


def main() -> None:
    """Load the model, assert label positions, and print scored sanity cases."""
    nli = NLIModel(device="cpu")

    # The whole LUQ signal inverts if these are wrong, so check loudly.
    assert nli.entail_idx == 1, f"expected entailment at index 1, got {nli.entail_idx}"
    assert nli.contra_idx == 0, f"expected contradiction at index 0, got {nli.contra_idx}"
    print(f"label2id OK: entailment={nli.entail_idx}, contradiction={nli.contra_idx}")

    pairs = [(premise, hypothesis) for premise, hypothesis, _ in CASES]
    scores = nli.entailment_prob_batch(pairs)

    ok = True
    for (premise, hypothesis, expect), score in zip(CASES, scores):
        passed = (score > 0.5) if expect == "high" else (score < 0.5)
        ok = ok and passed
        flag = "ok" if passed else "FAIL"
        print(f"[{flag}] P(entail)={score:.4f} expect={expect:>4} | {hypothesis}")

    assert ok, "NLI orientation check failed — entailment signal may be inverted"
    print("Sanity check passed.")


if __name__ == "__main__":
    main()
