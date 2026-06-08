"""LUQ core algorithm.

Computes a passage-level uncertainty by self-consistency against sibling samples
(Zhang et al., LUQ, EMNLP 2024), using the local NLI judge from :mod:`luq.nli`.

For each sentence of the main answer and each sibling sample:
    P(entail) with premise = sample, hypothesis = sentence    (direction matters!)
Per-sentence confidence is the mean P(entail) over samples; passage uncertainty is
``1 - mean(confidence)`` over sentences. This is the simple asymmetric variant (main
answer vs. its samples), as specified in the project's execution steps.

    uv run python -m luq.core
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from luq.nli import NLIModel


@dataclass(frozen=True)
class LUQResult:
    """Result of scoring one passage.

    Attributes:
        uncertainty: ``1 - mean(per_sentence_confidence)`` in [0, 1]; higher = less certain.
        confidence: ``1 - uncertainty`` in [0, 1]; the mean per-sentence support.
        per_sentence_confidence: Mean P(entail) over samples for each main-answer sentence.
    """

    uncertainty: float
    confidence: float
    per_sentence_confidence: list[float]


def luq_score(
    sentences: list[str],
    samples: list[str],
    nli: NLIModel,
    batch_size: int = 32,
) -> LUQResult:
    """Compute LUQ uncertainty for a passage given its sibling samples.

    Args:
        sentences: The main answer split into sentences (the hypotheses).
        samples: Sibling passages sampled from the same generator (the premises). Each is
            used as the context against which every sentence's support is judged.
        nli: The NLI judge providing ``entailment_prob_batch``.
        batch_size: Pairs per NLI forward pass.

    Returns:
        A :class:`LUQResult`. ``uncertainty`` is ``nan`` when ``sentences`` is empty.

    Raises:
        ValueError: If ``samples`` is empty (LUQ is undefined without siblings).
    """
    if not samples:
        raise ValueError("LUQ requires at least one sibling sample.")
    if not sentences:
        return LUQResult(uncertainty=float("nan"), confidence=float("nan"),
                         per_sentence_confidence=[])

    # Newlines in a premise hurt the NLI tokenization; flatten them (matches luq_vllm.py).
    premises = [s.replace("\n", " ") for s in samples]

    # Build all (premise=sample, hypothesis=sentence) pairs, grouped sentence-major so the
    # flat score list reshapes cleanly into [sentence][sample].
    pairs = [(premise, sentence) for sentence in sentences for premise in premises]
    scores = nli.entailment_prob_batch(pairs, batch_size=batch_size)

    n_samples = len(premises)
    per_sentence_confidence = [
        fmean(scores[i * n_samples : (i + 1) * n_samples]) for i in range(len(sentences))
    ]
    confidence = fmean(per_sentence_confidence)
    return LUQResult(
        uncertainty=1.0 - confidence,
        confidence=confidence,
        per_sentence_confidence=per_sentence_confidence,
    )


def main() -> None:
    """Score passage 0 of the dataset and print its uncertainty breakdown."""
    from luq.data import load_passages

    passage = load_passages()[0]
    nli = NLIModel(device="mps", max_length=512)
    result = luq_score(passage.sentences, passage.samples, nli)

    print(f"passage idx       : {passage.idx}")
    print(f"gold factuality   : {passage.factuality:.3f} (higher = less factual)")
    print(f"LUQ uncertainty   : {result.uncertainty:.3f} (higher = less certain)")
    print(f"LUQ confidence    : {result.confidence:.3f}")
    print("per-sentence confidence:")
    for conf, sentence in zip(result.per_sentence_confidence, passage.sentences):
        print(f"  {conf:.3f}  {sentence}")


if __name__ == "__main__":
    main()
