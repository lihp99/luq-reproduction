"""Loader for the ``potsawee/wiki_bio_gpt3_hallucination`` validation dataset.

Each of the 238 rows is a GPT-3 generated Wikipedia biography with:
- ``gpt3_sentences``: the passage pre-split into sentences (used verbatim — re-splitting
  would drift from the gold label indexing).
- ``gpt3_text_samples``: 20 sibling passages sampled from the same model (LUQ's ``n``
  extra samples; no generation needed).
- ``annotation``: a human factuality label per sentence in
  {accurate, minor_inaccurate, major_inaccurate}.

We aggregate the per-sentence annotations into a single passage-level *non-factuality*
score in [0, 1] (higher = less factual), which LUQ uncertainty is later correlated against.

    uv run python -m luq.data
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import load_dataset

DATASET_NAME = "potsawee/wiki_bio_gpt3_hallucination"
SPLIT = "evaluation"

# Ordinal non-factuality weight per human label (0 = factual, 1 = fully inaccurate).
ANNOTATION_SCORE: dict[str, float] = {
    "accurate": 0.0,
    "minor_inaccurate": 0.5,
    "major_inaccurate": 1.0,
}


@dataclass(frozen=True)
class Passage:
    """One biography passage with its samples and gold factuality.

    Attributes:
        sentences: The passage split into sentences (gold-aligned, from the dataset).
        samples: 20 sibling passages sampled from the same generator.
        factuality: Mean per-sentence non-factuality in [0, 1]; higher = less factual.
        wiki_bio_text: Reference Wikipedia first paragraph (for sanity checks).
        idx: The dataset's ``wiki_bio_test_idx``.
    """

    sentences: list[str]
    samples: list[str]
    factuality: float
    wiki_bio_text: str
    idx: int


def aggregate_factuality(annotations: list[str]) -> float:
    """Average per-sentence annotations into a passage non-factuality score.

    Args:
        annotations: Per-sentence labels, each one of :data:`ANNOTATION_SCORE`'s keys.

    Returns:
        Mean non-factuality in [0, 1]; ``0.0`` for an empty annotation list.

    Raises:
        KeyError: If an annotation value is not a recognized label.
    """
    if not annotations:
        return 0.0
    return sum(ANNOTATION_SCORE[a] for a in annotations) / len(annotations)


def load_passages() -> list[Passage]:
    """Load and normalize all 238 passages from the evaluation split.

    Returns:
        One :class:`Passage` per dataset row, in dataset order.
    """
    rows = load_dataset(DATASET_NAME)[SPLIT]
    return [
        Passage(
            sentences=row["gpt3_sentences"],
            samples=row["gpt3_text_samples"],
            factuality=aggregate_factuality(row["annotation"]),
            wiki_bio_text=row["wiki_bio_text"],
            idx=row["wiki_bio_test_idx"],
        )
        for row in rows
    ]


def main() -> None:
    """Load the dataset and print one passage as a sanity check."""
    passages = load_passages()
    print(f"Loaded {len(passages)} passages.")

    p = passages[0]
    print("--- passage 0 ---")
    print(f"wiki_bio_test_idx : {p.idx}")
    print(f"n sentences       : {len(p.sentences)}")
    print(f"n samples         : {len(p.samples)}")
    print(f"factuality (0..1) : {p.factuality:.3f}")
    print(f"first sentence    : {p.sentences[0]}")


if __name__ == "__main__":
    main()
