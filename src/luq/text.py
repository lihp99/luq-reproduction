"""Sentence splitting via spaCy.

The MCP tool receives a passage as a raw string, but the LUQ core operates on a list of
sentences. We split with ``en_core_web_sm`` — the same model the validation dataset used
to produce ``gpt3_sentences`` — so the tool's behavior matches the validated pipeline.

Kept separate from :mod:`luq.core` so the algorithm module stays free of the heavy spaCy
import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import spacy

if TYPE_CHECKING:
    from spacy.language import Language

SPACY_MODEL = "en_core_web_sm"


def load_spacy(model: str = SPACY_MODEL) -> "Language":
    """Load the spaCy pipeline used for sentence segmentation.

    Args:
        model: Name of an installed spaCy model.

    Returns:
        The loaded spaCy ``Language`` pipeline.
    """
    return spacy.load(model)


def split_sentences(text: str, nlp: "Language") -> list[str]:
    """Split ``text`` into non-empty, stripped sentences.

    Args:
        text: The passage to segment.
        nlp: A loaded spaCy pipeline (see :func:`load_spacy`).

    Returns:
        The passage's sentences, in order, with surrounding whitespace removed and any
        empty fragments dropped.
    """
    return [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]
