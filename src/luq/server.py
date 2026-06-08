"""FastMCP server exposing LUQ as an MCP tool.

Wraps the validated LUQ core ([luq.core][]) so an agent can score the uncertainty of a
long-form answer against sibling samples over MCP. Exposes a single primary tool,
``luq_score(passage, samples)`` — samples are supplied by the caller (reproducible, matches
the validation dataset). The NLI judge and spaCy pipeline are loaded once at startup and
reused across calls.

Run the server (stdio transport):

    uv run python -m luq.server

Or open it in the MCP Inspector (requires Node):

    uv run mcp dev src/luq/server.py
"""

from __future__ import annotations

import os

import torch
from mcp.server.fastmcp import FastMCP

from luq import core
from luq.nli import NLIModel
from luq.text import load_spacy, split_sentences

DEVICE = os.environ.get("LUQ_DEVICE", "cpu")

mcp = FastMCP("luq")

# Deterministic given an input: model is in eval mode (no dropout) and NLI does no sampling.
# The seed is belt-and-suspenders against any incidental RNG use.
torch.manual_seed(0)

# Heavy resources — load once at import, reuse for every tool call.
# max_length=512 because `samples` are full passages used as NLI premises, not short spans.
_nli = NLIModel(device=DEVICE, max_length=512)
_nlp = load_spacy()


@mcp.tool()
def luq_score(passage: str, samples: list[str]) -> dict:
    """Compute LUQ uncertainty for a long-form answer given sibling samples.

    LUQ measures self-consistency: each sentence of ``passage`` is checked for entailment
    against each sibling ``sample``; low average support means high uncertainty.

    Args:
        passage: The main long-form answer to score.
        samples: Sibling answers sampled from the same model for the same prompt
            (at least one). Each is used whole as a context the passage is checked against.

    Returns:
        A dict with ``uncertainty`` and ``confidence`` (= 1 − uncertainty), both in [0, 1];
        higher ``uncertainty`` means the answer is less self-consistent.

    Raises:
        ValueError: If ``samples`` is empty or ``passage`` yields no sentences.
    """
    if not samples:
        raise ValueError("Provide at least one sibling sample.")
    sentences = split_sentences(passage, _nlp)
    if not sentences:
        raise ValueError("Passage produced no sentences.")

    result = core.luq_score(sentences, samples, _nli)
    return {"uncertainty": result.uncertainty, "confidence": result.confidence}


if __name__ == "__main__":
    mcp.run()
