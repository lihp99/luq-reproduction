"""100-pair NLI timing benchmark: CPU vs MPS.

DeBERTa-v3 on Apple Silicon is the project's riskiest component, so we measure throughput
on each available device before committing to one for the LUQ core. Run:

    uv run python -m luq.bench

Picks whichever device is faster for later stages; flag if the winner is below the
~2 pairs/sec threshold from the research plan (would justify a hosted-API judge fallback).
"""

from __future__ import annotations

# Must be set before any MPS op so unsupported kernels fall back to CPU instead of erroring.
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time

import torch

from luq.nli import NLIModel, Pair

# Two reusable templates (one entailed, one contradictory) tiled to the requested count.
_TEMPLATES: list[Pair] = [
    ("A man is playing a guitar on stage.", "A person is making music."),
    ("The cat sat quietly on the mat.", "There is no animal in the room."),
]


def make_pairs(n: int = 100) -> list[Pair]:
    """Build ``n`` (premise, hypothesis) pairs by tiling the templates.

    Args:
        n: Number of pairs to generate.

    Returns:
        A list of ``n`` (premise, hypothesis) tuples.
    """
    return [_TEMPLATES[i % len(_TEMPLATES)] for i in range(n)]


def benchmark(
    device: str,
    pairs: list[Pair],
    batch_size: int = 16,
    warmup: int = 8,
) -> tuple[float, float, float]:
    """Time scoring ``pairs`` on ``device`` after a warmup pass.

    The warmup pays one-time costs (lazy init, Metal kernel compilation) so they don't
    pollute the measurement. On MPS we synchronize around the timed region because
    dispatch is asynchronous.

    Args:
        device: Torch device string (``"cpu"`` or ``"mps"``).
        pairs: The pairs to score in the timed region.
        batch_size: Pairs per forward pass.
        warmup: Number of pairs scored (untimed) before measuring.

    Returns:
        ``(seconds, pairs_per_sec, sample_entail_prob)`` for the timed run.
    """
    nli = NLIModel(device=device)
    nli.entailment_prob_batch(pairs[:warmup], batch_size=batch_size)
    if device == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    scores = nli.entailment_prob_batch(pairs, batch_size=batch_size)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed, len(pairs) / elapsed, scores[0]


def main() -> None:
    """Benchmark every available device on 100 pairs and report the faster one."""
    pairs = make_pairs(100)

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    results: dict[str, float] = {}
    for device in devices:
        elapsed, throughput, sample = benchmark(device, pairs)
        results[device] = throughput
        print(
            f"{device:>4}: {elapsed:6.2f}s  {throughput:6.1f} pairs/s  "
            f"(sample P(entail)={sample:.3f})"
        )

    best = max(results, key=results.get)
    print(f"\nFastest device: {best} ({results[best]:.1f} pairs/s)")
    if results[best] < 2.0:
        print(
            "WARNING: < 2 pairs/s after batching — consider a hosted-API NLI judge "
            "(see research plan threshold)."
        )


if __name__ == "__main__":
    main()
