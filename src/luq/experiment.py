"""Stage 4 experiment: validate LUQ on wiki_bio_gpt3_hallucination.

Runs the LUQ core over all 238 passages, then correlates LUQ uncertainty against the gold
human non-factuality label with both Pearson and Spearman. A significant *positive*
coefficient (high uncertainty -> high non-factuality) reproduces the LUQ paper's finding.

    uv run python -m luq.experiment        # full 238
    uv run python -m luq.experiment 10     # first 10 (smoke run)

Writes results/luq_results.csv and results/luq_scatter.png.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG without a display
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from luq.core import luq_score
from luq.data import Passage, load_passages
from luq.nli import NLIModel

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
CSV_PATH = RESULTS_DIR / "luq_results.csv"
PLOT_PATH = RESULTS_DIR / "luq_scatter.png"


def run_experiment(limit: int | None = None) -> list[dict]:
    """Score passages and return one result record per passage.

    Args:
        limit: If given, only the first ``limit`` passages are scored (for smoke runs).

    Returns:
        A list of dicts with keys ``idx``, ``n_sentences``, ``uncertainty``, ``factuality``.
    """
    passages: list[Passage] = load_passages()
    if limit is not None:
        passages = passages[:limit]

    nli = NLIModel(device="mps", max_length=512)

    records: list[dict] = []
    for p in tqdm(passages, desc="LUQ"):
        result = luq_score(p.sentences, p.samples, nli)
        records.append(
            {
                "idx": p.idx,
                "n_sentences": len(p.sentences),
                "uncertainty": result.uncertainty,
                "factuality": p.factuality,
            }
        )
    return records


def write_csv(records: list[dict], path: Path = CSV_PATH) -> None:
    """Write result records to a CSV file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "n_sentences", "uncertainty", "factuality"])
        writer.writeheader()
        writer.writerows(records)


def save_scatter(records: list[dict], r: float, rho: float, path: Path = PLOT_PATH) -> None:
    """Save a scatter plot of LUQ uncertainty vs gold non-factuality.

    Args:
        records: Result records (must contain ``uncertainty`` and ``factuality``).
        r: Pearson coefficient, shown in the title.
        rho: Spearman coefficient, shown in the title.
        path: Output PNG path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    uncertainty = [rec["uncertainty"] for rec in records]
    factuality = [rec["factuality"] for rec in records]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(uncertainty, factuality, alpha=0.5, edgecolor="none")
    ax.set_xlabel("LUQ uncertainty")
    ax.set_ylabel("Gold non-factuality")
    ax.set_title(f"LUQ vs non-factuality (n={len(records)})\nPearson r={r:.3f}, Spearman ρ={rho:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Run the experiment, print correlations, and save CSV + scatter plot."""
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    records = run_experiment(limit=limit)

    uncertainty = [rec["uncertainty"] for rec in records]
    factuality = [rec["factuality"] for rec in records]

    r, r_p = pearsonr(uncertainty, factuality)
    rho, rho_p = spearmanr(uncertainty, factuality)

    write_csv(records)
    save_scatter(records, r, rho)

    print(f"\nn = {len(records)} passages")
    print(f"Pearson  r = {r:+.3f}  (p = {r_p:.2e})")
    print(f"Spearman ρ = {rho:+.3f}  (p = {rho_p:.2e})")
    print(f"CSV  -> {CSV_PATH}")
    print(f"Plot -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
