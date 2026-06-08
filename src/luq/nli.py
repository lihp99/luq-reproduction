"""NLI judge for LUQ.

Wraps an NLI cross-encoder (default ``cross-encoder/nli-deberta-v3-base``) and exposes
the single quantity LUQ needs: the probability that a hypothesis is *entailed* by a
premise, computed as a 2-way softmax over the {entailment, contradiction} logits only
(the neutral logit is dropped, exactly as in the LUQ paper).

Design notes:
- Loaded via ``AutoModelForSequenceClassification`` (not sentence-transformers'
  ``CrossEncoder``) so we index raw logits explicitly and control batching/dtype.
- FP32 only: DeBERTa-v3's disentangled attention is numerically unstable in fp16/bf16.
- Label positions are read from ``config.label2id`` (never hardcoded) so the math stays
  correct if the checkpoint is swapped.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"

# (premise, hypothesis) pair.
Pair = tuple[str, str]


class NLIModel:
    """Entailment scorer over a {contradiction, entailment, neutral} NLI model.

    Args:
        model_name: HuggingFace model id of a 3-class NLI sequence classifier whose
            ``label2id`` contains ``"entailment"`` and ``"contradiction"``.
        device: Torch device string (e.g. ``"cpu"`` or ``"mps"``). The model is moved
            to this device and set to eval mode on construction.
        max_length: Token truncation length for each (premise, hypothesis) pair.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name, dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )

        # Read label positions rather than hardcoding; fail loudly if absent.
        label2id = self.model.config.label2id
        try:
            self.entail_idx = label2id["entailment"]
            self.contra_idx = label2id["contradiction"]
        except KeyError as exc:  # pragma: no cover - guards against a wrong checkpoint
            raise ValueError(
                f"Model {model_name!r} lacks an entailment/contradiction label; "
                f"got label2id={label2id}"
            ) from exc

    @torch.no_grad()
    def entailment_prob_batch(
        self, pairs: list[Pair], batch_size: int = 16
    ) -> list[float]:
        """Score many (premise, hypothesis) pairs.

        For each pair, returns ``P(entailment | entailment, contradiction)`` — a softmax
        over only the entailment and contradiction logits, dropping neutral.

        Args:
            pairs: List of ``(premise, hypothesis)`` string tuples.
            batch_size: Number of pairs per forward pass.

        Returns:
            One float in ``[0, 1]`` per input pair, in the same order.
        """
        if not pairs:
            return []

        probs: list[float] = []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            premises = [p for p, _ in chunk]
            hypotheses = [h for _, h in chunk]

            enc = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**enc).logits  # [B, num_labels], FP32
            # 2-way softmax over {contradiction, entailment}; column 1 is entailment.
            pair_logits = logits[:, [self.contra_idx, self.entail_idx]]
            entail = F.softmax(pair_logits, dim=-1)[:, 1]
            probs.extend(entail.cpu().tolist())

        return probs

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        """Score a single (premise, hypothesis) pair.

        Args:
            premise: The context text.
            hypothesis: The statement whose support by ``premise`` is scored.

        Returns:
            ``P(entailment | entailment, contradiction)`` in ``[0, 1]``.
        """
        return self.entailment_prob_batch([(premise, hypothesis)])[0]
