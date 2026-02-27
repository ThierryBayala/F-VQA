"""Task performance metrics for VQA: Exact Match, BLEU-1/4, CIDEr, ROUGE-L."""

import re
from typing import List, Optional, Sequence, Tuple

import numpy as np


def normalize_answer(s: str) -> str:
    """Normalize answer for Exact Match: lowercase, strip, collapse whitespace."""
    if not s or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # Remove punctuation at end for fair comparison
    s = re.sub(r"[^\w\s]$", "", s).strip()
    return s


def exact_match(references: Sequence[str], predictions: Sequence[str]) -> float:
    """
    Exact Match (EM): fraction of predictions that match (normalized) reference.
    references[i] can be a single string or list of acceptable answers.
    """
    assert len(references) == len(predictions)
    hits = 0
    for ref, pred in zip(references, predictions):
        norm_pred = normalize_answer(pred)
        if isinstance(ref, (list, tuple)):
            refs_norm = [normalize_answer(r) for r in ref]
            if norm_pred in refs_norm:
                hits += 1
        else:
            if norm_pred == normalize_answer(ref):
                hits += 1
    return (hits / len(references)) * 100.0 if references else 0.0


def _tokenize_for_bleu(s: str) -> List[str]:
    """Simple tokenization for BLEU: split on whitespace and lowercase."""
    return normalize_answer(s).split()


def _bleu_smooth():
    from nltk.translate.bleu_score import SmoothingFunction
    return SmoothingFunction().method1


def bleu_scores(
    references: Sequence[Sequence[str]], predictions: Sequence[str]
) -> Tuple[float, float]:
    """
    BLEU-1 and BLEU-4 (corpus-level).
    references[i] = list of reference strings for i-th sample.
    predictions[i] = single hypothesis string.
    Returns (bleu1, bleu4) in 0-100 scale.
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu
    except ImportError:
        try:
            import sacrebleu  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "BLEU requires either nltk or sacrebleu. Install: pip install nltk (and run nltk.download('punkt')) or pip install sacrebleu"
            )
        # sacrebleu path
        refs_flat = [[r[0] if r else ""] for r in references]
        preds_flat = list(predictions)
        b1 = sacrebleu.corpus_bleu(preds_flat, refs_flat, smooth_method="exp", smooth_value=0.01)
        b4 = sacrebleu.corpus_bleu(
            preds_flat, refs_flat, smooth_method="exp", smooth_value=0.01
        )
        return b1.score, b4.score

    # nltk path
    refs_tok = [[_tokenize_for_bleu(r) for r in ref_list] for ref_list in references]
    preds_tok = [_tokenize_for_bleu(p) for p in predictions]
    smooth = _bleu_smooth()
    bleu1 = corpus_bleu(
        refs_tok, preds_tok, weights=(1, 0, 0, 0), smoothing_function=smooth
    ) * 100
    bleu4 = corpus_bleu(
        refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth
    ) * 100
    return float(bleu1), float(bleu4)


def rouge_l(references: Sequence[str], predictions: Sequence[str]) -> float:
    """
    ROUGE-L F1 (corpus-level), 0-100 scale.
    references[i] and predictions[i] are strings.
    """
    try:
        from rouge_score import rouge_scorer  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("ROUGE requires rouge-score. Install: pip install rouge-score")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for ref, pred in zip(references, predictions):
        s = scorer.score(ref or "", pred or "")
        scores.append(s["rougeL"].fmeasure)
    return float(np.mean(scores)) * 100.0 if scores else 0.0


def cider_score(
    references: Sequence[Sequence[str]], predictions: Sequence[str], ids: Sequence[str]
) -> float:
    """
    CIDEr (corpus-level). Uses pycocoevalcap if available.
    references[i] = list of reference strings, predictions[i] = hypothesis.
    ids[i] = unique id for sample i (e.g. str(i)).
    Returns score in 0-10 scale (typical CIDEr scale).
    """
    try:
        from pycocoevalcap.cider.cider import Cider  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "CIDEr requires pycocoevalcap. Install: pip install pycocoevalcap"
        )
    gts = {iid: refs for iid, refs in zip(ids, references)}
    res = {iid: [pred] for iid, pred in zip(ids, predictions)}
    scorer = Cider()
    score, _ = scorer.compute_score(gts, res)
    return float(score)


def compute_all_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    refs_for_bleu: Optional[Sequence[Sequence[str]]] = None,
) -> dict:
    """
    Compute EM, BLEU-1, BLEU-4, ROUGE-L, CIDEr.
    references: list of reference answers (one per sample).
    predictions: list of model predictions.
    refs_for_bleu: optional list of list of refs per sample; if None, uses [[r] for r in references].
    """
    refs_for_bleu = refs_for_bleu or [[r] for r in references]
    ids = [str(i) for i in range(len(references))]

    em = exact_match(references, predictions)
    bleu1, bleu4 = bleu_scores(refs_for_bleu, predictions)
    rouge = rouge_l(references, predictions)
    cider = cider_score(
        [[r] for r in references], list(predictions), ids
    )

    return {
        "Exact Match (EM)": em,
        "BLEU-1": bleu1,
        "BLEU-4": bleu4,
        "CIDEr": cider,
        "ROUGE-L": rouge,
    }


def print_metrics_report(metrics: dict, title: str = "Task Performance Metrics Report") -> None:
    """Print a formatted metrics report."""
    print(f"\n{'='*60}")
    print(title)
    print("=" * 60)
    for name, value in metrics.items():
        if "EM" in name or "ROUGE" in name or "BLEU" in name:
            print(f"  {name}: {value:.2f}%")
        else:
            print(f"  {name}: {value:.4f}")
    print("=" * 60)
