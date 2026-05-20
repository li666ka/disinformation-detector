"""Voting strategies для ансамблів моделей (api-side copy).

Дзеркалює ml_server/ensemble_voting.py — щоб api не залежав від того, чи
ml_server імпортабельний у поточному оточенні (на Colab — так, локально — ні).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def hard_vote(
    predictions: list[np.ndarray],
    tie_breaker: str = "fake",
) -> np.ndarray:
    """Hard voting: вибираємо мітку яку обрала більшість моделей."""
    if not predictions:
        raise ValueError("predictions list is empty")

    stacked = np.stack(predictions, axis=0)
    n_models, n_samples = stacked.shape
    fake_votes = stacked.sum(axis=0)
    threshold = n_models / 2

    if tie_breaker == "fake":
        result = (fake_votes >= threshold).astype(np.int64)
    else:
        result = (fake_votes > threshold).astype(np.int64)

    logger.info(
        f"Hard vote: n_models={n_models}, n_samples={n_samples}, "
        f"tie_breaker={tie_breaker}, fake_rate={result.mean():.3f}"
    )
    return result


def soft_vote(
    probabilities: list[np.ndarray],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft voting: середнє ймовірностей FAKE."""
    if not probabilities:
        raise ValueError("probabilities list is empty")

    stacked = np.stack(probabilities, axis=0)
    n_models, n_samples = stacked.shape
    avg_proba = stacked.mean(axis=0)
    predictions = (avg_proba >= threshold).astype(np.int64)

    logger.info(
        f"Soft vote: n_models={n_models}, n_samples={n_samples}, "
        f"threshold={threshold}, fake_rate={predictions.mean():.3f}"
    )
    return predictions, avg_proba


def weighted_vote(
    probabilities: list[np.ndarray],
    weights: list[float],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted voting: зважена сума ймовірностей."""
    if not probabilities:
        raise ValueError("probabilities list is empty")
    if len(probabilities) != len(weights):
        raise ValueError(
            f"Mismatched lengths: {len(probabilities)} probas vs {len(weights)} weights"
        )

    weights_arr = np.asarray(weights, dtype=np.float32)
    if (weights_arr <= 0).any():
        raise ValueError("All weights must be > 0")

    weights_norm = weights_arr / weights_arr.sum()
    stacked = np.stack(probabilities, axis=0)
    weighted_proba = np.tensordot(weights_norm, stacked, axes=([0], [0]))
    predictions = (weighted_proba >= threshold).astype(np.int64)

    logger.info(
        f"Weighted vote: n_models={len(probabilities)}, "
        f"weights={weights_norm.tolist()}, "
        f"fake_rate={predictions.mean():.3f}"
    )
    return predictions, weighted_proba


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict:
    """Метрики бінарної класифікації. Спершу пробуємо ml_server.utils,
    інакше — локальний sklearn-based fallback."""
    try:
        from ml_server.utils import compute_metrics
        return compute_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba)
    except ImportError:
        pass

    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_auc": None,
    }
    if y_proba is not None and len(np.unique(y_true)) >= 2:
        try:
            proba = np.asarray(y_proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                proba = proba[:, 1]
            result["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:
            result["roc_auc"] = None
    return result


def _align_members_by_article_id(
    members_data: list[dict],
) -> tuple[list[dict], list[str]]:
    """Soft-align members на спільні article_ids (intersection).

    Беремо перетин ВСІХ членів — це дозволяє ансамблювати моделі, тренувалися
    з різними preprocessing (різні min_text_length, require_tweets тощо).
    """
    logger.info("Aligning test sets across members...")

    member_id_sets = [set(m["article_ids"]) for m in members_data]
    member_sizes = [len(s) for s in member_id_sets]
    logger.info(f"Member test sizes: {member_sizes}")

    common_ids = set.intersection(*member_id_sets)
    n_common = len(common_ids)

    if n_common == 0:
        raise ValueError(
            "No common article_ids across members. "
            f"Member sizes: {member_sizes}. "
            "Models tested on completely different data — cannot ensemble."
        )

    max_member_size = max(member_sizes)
    loss_pct = (max_member_size - n_common) / max_member_size * 100

    if n_common < max_member_size:
        logger.warning(
            f"Test sets mismatch detected. "
            f"Common test set: {n_common}/{max_member_size} "
            f"({loss_pct:.1f}% reduction). "
            "Ensemble evaluation на перетині article_ids."
        )
    if loss_pct > 20:
        logger.warning(
            f"HIGH MISMATCH ({loss_pct:.1f}% reduction). "
            "Consider re-training members with unified preprocessing."
        )

    reference_ids = sorted(common_ids)

    aligned: list[dict] = []
    for m in members_data:
        member_map = {aid: idx for idx, aid in enumerate(m["article_ids"])}
        new_indices = np.asarray(
            [member_map[ref_aid] for ref_aid in reference_ids],
            dtype=np.int64,
        )
        aligned.append({
            **m,
            "article_ids": reference_ids,
            "y_true": np.asarray(m["y_true"])[new_indices],
            "y_pred": np.asarray(m["y_pred"])[new_indices],
            "y_proba_fake": np.asarray(m["y_proba_fake"])[new_indices],
        })

    logger.info(f"Aligned to {len(reference_ids)} common samples")
    return aligned, reference_ids


def evaluate_ensemble(
    voting_type: str,
    members_data: list[dict],
    weights: Optional[dict] = None,
    threshold: float = 0.5,
) -> dict:
    """Predictions членів → voting → metrics.

    members_data: list of dicts {article_ids, y_true, y_pred, y_proba_fake,
                                 model_record_id}.
    """
    if not members_data:
        raise ValueError("No members data")

    aligned, reference_ids = _align_members_by_article_id(members_data)
    reference_y_true = np.asarray(aligned[0]["y_true"])

    for i, m in enumerate(aligned[1:], 1):
        if not np.array_equal(np.asarray(m["y_true"]), reference_y_true):
            raise ValueError(
                f"Member {i} (model_id={m.get('model_record_id')}) has different "
                "y_true. Members tested on different ground truth."
            )

    vt = (voting_type or "").lower()
    if vt == "hard":
        preds_list = [np.asarray(m["y_pred"]) for m in aligned]
        y_pred = hard_vote(preds_list)
        y_proba = None
    elif vt == "soft":
        proba_list = [np.asarray(m["y_proba_fake"], dtype=np.float32) for m in aligned]
        y_pred, y_proba = soft_vote(proba_list, threshold=threshold)
    elif vt == "weighted":
        if not weights:
            raise ValueError("Weighted voting requires weights dict")
        proba_list: list[np.ndarray] = []
        weight_values: list[float] = []
        for m in aligned:
            mid_str = str(m["model_record_id"])
            if mid_str not in weights:
                raise ValueError(f"Missing weight for model_id={mid_str}")
            proba_list.append(np.asarray(m["y_proba_fake"], dtype=np.float32))
            weight_values.append(float(weights[mid_str]))
        y_pred, y_proba = weighted_vote(proba_list, weight_values, threshold=threshold)
    else:
        raise ValueError(f"Unknown voting_type: {voting_type}")

    metrics = _compute_metrics(reference_y_true, y_pred, y_proba)

    return {
        "predictions": y_pred,
        "probabilities": y_proba,
        "y_true": reference_y_true,
        "article_ids": reference_ids,
        "metrics": metrics,
    }


def load_predictions(predictions_path) -> dict:
    """Завантажити predictions.json. Без залежності від ml_server."""
    import json
    from pathlib import Path

    p = Path(predictions_path)
    if not p.exists():
        raise FileNotFoundError(f"Predictions not found: {p}")
    with open(p) as f:
        data = json.load(f)

    preds = data.get("predictions", [])
    return {
        "article_ids": [str(x["article_id"]) for x in preds],
        "y_true": np.array([int(x["y_true"]) for x in preds], dtype=np.int64),
        "y_pred": np.array([int(x["y_pred"]) for x in preds], dtype=np.int64),
        "y_proba_fake": np.array(
            [float(x["y_proba_fake"]) for x in preds], dtype=np.float32
        ),
        "metrics": data.get("metrics", {}),
        "splits_used": data.get("splits_used", ""),
        "dataset_id": data.get("dataset_id", ""),
        "model_type": data.get("model_type", ""),
        "test_size": data.get("test_size", len(preds)),
    }
