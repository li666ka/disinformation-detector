"""Локальна evaluation LLM presets на FakeNewsNet test split.

Викликається з api/routers/models_router.py для model_type="llm" замість
маршруту на Colab — LLM логіка живе у api/llm_predictor.py через Claude CLI.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from api.llm_predictor import predict_batch_with_preset
from api.llm_social_context import build_social_context

logger = logging.getLogger(__name__)

DATASETS_ROOT = Path("uploaded_datasets")


def _label_to_int(label: str) -> int:
    """REAL → 0, FAKE → 1."""
    return 1 if str(label).upper() == "FAKE" else 0


def _llm_label_to_int(label: str) -> int:
    """Парсить LLM output: 'FAKE' → 1, 'REAL' → 0, інакше -1."""
    if not label:
        return -1
    s = str(label).upper().strip()
    if "FAKE" in s:
        return 1
    if "REAL" in s:
        return 0
    return -1


def evaluate_llm_preset(
    user_id: int,
    dataset_id: int,
    preset_config: dict,
    max_samples: int = 100,
    splits_subdir: str = "splits_cross_domain",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Evaluate LLM preset на test split локального dataset.

    Args:
        user_id: id юзера, шлях uploaded_datasets/user_<id>/dataset_<dataset_id>/
        dataset_id: id датасету
        preset_config: конфіг preset з ModelRecord.llm_config (dict)
        max_samples: скільки статей з test split взяти (default 100)
        splits_subdir: "splits_in_domain" / "splits_cross_domain" / "splits_mixed"
        progress_callback: функція(current, total) для UI прогресу

    Returns:
        dict з accuracy, f1_macro, roc_auc, confusion_matrix, тощо.
    """
    t_start = time.time()

    # ── Завантажити news.csv ──
    ds_path = DATASETS_ROOT / f"user_{user_id}" / f"dataset_{dataset_id}"
    news_path = ds_path / "news.csv"
    if not news_path.exists():
        raise FileNotFoundError(f"news.csv не знайдено: {news_path}")

    logger.info(f"Loading {news_path}")
    news_df = pd.read_csv(news_path, low_memory=False)

    # ── Завантажити test split ──
    test_path = ds_path / splits_subdir / "split_test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"test split не знайдено: {test_path}")

    test_split = pd.read_csv(test_path)
    test_aids = test_split["article_id"].astype(str).tolist()

    # ── Filter news to test articles ──
    news_df["article_id"] = news_df["article_id"].astype(str)
    test_df = news_df[news_df["article_id"].isin(test_aids)].copy()
    test_df["article_text_full"] = (
        test_df["article_title"].fillna("").astype(str)
        + ". "
        + test_df["article_text"].fillna("").astype(str)
    )
    test_df["y_true"] = test_df["article_label"].apply(_label_to_int)

    logger.info(f"Test set: {len(test_df)} articles ({splits_subdir})")

    # ── Sample max_samples (stratified) ──
    if len(test_df) > max_samples:
        from sklearn.model_selection import train_test_split as _tts

        sampled, _ = _tts(
            test_df,
            train_size=max_samples,
            stratify=test_df["y_true"],
            random_state=42,
        )
        test_df = sampled.reset_index(drop=True)
        logger.info(f"Sampled to {len(test_df)} (stratified)")

    include_social = bool(preset_config.get("include_social_context", False))

    n_total_articles = len(test_df)

    # Initial progress signal — UI shows 0/N immediately (instead of 0/0 until first batch)
    if progress_callback:
        progress_callback(0, n_total_articles)

    texts: list[str] = []
    n_with_ctx = 0
    n_without_ctx = 0
    t_prompt_start = time.time()

    for prompt_idx, (_, row) in enumerate(test_df.iterrows(), 1):
        article_text = str(row["article_text_full"])

        if include_social:
            social_ctx = build_social_context(
                user_id=user_id,
                dataset_id=dataset_id,
                article_id=str(row["article_id"]),
                n_tweets=5,
                n_replies_per_tweet=1,
                max_tweet_chars=250,
                max_reply_chars=150,
                include_overview=True,
                include_minority_voices=True,
            )
            if social_ctx:
                n_with_ctx += 1
                texts.append(f"{social_ctx}\n\n[ARTICLE]\n{article_text}")
            else:
                n_without_ctx += 1
                texts.append(f"[NO SOCIAL CONTEXT AVAILABLE]\n\n[ARTICLE]\n{article_text}")
        else:
            texts.append(article_text)

        # Per-record log (every 10) so user can tell prompt building is alive
        if prompt_idx % 10 == 0 or prompt_idx == n_total_articles:
            logger.info(
                f"Prompts built: {prompt_idx}/{n_total_articles} "
                f"(social_ctx={n_with_ctx} found, {n_without_ctx} missing)"
            )

    prompt_elapsed = time.time() - t_prompt_start
    if include_social:
        if n_with_ctx == 0:
            logger.warning(
                "include_social_context=True but 0/%d articles got tweets — "
                "перевірте чи tweets.csv підвантажений у dataset",
                n_total_articles,
            )
        logger.info(
            "Built %d prompts in %.1fs (social_context=on, with_ctx=%d, missing=%d)",
            len(texts), prompt_elapsed, n_with_ctx, n_without_ctx,
        )
    else:
        logger.info("Built %d prompts in %.1fs (social_context=off)", len(texts), prompt_elapsed)

    y_true = test_df["y_true"].values

    # ── Predict batch by batch ──
    batch_size = 10
    n_total = len(texts)
    n_batches = (n_total + batch_size - 1) // batch_size

    y_pred: list[int] = []
    confidences: list[float] = []

    t_predict_start = time.time()

    for i in range(n_batches):
        batch_start = i * batch_size
        batch_end = min(batch_start + batch_size, n_total)
        batch_texts = texts[batch_start:batch_end]

        logger.info(
            f"Batch {i + 1}/{n_batches} ({len(batch_texts)} texts) — "
            f"processed so far: {batch_start}/{n_total}"
        )

        try:
            results = predict_batch_with_preset(batch_texts, preset_config)
        except Exception as e:
            logger.error(f"Batch {i + 1} failed: {e}")
            results = [
                {"label": "ERROR", "confidence": 0.0} for _ in batch_texts
            ]

        for r in results:
            label_int = _llm_label_to_int(r.get("label", ""))
            y_pred.append(label_int)
            # ROC-AUC потребує prob того що FAKE.
            #   predict=FAKE → prob_fake = confidence
            #   predict=REAL → prob_fake = 1 - confidence
            #   error/uncertain → 0.5 (neutral)
            conf = float(r.get("confidence", 0.5))
            if label_int == 1:
                confidences.append(conf)
            elif label_int == 0:
                confidences.append(1.0 - conf)
            else:
                confidences.append(0.5)

        elapsed = time.time() - t_predict_start
        eta = (n_batches - i - 1) * (elapsed / (i + 1))
        logger.info(
            f"Progress: {batch_end}/{n_total} processed "
            f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
        )

        if progress_callback:
            progress_callback(batch_end, n_total)

    y_pred_arr = np.array(y_pred)
    conf_arr = np.array(confidences)

    # ── Filter ERROR predictions з metrics ──
    valid_mask = y_pred_arr >= 0
    n_errors = int((~valid_mask).sum())
    if n_errors > 0:
        logger.warning(f"{n_errors} prediction errors (excluded from metrics)")

    y_pred_clean = y_pred_arr[valid_mask]
    y_true_clean = y_true[valid_mask]
    conf_clean = conf_arr[valid_mask]

    if len(y_true_clean) == 0:
        raise RuntimeError("All LLM predictions failed")

    # ── Compute metrics ──
    metrics: dict = {
        "accuracy": float(accuracy_score(y_true_clean, y_pred_clean)),
        "precision": float(
            precision_score(
                y_true_clean, y_pred_clean, pos_label=1, zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_clean, y_pred_clean, pos_label=1, zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true_clean, y_pred_clean, pos_label=1, zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true_clean, y_pred_clean, average="macro", zero_division=0,
            )
        ),
        "f1_fake": float(
            f1_score(
                y_true_clean, y_pred_clean, pos_label=1, zero_division=0,
            )
        ),
        "f1_real": float(
            f1_score(
                y_true_clean, y_pred_clean, pos_label=0, zero_division=0,
            )
        ),
    }

    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    # ROC-AUC потребує >= 2 класів у y_true
    if len(np.unique(y_true_clean)) >= 2:
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true_clean, conf_clean)
            )
        except Exception as e:
            logger.warning(f"ROC-AUC failed: {e}")
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    metrics["n_evaluated"] = int(len(y_true_clean))
    metrics["n_errors"] = n_errors
    metrics["splits_subdir"] = splits_subdir
    metrics["evaluation_time"] = round(time.time() - t_start, 2)

    logger.info(
        f"LLM evaluation done: acc={metrics['accuracy']:.4f}, "
        f"f1_macro={metrics['f1_macro']:.4f}, "
        f"roc_auc={metrics['roc_auc']}, "
        f"time={metrics['evaluation_time']}s"
    )

    return metrics
