# app/model.py
import random
import os
import math
import logging

import joblib
import numpy as np

from app.features import compute_features, compute_features_array

logger = logging.getLogger(__name__)


class FakeNewsModel:
    """
    Модель виявлення дезінформації.
    Завантажує sklearn pipeline (.pkl) або dict-формат (vectorizer+classifier+social).
    Працює як заглушка коли модель не завантажена.
    """

    def __init__(self, model_path: str | None = None):
        self.pipeline = None       # sklearn Pipeline (simple model)
        self._model_dict = None    # dict format: {vectorizer, classifier, has_social}
        self._model_path = model_path
        if model_path and os.path.exists(model_path):
            self.load_from_file(model_path)
        else:
            logger.info("No trained model loaded. Using stub for predictions.")

    def load_from_file(self, path: str) -> None:
        """Завантажити модель з .pkl файлу."""
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        loaded = joblib.load(path)
        self._model_path = path

        if isinstance(loaded, dict) and "classifier" in loaded:
            self._model_dict = loaded
            self.pipeline = None
            logger.info(f"Dict-format model loaded from {path} (has_social={loaded.get('has_social', False)})")
        else:
            self.pipeline = loaded
            self._model_dict = None
            logger.info(f"Pipeline model loaded from {path}")

    def predict(self, text: str, additional_features: dict | None = None,
                metadata: dict | None = None):
        """
        Класифікація тексту.

        Args:
            text: input text
            additional_features: {groups: [...], mask: {key: bool}} or None
            metadata: optional metadata dict for social features
        """
        feature_values = None
        if additional_features and additional_features.get("mask"):
            feature_values = compute_features(text, additional_features["mask"], metadata)

        # Dict-format model (vectorizer + classifier, optionally with social features)
        if self._model_dict is not None:
            return self._predict_dict(text, feature_values, metadata)

        # Standard sklearn Pipeline
        if self.pipeline is not None:
            return self._predict_pipeline(text, feature_values)

        # Stub
        return self._predict_stub(text, feature_values)

    def _predict_dict(self, text: str, feature_values: dict | None,
                      metadata: dict | None) -> dict:
        """Predict using dict-format model {vectorizer, classifier, has_social}."""
        vectorizer = self._model_dict["vectorizer"]
        clf = self._model_dict["classifier"]
        has_social = self._model_dict.get("has_social", False)

        X_tfidf = vectorizer.transform([text])

        if has_social and metadata:
            from scipy.sparse import hstack as sparse_hstack
            from app.features import (
                _social_upvote_ratio, _social_score_normalized,
                _social_num_comments_norm, _social_domain_credibility,
                _social_account_age_norm, _social_has_url,
            )
            social = np.array([[
                _social_upvote_ratio(metadata),
                _social_score_normalized(metadata),
                _social_num_comments_norm(metadata),
                _social_domain_credibility(metadata),
                _social_account_age_norm(metadata),
                _social_has_url(text),
            ]])
            X = sparse_hstack([X_tfidf, social])
        else:
            X = X_tfidf

        fake_prob = self._get_probability(clf, X)
        label = "FAKE" if fake_prob > 0.5 else "REAL"

        result = {
            "label": label,
            "score": fake_prob,
            "details": {"length": len(text), "exclamation_count": text.count("!")},
        }
        if feature_values is not None:
            result["feature_values"] = feature_values
        return result

    def _predict_pipeline(self, text: str, feature_values: dict | None) -> dict:
        """Predict using standard sklearn Pipeline."""
        clf = self.pipeline.steps[-1][1]
        fake_prob = self._get_probability(clf, None, pipeline_text=text)
        label = "FAKE" if fake_prob > 0.5 else "REAL"

        result = {
            "label": label,
            "score": fake_prob,
            "details": {"length": len(text), "exclamation_count": text.count("!")},
        }
        if feature_values is not None:
            result["feature_values"] = feature_values
        return result

    def _predict_stub(self, text: str, feature_values: dict | None) -> dict:
        """Random stub when no model is loaded."""
        fake_prob = random.random()
        label = "FAKE" if fake_prob > 0.5 else "REAL"
        result = {
            "label": label,
            "score": fake_prob,
            "details": {"length": len(text), "exclamation_count": text.count("!")},
        }
        if feature_values is not None:
            result["feature_values"] = feature_values
        return result

    def _get_probability(self, clf, X, pipeline_text: str | None = None) -> float:
        """Extract P(FAKE) from classifier, handling various sklearn estimators."""
        if pipeline_text is not None:
            # Use full pipeline
            if hasattr(self.pipeline, "predict_proba"):
                proba = self.pipeline.predict_proba([pipeline_text])[0]
                fake_idx = list(self.pipeline.classes_).index(1) if 1 in self.pipeline.classes_ else 1
                return float(proba[fake_idx])
            elif hasattr(self.pipeline, "decision_function"):
                score = self.pipeline.decision_function([pipeline_text])[0]
                return 1 / (1 + math.exp(-float(score)))
            else:
                pred = self.pipeline.predict([pipeline_text])[0]
                return 1.0 if pred == 1 else 0.0

        # Direct classifier with pre-transformed X
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)[0]
            classes = list(clf.classes_)
            fake_idx = classes.index(1) if 1 in classes else 1
            return float(proba[min(fake_idx, len(proba) - 1)])
        elif hasattr(clf, "decision_function"):
            score = clf.decision_function(X)[0]
            return 1 / (1 + math.exp(-float(score)))
        else:
            pred = clf.predict(X)[0]
            return 1.0 if pred == 1 else 0.0
