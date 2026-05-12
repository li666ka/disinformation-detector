# api/model.py
import os
import math
import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def _resolve_legacy_nb_path(path: str) -> str:
    """Backward-compat: старі шляхи user_X/model_<exp>.pkl були перенесені
    у підпапки user_X/nb_<exp>/model.pkl.  Якщо переданий шлях не існує —
    спробувати знайти у новій структурі.
    """
    if not path:
        return path
    if os.path.exists(path):
        return path

    p = Path(path)
    if p.suffix != ".pkl" or not p.name.startswith("model_"):
        return path

    # model_<exp>.pkl  → nb_<exp>/model.pkl
    # model_nb_<exp>.pkl → nb_<exp>/model.pkl (legacy NB прапор)
    stem_id = p.stem[len("model_"):]
    candidates = []
    if stem_id.startswith("nb_aggregated_"):
        candidates.append(p.parent / stem_id / "model.pkl")
    if stem_id.startswith("nb_"):
        candidates.append(p.parent / stem_id / "model.pkl")
        candidates.append(p.parent / stem_id[3:] / "model.pkl")  # без nb_ префіксу
    candidates.append(p.parent / f"nb_{stem_id}" / "model.pkl")
    candidates.append(p.parent / stem_id / "model.pkl")

    for cand in candidates:
        if cand.exists():
            logger.info(f"Resolved legacy NB path {path} → {cand}")
            return str(cand)
    return path


class FakeNewsModel:
    """
    Модель виявлення дезінформації.
    Завантажує sklearn pipeline (.pkl) або dict-формат (vectorizer+classifier+social).
    Працює як заглушка коли модель не завантажена.
    """

    def __init__(self, model_path: str | None = None):
        self.pipeline = None       # sklearn Pipeline (simple model)
        self._model_dict = None    # dict format: {vectorizer, classifier, has_social}
        resolved = _resolve_legacy_nb_path(model_path) if model_path else None
        self._model_path = resolved
        if resolved and os.path.exists(resolved):
            self.load_from_file(resolved)
        else:
            logger.info("No trained model loaded. Using stub for predictions.")

    def load_from_file(self, path: str) -> None:
        """Завантажити модель з .pkl файлу."""
        path = _resolve_legacy_nb_path(path)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        loaded = joblib.load(path)
        self._model_path = path

        if isinstance(loaded, dict) and "pipeline" in loaded:
            # New bundle format: {pipeline, preprocessing, emotional_features}
            self.pipeline = loaded["pipeline"]
            self._model_dict = None
            logger.info(f"Bundle model loaded from {path} (preprocessing={loaded.get('preprocessing')}, features={loaded.get('emotional_features')})")
        elif isinstance(loaded, dict) and "classifier" in loaded:
            self._model_dict = loaded
            self.pipeline = None
            logger.info(f"Dict-format model loaded from {path}")
        else:
            self.pipeline = loaded
            self._model_dict = None
            logger.info(f"Pipeline model loaded from {path}")

    def predict(self, text: str, feature_values: dict | None = None, use_text: bool = True):
        """
        Класифікація тексту.

        Args:
            text: preprocessed input text
            feature_values: already-computed emotional features dict (key → float), or None
            use_text: if False, skip TF-IDF text column (only emotional features)
        """
        # Dict-format model
        if self._model_dict is not None:
            return self._predict_dict(text, feature_values, use_text)

        # Standard sklearn Pipeline (possibly with ColumnTransformer)
        if self.pipeline is not None:
            return self._predict_pipeline(text, feature_values, use_text)

        # Stub
        return self._predict_stub(text, feature_values)

    def _predict_dict(self, text: str, feature_values: dict | None, use_text: bool = True) -> dict:
        """Predict using dict-format model {vectorizer, classifier}."""
        vectorizer = self._model_dict["vectorizer"]
        clf = self._model_dict["classifier"]

        X = vectorizer.transform([text]) if use_text else None
        if X is None:
            return {"label": "REAL", "score": 0.5, "details": {}, "feature_values": feature_values}
        fake_prob = self._get_probability(clf, X)
        label = "FAKE" if fake_prob > 0.5 else "REAL"
        result = {"label": label, "score": fake_prob, "details": {}}
        if feature_values is not None:
            result["feature_values"] = feature_values
        return result

    def _predict_pipeline(self, text: str, feature_values: dict | None, use_text: bool = True) -> dict:
        """
        Predict using sklearn Pipeline.

        If the pipeline has a ColumnTransformer (trained with emotional features),
        we build a DataFrame with the exact columns the model expects.
        The "text_processed" column is included only if use_text=True.
        Otherwise, we pass text directly.
        """
        import pandas as pd

        preprocessor = self.pipeline.named_steps.get("preprocessor")
        clf = self.pipeline.steps[-1][1]

        if preprocessor is not None and hasattr(preprocessor, "transformers_"):
            # ColumnTransformer pipeline — build DataFrame with required columns
            row = {}
            for _name, _transformer, cols in preprocessor.transformers_:
                if isinstance(cols, str):
                    # text column
                    row[cols] = text if use_text else ""
                elif isinstance(cols, list):
                    # emotional feature columns
                    for col in cols:
                        row[col] = float(feature_values.get(col, 0.0)) if feature_values else 0.0

            X = pd.DataFrame([row])
            X_transformed = preprocessor.transform(X)
            fake_prob = self._get_probability(clf, X_transformed)
        else:
            # Simple text-only pipeline
            fake_prob = self._get_probability(clf, None, pipeline_text=text if use_text else "")

        label = "FAKE" if fake_prob > 0.5 else "REAL"
        result = {"label": label, "score": fake_prob, "details": {}}
        if feature_values is not None:
            result["feature_values"] = feature_values
        return result

    def _predict_stub(self, text: str, feature_values: dict | None) -> dict:
        """
        Called when no model is loaded. Raises HTTPException instead of
        returning random results — safer for production and defense demo.
        """
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=(
                "No model is currently loaded. "
                "Train a model in the 'Training' tab or activate an existing one "
                "in the 'Models' tab before running predictions."
            ),
        )

    def get_top_words(self, n: int = 20) -> dict | None:
        """
        Витягнути топ-N дискримінативних слів для FAKE та REAL класів.
        Працює тільки з Naive Bayes моделями (feature_log_prob_).
        """
        vectorizer = None
        clf = None

        if self._model_dict is not None:
            vectorizer = self._model_dict.get("vectorizer")
            clf = self._model_dict.get("classifier")
        elif self.pipeline is not None:
            # Try to extract from pipeline steps
            for name, step in self.pipeline.steps:
                if hasattr(step, "get_feature_names_out"):
                    vectorizer = step
                elif hasattr(step, "feature_log_prob_"):
                    clf = step

        if vectorizer is None or clf is None:
            return None
        if not hasattr(clf, "feature_log_prob_"):
            return None

        try:
            feature_names = vectorizer.get_feature_names_out()
            log_probs = clf.feature_log_prob_  # shape: (n_classes, n_features)

            if log_probs.shape[0] != 2:
                return None

            # classes_[0] = 0 (REAL), classes_[1] = 1 (FAKE) typically
            classes = list(clf.classes_)
            fake_idx = classes.index(1) if 1 in classes else 1
            real_idx = 1 - fake_idx

            diff = log_probs[fake_idx] - log_probs[real_idx]

            top_fake_indices = diff.argsort()[-n:][::-1]
            top_real_indices = diff.argsort()[:n]

            top_fake = [
                {"word": feature_names[i], "score": round(float(diff[i]), 4)}
                for i in top_fake_indices
            ]
            top_real = [
                {"word": feature_names[i], "score": round(float(-diff[i]), 4)}
                for i in top_real_indices
            ]

            return {"fake": top_fake, "real": top_real}
        except Exception as e:
            logger.warning(f"Failed to extract top words: {e}")
            return None

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
