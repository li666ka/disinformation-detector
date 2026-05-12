from pydantic import BaseModel, EmailStr, Field, field_validator
from datetime import datetime
from typing import Optional, Literal, Annotated, Union
import pydantic


# ── Auth ────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Experiments ─────────────────────────────────────────────────────────
class ExperimentResponse(BaseModel):
    id: int
    experiment_id: str
    model_type: str
    model_file: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    training_time: Optional[str] = None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Models ──────────────────────────────────────────────────────────────
class ModelRecordResponse(BaseModel):
    id: int
    experiment_id: Optional[str] = None
    filename: Optional[str] = None   # LLM presets have no file
    name: str                         # required for all records
    model_type: str
    pipeline_type: str = "tweet"      # "tweet" | "article" | "aggregated"
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    # Extra metrics — заповнюються із metrics_json через _enrich_with_metrics_json.
    # Не зберігаються окремими SQL колонками: source of truth — metrics_json.
    f1_macro: Optional[float] = None
    roc_auc: Optional[float] = None
    metrics_json: Optional[str] = None
    is_active: bool
    splits_used: Optional[str] = None  # "in_domain" | "cross_domain" | "mixed" | NULL
    dataset_id: Optional[int] = None
    dataset_name: Optional[str] = None  # резолвиться у serializer (не SQL колонка)
    llm_config: Optional[str] = None  # JSON string for LLM presets
    created_at: datetime

    model_config = {"from_attributes": True}


# ── LLM Presets ─────────────────────────────────────────────────────────
AVAILABLE_BASE_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

AVAILABLE_MODES = ["zero_shot", "few_shot", "cot", "bagging"]

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert media literacy analyst. Your task is to classify text as either "
    "REAL (factual, credible information) or FAKE (misinformation, disinformation, propaganda).\n\n"
    "Analyze the text for: factual claims without evidence, emotional manipulation, "
    "anonymous authority references, conspiracy framing, clickbait patterns, logical fallacies.\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"label": "REAL" or "FAKE", "confidence": 0.0-1.0, "reason": "brief explanation"}'
)

DEFAULT_COT_INSTRUCTION = (
    "Before giving your final answer, reason step by step:\n"
    "1. What are the key claims in this text?\n"
    "2. Are these claims verifiable from reliable sources?\n"
    "3. Is there emotional manipulation or sensationalism?\n"
    "4. Does the language match credible journalism or clickbait?\n"
    "After this reasoning, output ONLY the final JSON with label, confidence and reason."
)


class FewShotExample(BaseModel):
    text: str = Field(..., min_length=5, max_length=2000)
    label: Literal["FAKE", "REAL"]


class LLMPresetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    base_model: str = Field(default="claude-haiku-4-5")
    mode: str = Field(default="zero_shot")
    system_prompt: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=200, ge=50, le=2000)
    few_shot_examples: Optional[list[FewShotExample]] = None
    cot_instruction: Optional[str] = None
    bagging_n_calls: Optional[int] = Field(default=3, ge=2, le=10)
    include_social_context: bool = Field(default=False)

    @field_validator("base_model")
    @classmethod
    def validate_base_model(cls, v):
        if v not in AVAILABLE_BASE_MODELS:
            raise ValueError(f"base_model must be one of {AVAILABLE_BASE_MODELS}")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v):
        if v not in AVAILABLE_MODES:
            raise ValueError(f"mode must be one of {AVAILABLE_MODES}")
        return v


class LLMPresetUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class LLMPresetTestRequest(BaseModel):
    base_model: str
    mode: str
    system_prompt: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 200
    few_shot_examples: Optional[list[FewShotExample]] = None
    cot_instruction: Optional[str] = None
    bagging_n_calls: Optional[int] = 3
    include_social_context: bool = False
    test_text: str = Field(..., min_length=5, max_length=2000)


class LLMPresetTestResponse(BaseModel):
    label: str
    confidence: float
    reason: str
    base_model_used: str
    elapsed_seconds: float


class RandomSamplesRequest(BaseModel):
    n_fake: int = Field(default=2, ge=0, le=10)
    n_real: int = Field(default=2, ge=0, le=10)
    splits_subdir: str = Field(
        default="splits_in_domain",
        description="splits_in_domain | splits_cross_domain | splits_mixed",
    )


class RandomSamplesResponse(BaseModel):
    examples: list[FewShotExample]


# ── Prediction ─────────────────────────────────────────────────────────

class AdditionalFeatures(BaseModel):
    groups: list[str]
    mask: dict[str, bool]


class NBConfig(BaseModel):
    model: Literal["nb"]
    variant: str = "complement"
    vectorizer: str = "tfidf"
    ngram_range: str = "1,1"
    alpha: str = "1.0"
    additional_features: Optional[AdditionalFeatures] = None


class DistilBERTConfig(BaseModel):
    model: Literal["distilbert"]
    integration_mode: str = "concat"
    additional_features: Optional[AdditionalFeatures] = None


class DeBERTaConfig(BaseModel):
    """Legacy alias — accepted for backward-compat with old payloads / records.
    New code should use DistilBERTConfig (model: 'distilbert')."""
    model: Literal["deberta"]
    integration_mode: str = "concat"
    additional_features: Optional[AdditionalFeatures] = None


class LLMConfig(BaseModel):
    model: Literal["llm"]
    preset_id: Optional[int] = None   # reference to ModelRecord.id
    mode: str = "zero_shot"            # fallback for old inline configs
    additional_features: Optional[AdditionalFeatures] = None


class GINConfig(BaseModel):
    """GIN model config. Не приймає additional_features (GNN використовує MiniLM)."""
    model: Literal["gin"]
    hidden_dim: str = "128"
    num_layers: str = "3"
    dropout: str = "0.5"
    learning_rate: str = "0.001"
    epochs: str = "50"
    pooling: Literal["mean", "sum", "max"] = "mean"
    additional_features: Optional[AdditionalFeatures] = None

    @field_validator("additional_features")
    @classmethod
    def warn_if_features_set(cls, v):
        if v is not None:
            import logging
            logging.getLogger(__name__).warning(
                "GIN model received additional_features but ignores them "
                "(node features = MiniLM embeddings). Setting to None."
            )
        return None


class GraphSAGEConfig(BaseModel):
    """GraphSAGE model config. Не приймає additional_features."""
    model: Literal["sage"]
    hidden_dim: str = "128"
    num_layers: str = "2"
    dropout: str = "0.5"
    learning_rate: str = "0.001"
    epochs: str = "50"
    aggregator: Literal["mean", "max", "lstm"] = "mean"
    additional_features: Optional[AdditionalFeatures] = None

    @field_validator("additional_features")
    @classmethod
    def warn_if_features_set(cls, v):
        if v is not None:
            import logging
            logging.getLogger(__name__).warning(
                "SAGE model received additional_features but ignores them. "
                "Setting to None."
            )
        return None


ModelConfig = Annotated[
    Union[NBConfig, DistilBERTConfig, DeBERTaConfig, LLMConfig, GINConfig, GraphSAGEConfig],
    pydantic.Discriminator("model"),
]


class EnsembleConfig(BaseModel):
    strategy: str
    weights: Optional[dict[str, float]] = None


class PostMetadata(BaseModel):
    upvote_ratio: float = 0.5
    score: int = 0
    num_comments: int = 0
    domain: str = ""
    account_age_days: int = 365


class PredictRequest(BaseModel):
    text: str
    mode: str = "single"
    models: list[ModelConfig]
    ensemble: Optional[EnsembleConfig] = None
    metadata: Optional[PostMetadata] = None


# ── Ensembles ───────────────────────────────────────────────────────────
VOTING_TYPES = ["hard", "soft", "weighted"]


class EnsembleMemberRef(BaseModel):
    """Reference до моделі-члена ансамблю."""
    model_id: int
    weight: Optional[float] = None  # тільки для weighted voting


class EnsembleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    voting_type: Literal["hard", "soft", "weighted"]
    member_model_ids: list[int] = Field(..., min_length=2)
    weights: Optional[dict[str, float]] = None  # {"model_id_str": weight}

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v, info):
        voting_type = info.data.get("voting_type")
        member_ids = info.data.get("member_model_ids", [])

        if voting_type == "weighted":
            if not v:
                raise ValueError("Weighted voting requires weights dict")

            for mid in member_ids:
                if str(mid) not in v:
                    raise ValueError(f"Missing weight for model_id={mid}")

            if any(w <= 0 for w in v.values()):
                raise ValueError("All weights must be > 0")
        else:
            if v:
                return None

        return v


class EnsembleResponse(BaseModel):
    id: int
    name: str
    voting_type: str
    member_model_ids: list[int]
    weights: Optional[dict[str, float]] = None

    # Member details (для UI)
    member_models: Optional[list[dict]] = None

    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    f1_macro: Optional[float] = None
    roc_auc: Optional[float] = None

    confusion_matrix: Optional[dict] = None
    splits_used: Optional[str] = None
    dataset_id: Optional[int] = None

    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class EnsembleSummary(BaseModel):
    """Compact view для списку ансамблів."""
    id: int
    name: str
    voting_type: str
    member_count: int
    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None
    splits_used: Optional[str] = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
