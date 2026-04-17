from pydantic import BaseModel, EmailStr
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
    filename: str
    name: Optional[str] = None
    model_type: str
    accuracy: Optional[float] = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Prediction ─────────────────────────────────────────────────────────

class AdditionalFeatures(BaseModel):
    groups: list[str]           # ["emotional", "stylistic", "rhetorical"]
    mask: dict[str, bool]       # 13 keys -> bool


class NBConfig(BaseModel):
    model: Literal["nb"]
    variant: str = "multinomial"
    vectorizer: str = "tfidf"
    ngram_range: str = "1,1"
    additional_features: Optional[AdditionalFeatures] = None


class DeBERTaConfig(BaseModel):
    model: Literal["deberta"]
    integration_mode: str = "concat"
    additional_features: Optional[AdditionalFeatures] = None


class LLMConfig(BaseModel):
    model: Literal["llm"]
    mode: str = "single"
    additional_features: Optional[AdditionalFeatures] = None


ModelConfig = Annotated[
    Union[NBConfig, DeBERTaConfig, LLMConfig],
    pydantic.Discriminator("model"),
]


class EnsembleConfig(BaseModel):
    strategy: str
    weights: Optional[dict[str, float]] = None


class PostMetadata(BaseModel):
    """Optional metadata for social features."""
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


