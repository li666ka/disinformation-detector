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
    accuracy: Optional[float] = None
    is_active: bool
    llm_config: Optional[str] = None  # JSON string for LLM presets
    created_at: datetime

    model_config = {"from_attributes": True}


# ── LLM Presets ─────────────────────────────────────────────────────────
AVAILABLE_BASE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
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
    base_model: str = Field(default="gemini-2.5-flash-lite")
    mode: str = Field(default="zero_shot")
    system_prompt: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=200, ge=50, le=2000)
    few_shot_examples: Optional[list[FewShotExample]] = None
    cot_instruction: Optional[str] = None
    bagging_n_calls: Optional[int] = Field(default=3, ge=2, le=10)

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


class LLMPresetTestRequest(BaseModel):
    base_model: str
    mode: str
    system_prompt: Optional[str] = None
    temperature: float = 0.0
    max_output_tokens: int = 200
    few_shot_examples: Optional[list[FewShotExample]] = None
    cot_instruction: Optional[str] = None
    bagging_n_calls: Optional[int] = 3
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


class RandomSamplesResponse(BaseModel):
    examples: list[FewShotExample]


# ── Prediction ─────────────────────────────────────────────────────────

class AdditionalFeatures(BaseModel):
    groups: list[str]
    mask: dict[str, bool]


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
    preset_id: Optional[int] = None   # reference to ModelRecord.id
    mode: str = "zero_shot"            # fallback for old inline configs
    additional_features: Optional[AdditionalFeatures] = None


ModelConfig = Annotated[
    Union[NBConfig, DeBERTaConfig, LLMConfig],
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
