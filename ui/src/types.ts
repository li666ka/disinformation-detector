// ── Auth ─────────────────────────────────────────────────────────────────────

export interface User {
  id: number;
  username: string;
  email: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

// ── Preprocessing (text cleaning before vectorization) ──────────────────────

export interface PreprocessingConfig {
  removeUrls: boolean;
  removeMentions: boolean;
  cleaning: boolean;
  lowercase: boolean;
  removePunctuation: boolean;
  removeNumbers: boolean;
  removeStopwords: boolean;
  stemming: boolean;
  lemmatization: boolean;
}

// ── Feature keys per group ──────────────────────────────────────────────────

export type SemanticFeatureKey = "text";

export type EmotionalFeatureKey =
  | "sentiment_score"
  | "emotion_intensity"
  | "emoji_count"
  | "exclamation_count"
  | "anger_score"
  | "fear_score"
  | "anticipation_score"
  | "trust_score"
  | "surprise_score"
  | "sadness_score"
  | "joy_score"
  | "disgust_score"
  | "positive_score"
  | "negative_score";

export type StylisticFeatureKey =
  | "caps_ratio"
  | "ttr"
  | "repetition_score"
  | "avg_word_length";

export type RhetoricalFeatureKey =
  | "clickbait_score"
  | "authority_refs"
  | "pronoun_ratio"
  | "question_count";

export type SocialFeatureKey =
  | "upvote_ratio"
  | "score_normalized"
  | "num_comments_norm"
  | "domain_credibility"
  | "account_age_norm"
  | "has_url";

/** All extractable numeric feature keys. */
export type FeatureKey =
  | SemanticFeatureKey
  | EmotionalFeatureKey
  | StylisticFeatureKey
  | RhetoricalFeatureKey
  | SocialFeatureKey;

export type FeatureMask = Record<FeatureKey, boolean>;

// ── Feature groups ──────────────────────────────────────────────────────────

/**
 * "semantic" — TF-IDF text vectorization (on/off toggle, configured via PreprocessingConfig).
 * The rest — extractable numeric features with individual sub-toggles.
 */
export type FeatureGroupId =
  | "semantic"
  | "emotional"
  | "stylistic"
  | "rhetorical"
  | "social";

export interface FeatureDefinition {
  key: FeatureKey;
  label: string;
  type: FeatureGroupId;
}

export interface FeatureGroupDef {
  label: string;
  description: string;
  /** Empty for "semantic" — it has no sub-features, only preprocessing config. */
  features: FeatureDefinition[];
}

export interface AdditionalFeatures {
  groups: FeatureGroupId[];
  mask: Partial<FeatureMask>;
}

// ── Model Configs (sent to backend) ─────────────────────────────────────────

export type ModelType = "nb" | "deberta" | "llm";

interface BaseModelConfig {
  model: ModelType;
  additional_features: AdditionalFeatures | null;
}

export interface NBModelConfig extends BaseModelConfig {
  model: "nb";
  variant: "multinomial" | "complement";
  vectorizer: "tfidf" | "count";
  ngram_range: "1,1" | "1,2" | "1,3";
  alpha: string;
}

export interface DeBERTaModelConfig extends BaseModelConfig {
  model: "deberta";
  integration_mode: "concat" | "multiview";
}

export interface LLMModelConfig extends BaseModelConfig {
  model: "llm";
  mode: "single" | "bagging";
}

export type ModelConfig = NBModelConfig | DeBERTaModelConfig | LLMModelConfig;

// ── Ensemble ────────────────────────────────────────────────────────────────

export type EnsembleStrategyId = "hard" | "soft" | "weighted";

export interface EnsembleConfig {
  strategy: EnsembleStrategyId;
  weights?: Record<ModelType, number>;
}

// ── Requests ────────────────────────────────────────────────────────────────

export interface PredictRequest {
  text: string;
  mode: "single" | "ensemble";
  models: ModelConfig[];
  ensemble?: EnsembleConfig;
  metadata?: PostMetadata;
}

export interface TrainRequest {
  mode: "single" | "ensemble";
  models: ModelConfig[];
  ensemble?: EnsembleConfig;
  preprocessing?: PreprocessingConfig;
}

export interface PostMetadata {
  upvote_ratio: number;
  score: number;
  num_comments: number;
  domain: string;
  account_age_days: number;
}

// ── Responses ───────────────────────────────────────────────────────────────

export interface TopWords {
  fake: Array<{ word: string; score: number }>;
  real: Array<{ word: string; score: number }>;
}

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface IndividualResult {
  model: ModelType;
  label: "FAKE" | "REAL";
  probability: number | null;
  feature_values?: Record<string, number>;
  reason?: string;
}

export interface EnsembleResult {
  label: "FAKE" | "REAL";
  confidence: number;
  strategy: string;
  votes: { FAKE: number; REAL: number };
  excluded: string[];
}

export interface PredictResponse {
  mode: "single" | "ensemble";
  is_fake: boolean;
  confidence: number;
  label: 0 | 1;
  features: Record<string, unknown>;
  feature_values?: Record<string, number>;
  individual_results?: IndividualResult[];
  ensemble_result?: EnsembleResult;
  top_words?: TopWords;
}

export interface TrainResponse {
  // flat metrics (merged from response.metrics by App.tsx)
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix?: ConfusionMatrix;
  train_size: number;
  test_size: number;
  training_time: number | string;
  top_words?: TopWords;
  // envelope fields from API
  status?: string;
  message?: string;
  path?: string;
  download_url?: string;
}

// ── DB Records ──────────────────────────────────────────────────────────────

export interface Experiment {
  id: number;
  experiment_id: string;
  model_type: string;
  model_file?: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  train_size?: number;
  test_size?: number;
  training_time?: string;
  status: "success" | "failed";
  created_at: string;
}

export interface EvalSample {
  index: number;
  text: string;
  true_label: 0 | 1;
  pred_label: 0 | 1;
  pred_str: "REAL" | "FAKE" | "UNCERTAIN";
  true_str: "REAL" | "FAKE";
  confidence: number;
  reason: string;
  correct: boolean;
}

export interface EvalMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: ConfusionMatrix;
  samples_evaluated: number;
}

export interface EvalResponse {
  metrics: EvalMetrics;
  samples: EvalSample[];
}

export interface ModelRecord {
  id: number;
  experiment_id?: string;
  filename: string;
  name?: string;
  model_type: string;
  accuracy?: number;
  is_active: boolean;
  created_at: string;
}

// ── Wizard internal state ───────────────────────────────────────────────────

export interface NBParams {
  variant: NBModelConfig["variant"];
  vectorizer: NBModelConfig["vectorizer"];
  ngram: NBModelConfig["ngram_range"];
  alpha: string;
  additional_groups: FeatureGroupId[];
  feature_mask: Partial<FeatureMask>;
  preprocessing: PreprocessingConfig;
}

export interface DeBERTaParams {
  integration_mode: DeBERTaModelConfig["integration_mode"];
  additional_groups: FeatureGroupId[];
  feature_mask: Partial<FeatureMask>;
}

export interface LLMParams {
  mode: LLMModelConfig["mode"];
  lang: "uk" | "ru" | "auto";
  additional_groups: FeatureGroupId[];
  feature_mask: Partial<FeatureMask>;
}

export type ModelParams = NBParams | DeBERTaParams | LLMParams;

// ── Real-data sources (Bluesky / Mastodon / RSS) ────────────────────────────

export type SourceType = "bluesky" | "mastodon" | "rss";

export interface NewsItem {
  id: string;
  source: SourceType;
  text: string;
  title?: string;
  author?: string;
  author_handle?: string;
  created_at?: string;
  url?: string;
  likes_count?: number | null;
  reposts_count?: number | null;
  replies_count?: number | null;
}

export interface PostClassification {
  label: "FAKE" | "REAL";
  confidence: number;
  probability?: number | null;
}

export interface ClassifiedPost extends NewsItem {
  classification: PostClassification | null;
}
