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
  label: "FAKE" | "REAL" | "UNCERTAIN";
  probability: number | null;
  feature_values?: Record<string, number>;
  reason?: string;
}

export interface EnsembleResult {
  label: "FAKE" | "REAL" | "UNCERTAIN";
  confidence: number;
  strategy: string;
  votes: { FAKE: number; REAL: number; UNCERTAIN?: number };
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
  filename?: string | null;
  name: string;
  model_type: string;
  accuracy?: number;
  is_active: boolean;
  llm_config?: string | null;
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
  source: SourceType | string;
  text: string;
  title?: string;
  url?: string;
  created_at?: string;
  language?: string;

  // Author signals
  author?: string;
  author_handle?: string;
  author_account_age_days?: number | null;
  author_followers_count?: number | null;
  author_following_count?: number | null;
  author_posts_count?: number | null;
  author_is_verified?: boolean | null;
  author_has_custom_domain?: boolean | null;
  author_has_description?: boolean | null;

  // Engagement signals
  likes_count?: number | null;
  reposts_count?: number | null;
  replies_count?: number | null;
  quote_count?: number | null;

  // Platform-specific
  has_url_in_text?: boolean | null;
  has_mentions?: boolean | null;
  is_reply?: boolean | null;
  labels?: string[] | null;

  raw_metadata?: Record<string, unknown> | null;
}

export interface PostClassification {
  label: "FAKE" | "REAL";
  confidence: number;
  probability?: number | null;
}

export interface ClassifiedPost extends NewsItem {
  classification: {
    label: "FAKE" | "REAL" | "UNCERTAIN";
    confidence: number;
    probability: number | null; // null when UNCERTAIN
    reason?: string; // optional explanation from LLM
  } | null;
}

// ── Datasets ──────────────────────────────────────────────────────────────
export interface Dataset {
  id: number;
  user_id: number;
  name: string;
  description: string | null;
  total_news: number;
  fake_count: number;
  real_count: number;
  unlabeled_count: number;
  has_news: boolean;
  has_tweets: boolean;
  has_retweets: boolean;
  has_replies: boolean;
  has_likes: boolean;
  has_users: boolean;
  has_evidence: boolean;
  file_size_bytes: number;
  is_active: boolean;
  created_at: string;
}

export interface DatasetUploadResponse {
  dataset: Dataset;
  preview: {
    news_head: Record<string, any>[];
    counts: Record<string, number>;
    news_columns: string[];
  };
  warnings: string[];
}

export interface DatasetStats {
  dataset_id: number;
  total_news: number;
  fake_count: number;
  real_count: number;
  unlabeled_count: number;
  text_length_stats: {
    min: number;
    median: number;
    mean: number;
    max: number;
    p95: number;
  };
  empty_text_count: number;
  duplicate_text_count: number;
  total_tweets?: number;
  avg_tweets_per_news?: number;
  news_with_tweets_pct?: number;
  total_likes?: number;
  total_retweets?: number;
  total_replies?: number;
  total_users?: number;
  verified_users_pct?: number;
  avg_followers_count?: number;
  top_domains: { domain: string; count: number }[];
}

// ── LLM Presets ───────────────────────────────────────────────────────────
export type LLMMode = "zero_shot" | "few_shot" | "cot" | "bagging";

export interface FewShotExample {
  text: string;
  label: "FAKE" | "REAL";
}

export interface LLMPresetDefaults {
  base_models: string[];
  modes: LLMMode[];
  default_system_prompt: string;
  default_cot_instruction: string;
  default_bagging_n_calls: number;
  default_temperature: number;
  default_max_output_tokens: number;
}

export interface LLMPresetCreate {
  name: string;
  base_model: string;
  mode: LLMMode;
  system_prompt?: string | null;
  temperature: number;
  max_output_tokens: number;
  few_shot_examples?: FewShotExample[] | null;
  cot_instruction?: string | null;
  bagging_n_calls?: number;
}

export interface LLMPresetTestRequest {
  base_model: string;
  mode: LLMMode;
  system_prompt?: string | null;
  temperature: number;
  max_output_tokens: number;
  few_shot_examples?: FewShotExample[] | null;
  cot_instruction?: string | null;
  bagging_n_calls?: number;
  test_text: string;
}

export interface LLMPresetTestResponse {
  label: "FAKE" | "REAL" | "UNCERTAIN";
  confidence: number;
  reason: string;
  base_model_used: string;
  elapsed_seconds: number;
}

// ── Cross-platform verification ──────────────────────────────────────────

export interface VerificationStats {
  total_related: number;
  by_source: Record<string, number>;
  original_domain: string | null;
  domain_mentioned_count: number;
  total_engagement: {
    likes: number;
    reposts: number;
    replies: number;
  };
  verified_authors_pct: number | null;
  custom_domain_authors_pct: number | null;
  avg_account_age_days: number | null;
  min_account_age_days: number | null;
  avg_followers_count: number | null;
  median_followers_count: number | null;
  posts_with_url_pct: number;
  reply_ratio_pct: number;
}

export interface VerificationSignal {
  type: string;
  severity: "info" | "warn" | "alert";
  value?: number;
  note: string;
}

export interface VerifyNewsRequest {
  url?: string;
  text?: string;
  title?: string;
  social_sources?: string[];
  limit_per_source?: number;
}

export interface VerifyNewsResponse {
  original: Record<string, unknown>;
  query_used: string;
  related_posts: NewsItem[];
  stats: VerificationStats;
  signals: VerificationSignal[];
}

// ── User Profile (author, liker, reposter, reply author) ────────────────

export interface UserProfile {
  id: string;
  source: string;
  handle?: string | null;
  display_name?: string | null;

  description?: string | null;
  avatar_url?: string | null;
  created_at?: string | null;
  account_age_days?: number | null;

  followers_count?: number | null;
  following_count?: number | null;
  posts_count?: number | null;

  is_verified?: boolean | null;
  has_custom_domain?: boolean | null;
  is_bot?: boolean | null;
  has_description?: boolean | null;

  followers_following_ratio?: number | null;
}

// ── Reply (post response with full author profile) ──────────────────────

export interface Reply {
  id: string;
  text: string;
  created_at?: string | null;
  author?: UserProfile | null;

  likes_count?: number | null;
  reposts_count?: number | null;
  replies_count?: number | null;

  url?: string | null;
}

// ── Profile group stats (computed aggregates) ───────────────────────────

export interface ProfileGroupStats {
  total: number;
  verified_count?: number;
  verified_pct?: number;
  custom_domain_count?: number;
  custom_domain_pct?: number;
  bots_count?: number;
  with_description_pct?: number;
  avg_account_age_days?: number | null;
  median_account_age_days?: number | null;
  young_accounts_30d_pct?: number;
  young_accounts_90d_pct?: number;
  avg_followers_count?: number | null;
  median_followers_count?: number | null;
  low_followers_pct?: number;
  suspicious_ratio_pct?: number;
}

// ── Post Details (full interaction data) ────────────────────────────────

export interface PostDetailsStats {
  likers?: ProfileGroupStats;
  reposters?: ProfileGroupStats;
  repliers?: ProfileGroupStats;
  quoters?: ProfileGroupStats;
  all_participants?: ProfileGroupStats;
}

export interface PostDetailsFetchedLimits {
  likes_fetched?: number;
  likes_total?: number | null;
  reposts_fetched?: number;
  reposts_total?: number | null;
  replies_fetched?: number;
  replies_total?: number | null;
}

export interface PostDetailsResponse {
  post: NewsItem;
  replies: Reply[];
  reposted_by: UserProfile[];
  liked_by: UserProfile[];
  quoted_by: UserProfile[];
  stats?: PostDetailsStats;
  fetched_limits?: PostDetailsFetchedLimits;
}
