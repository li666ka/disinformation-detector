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
  // Stylistic (form)
  | "caps_ratio"
  | "ttr"
  | "repetition_score"
  | "avg_word_length"
  // Rhetorical (manipulation patterns) — раніше окрема група
  | "clickbait_score"
  | "authority_refs"
  | "pronoun_ratio"
  | "question_count";

export type SocialFeatureKey =
  // Profile counts
  | "followers_count_norm"
  | "friends_count_norm"
  | "ff_ratio"
  | "statuses_count_norm"
  | "account_age_norm"
  | "statuses_per_day"
  // Profile flags + strings
  | "verified"
  | "has_description"
  | "has_location"
  | "description_length_norm"
  | "screen_name_length_norm"
  | "screen_name_digits_ratio"
  // Engagement
  | "like_count_norm"
  | "retweet_count_norm"
  | "reply_count_norm"
  | "like_to_retweet_ratio"
  | "engagement_rate"
  // Graph cascade (per-article)
  | "cascade_depth_norm"
  | "cascade_breadth_norm"
  | "lifetime_hours_norm"
  | "retweets_per_tweet"
  | "replies_per_tweet"
  | "unique_users_norm";

/** All extractable numeric feature keys. */
export type FeatureKey =
  | SemanticFeatureKey
  | EmotionalFeatureKey
  | StylisticFeatureKey
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
  | "social";

export interface FeatureDefinition {
  key: FeatureKey;
  label: string;
  type: FeatureGroupId;
  /** True для cascade/graph features у social group — для UI badge. */
  isGraph?: boolean;
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

export type ModelType = "nb" | "distilbert" | "llm" | "gin" | "sage";

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
  /** True (default) — pipeline містить TF-IDF tokens.
   *  False — тренування тільки на additional_features (для ablation). */
  use_text?: boolean;
}

export interface DistilBERTModelConfig extends BaseModelConfig {
  model: "distilbert";
  integration_mode: "concat" | "multiview";
  epochs?: number;
  max_length?: number;
  freeze_base?: boolean;
}

export interface LLMModelConfig extends BaseModelConfig {
  model: "llm";
  mode: "single" | "bagging";
}

/**
 * GNN models не використовують handcrafted features.
 * Node features = MiniLM sentence embeddings; граф (article→tweets→retweets/replies)
 * задає індуктивний bias.
 */
export interface GINModelConfig {
  model: "gin";
  hidden_dim: string;
  num_layers: string;
  dropout: string;
  learning_rate: string;
  epochs: string;
  pooling: "mean" | "sum" | "max";
  /** Завжди null для GNN — не використовуємо handcrafted features. */
  additional_features: null;
}

export interface SAGEModelConfig {
  model: "sage";
  hidden_dim: string;
  num_layers: string;
  dropout: string;
  learning_rate: string;
  epochs: string;
  aggregator: "mean" | "max" | "lstm";
  /** Завжди null для GNN — не використовуємо handcrafted features. */
  additional_features: null;
}

export type ModelConfig =
  | NBModelConfig
  | DistilBERTModelConfig
  | LLMModelConfig
  | GINModelConfig
  | SAGEModelConfig;

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
}

export interface TrainRequest {
  mode: "single" | "ensemble";
  models: ModelConfig[];
  ensemble?: EnsembleConfig;
  preprocessing?: PreprocessingConfig;
}

// ── Responses ───────────────────────────────────────────────────────────────

export interface TopWords {
  fake: Array<{ word: string; score: number }>;
  real: Array<{ word: string; score: number }>;
}

/** Confusion matrix для бінарної класифікації, де FAKE = positive class (1). */
export interface ConfusionMatrix {
  /** True Negative — REAL правильно класифіковано як REAL. */
  tn: number;
  /** False Positive — REAL помилково класифіковано як FAKE. */
  fp: number;
  /** False Negative — FAKE помилково класифіковано як REAL. */
  fn: number;
  /** True Positive — FAKE правильно класифіковано як FAKE. */
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
  /** Загальна точність для обох класів. */
  accuracy: number;
  /** Precision для FAKE класу (pos_label=1). */
  precision: number;
  /** Recall для FAKE класу (pos_label=1). */
  recall: number;
  /** F1 для FAKE класу (pos_label=1). */
  f1_score: number;
  /** Незважене середнє F1 для обох класів (FAKE та REAL). */
  f1_macro?: number;
  /** ROC-AUC: probabilistic метрика; FAKE = positive class. */
  roc_auc?: number;
  confusion_matrix?: ConfusionMatrix;
  train_size: number;
  test_size: number;
  training_time: number | string;
  top_words?: TopWords;
  feature_samples?: FeatureSample[];
  // envelope fields from API
  status?: string;
  message?: string;
  path?: string;
  download_url?: string;
}

export interface FeatureSample {
  label: "FAKE" | "REAL";
  source: string;
  text_raw: string;
  text_processed: string;
  emotional: Record<string, number>;
  stylistic: Record<string, number>;
  social?: Record<string, number>;
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

export type PipelineType = "tweet" | "article" | "aggregated" | "graph";

export interface ModelRecord {
  id: number;
  experiment_id?: string;
  filename?: string | null;
  name: string;
  model_type: string;
  pipeline_type?: PipelineType;
  /** Загальна точність для обох класів. */
  accuracy?: number;
  /** Precision для FAKE класу (pos_label=1). */
  precision?: number;
  /** Recall для FAKE класу (pos_label=1). */
  recall?: number;
  /** F1 для FAKE класу (pos_label=1). */
  f1_score?: number;
  // Additional metrics (optional — present in metrics_json after evaluation)
  /** Незважене середнє F1 для обох класів. */
  f1_macro?: number;
  /** ROC-AUC: probabilistic метрика; FAKE = positive class. */
  roc_auc?: number;
  best_epoch?: number;
  metrics_json?: string | null;
  is_active: boolean;
  splits_used?: string | null;
  dataset_id?: number | null;
  dataset_name?: string | null;
  llm_config?: string | null;
  created_at: string;
}

// ── Wizard internal state ───────────────────────────────────────────────────

export interface NBParams {
  variant: NBModelConfig["variant"];
  vectorizer: NBModelConfig["vectorizer"];
  ngram: NBModelConfig["ngram_range"];
  alpha: string;
  use_text?: boolean;
  additional_groups: FeatureGroupId[];
  feature_mask: Partial<FeatureMask>;
  preprocessing: PreprocessingConfig;
}

export interface DistilBERTParams {
  integration_mode: DistilBERTModelConfig["integration_mode"];
  epochs?: number;
  max_length?: number;
  freeze_base?: boolean;
  additional_groups: FeatureGroupId[];
  feature_mask: Partial<FeatureMask>;
}

export interface LLMParams {
  mode: LLMModelConfig["mode"];
  lang: "uk" | "ru" | "auto";
  additional_groups: FeatureGroupId[];
  feature_mask: Partial<FeatureMask>;
}

export type ModelParams = NBParams | DistilBERTParams | LLMParams;

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

  // Доданий роутером /sources/search після token-overlap фільтрації
  // (api/sources/_relevance.py). ∈ [0,1]; недоступний у /sources/recent.
  _relevance_score?: number;
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
  factCheck?: FactCheckResult;
  factCheckLoading?: boolean;
  // LLM claim extraction — окремий етап перед класифікацією
  extraction?: PostExtraction;
}

// ── Extracted claim (LLM розпакування поста) ─────────────────────────────

export interface ExtractedClaimItem {
  claim: string;
  stance: ClaimStance;
  author_verdict: "REAL" | "FAKE" | "MIXED";
}

export interface PostExtraction {
  status: "idle" | "loading" | "done" | "error";
  claims?: ExtractedClaimItem[];
  method?: "llm" | "fallback";
  error?: string;
}

// ── Fact Check (Google Fact Check Tools API) ──────────────────────────────

export type ClaimStance = "supports" | "refutes" | "neutral";

export interface ClaimResult {
  claim: string;
  stance: ClaimStance;
  author_verdict_initial: "REAL" | "FAKE" | "UNKNOWN";
  found: boolean;
  verdict?: string | null;
  verdict_normalized: "FAKE" | "REAL" | "MIXED" | "UNKNOWN";
  effective_author_verdict: "FAKE" | "REAL" | "MIXED" | "UNKNOWN";
  publisher?: string | null;
  url?: string | null;
  review_date?: string | null;
  review_title?: string | null;
  claim_text_matched?: string | null;
  match_similarity?: number | null;
}

export interface FactCheckResult {
  fact_check_found: boolean;
  extraction_method?: "llm" | "fallback";

  // Окремі claims
  claims_extracted?: string[];
  claims_results?: ClaimResult[];

  // Aggregate verdict (top found claim або overall)
  verdict?: string | null;
  verdict_normalized: "FAKE" | "REAL" | "MIXED" | "UNKNOWN";
  publisher?: string | null;
  url?: string | null;
  review_date?: string | null;
  review_title?: string | null;
  claim_text_matched?: string | null;
  claim_query_used?: string | null;

  // Stats
  claims_total?: number;
  claims_found?: number;
  fake_count?: number;
  real_count?: number;
  mixed_count?: number;

  // Comparison
  model_label?: "FAKE" | "REAL" | "UNCERTAIN" | null;
  model_confidence?: number | null;
  match?: boolean | null;
  comparison_status: "MATCH" | "MISMATCH" | "NO_DATA" | "MIXED" | "NO_MODEL";
  error?: string | null;
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
  active_split?: string | null;
  created_at: string;
}

export interface SplitInfo {
  name: string;
  folder: string;
  train_count: number;
  val_count: number;
  test_count: number;
}

export interface DatasetSplitsResponse {
  dataset_id: string | number;
  splits: SplitInfo[];
  has_legacy_splits: boolean;
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

  // NEW
  coverage_pct?: number | null;
  coverage_gap_fake_real?: number | null;
  synthetic_articles_count?: number | null;
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
  include_social_context?: boolean;
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
  include_social_context?: boolean;
  test_text: string;
}

/** Parsed shape of ModelRecord.llm_config JSON string. */
export interface LLMPresetConfig {
  base_model: string;
  mode: LLMMode;
  system_prompt?: string;
  temperature?: number;
  max_output_tokens?: number;
  few_shot_examples?: FewShotExample[];
  cot_instruction?: string;
  bagging_n_calls?: number;
  include_social_context?: boolean;
}

export interface LLMPresetTestResponse {
  label: "FAKE" | "REAL" | "UNCERTAIN";
  confidence: number;
  reason: string;
  base_model_used: string;
  elapsed_seconds: number;
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

// ── Claim Verification ─────────────────────────────────────────────────

export interface Claim {
  text: string;
  claim_type: "statement" | "causal" | "statistical" | "quote";
  entities: string[];
  verifiable: boolean;
  original_text?: string | null;
}

export type EvidenceStance = "supports" | "refutes" | "unrelated" | "unknown";

export interface Evidence {
  source_type: "rss" | "bluesky" | "mastodon";
  source_name: string;
  title?: string | null;
  text: string;
  url?: string | null;
  published_at?: string | null;
  author?: string | null;
  author_verified?: boolean | null;
  author_followers_count?: number | null;

  stance?: EvidenceStance | null;
  stance_confidence?: number | null;
  stance_reasoning?: string | null;
  authority_weight?: number | null;
}

export interface EvidenceBundle {
  claim: Claim;
  rss_evidence: Evidence[];
  social_evidence: Evidence[];
  total_found: number;
  retrieval_time_seconds: number;
  query_used?: string | null;
}

export interface EvidenceBreakdown {
  supports: number;
  refutes: number;
  unrelated: number;
  unknown: number;
  weighted_supports: number;
  weighted_refutes: number;
}

export interface Verdict {
  label: "FAKE" | "REAL" | "UNCERTAIN";
  confidence: number;
  reasoning: string;
  breakdown: EvidenceBreakdown;
  evidence_count: number;
  highest_authority_evidence?: Evidence | null;
  model_used?: string | null;
  verdict_time_seconds: number;
}

export interface VerificationResult {
  original_text: string;
  claims: Claim[];
  evidence_bundles: EvidenceBundle[];
  verdicts: Verdict[];
  overall_verdict?: Verdict | null;
  total_time_seconds: number;
}

export interface VerifyRequest {
  text: string;
  max_claims?: number;
  limit_rss?: number;
  limit_social?: number;
  sources?: string[];
  use_llm_verdict?: boolean;
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

// ── Ensembles ──────────────────────────────────────────────────────────

export type VotingType = "hard" | "soft" | "weighted";

export interface EligibleModel {
  id: number;
  name: string;
  model_type: string;          // "nb" | "distilbert" | "gin" | "sage" | "llm"
  splits_used: string | null;
  dataset_id: number | null;
  f1_score: number | null;
  f1_macro: number | null;
  roc_auc: number | null;
  accuracy: number | null;
  has_predictions: boolean;
  predictions_path: string | null;
}

export interface EligibleModelsResponse {
  models: EligibleModel[];
  total: number;
  with_predictions: number;
}

export interface EnsembleMemberInfo {
  id: number;
  name: string;
  model_type: string;
  f1_score: number | null;
  f1_macro: number | null;
  accuracy: number | null;
  weight: number | null;
}

export interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

export interface EnsembleSummary {
  id: number;
  name: string;
  voting_type: VotingType;
  member_count: number;
  accuracy: number | null;
  f1_macro: number | null;
  splits_used: string | null;
  is_active: boolean;
  created_at: string;
}

export interface Ensemble {
  id: number;
  name: string;
  voting_type: VotingType;
  member_model_ids: number[];
  weights: Record<string, number> | null;
  member_models: EnsembleMemberInfo[] | null;

  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
  f1_macro: number | null;
  roc_auc: number | null;
  confusion_matrix: ConfusionMatrix | null;
  alignment_info?: {
    common_test_size: number;
    member_test_sizes: number[];
    max_member_size: number;
  } | null;

  splits_used: string | null;
  dataset_id: number | null;
  is_active: boolean;
  created_at: string;
}

export interface CreateEnsembleRequest {
  name: string;
  voting_type: VotingType;
  member_model_ids: number[];
  weights?: Record<string, number>;
}


// ── Unified Analyzer (POST /analyze/v2) ─────────────────────────────────

export type AnalyzeInputMode = "text" | "url" | "claim_search";

export interface AnalyzeV2Options {
  extract_claim?: boolean;
  classify?: boolean;
  fact_check?: boolean;
  search_sources?: string[];
  search_limit?: number;
  classify_extracted?: boolean;
  // Default true на backend → ExplanationPanel рендериться, якщо
  // class.explanation повертається (NB log-odds, DistilBERT IG).
  explain?: boolean;
}

export interface AnalyzeV2Request {
  input_mode: AnalyzeInputMode;
  input: string;
  model_id?: number | null;
  options?: AnalyzeV2Options;
}

export interface AnalyzeV2ExtractedClaim {
  claim: string;
  stance: ClaimStance;
  author_verdict: "REAL" | "FAKE" | "MIXED";
}

export interface AnalyzeV2Extraction {
  claims: AnalyzeV2ExtractedClaim[];
  method: "llm" | "fallback" | string;
}

export interface AnalyzeV2Classification {
  label: "FAKE" | "REAL" | "UNCERTAIN";
  confidence: number;
  probability?: number | null;
  reason?: string;
  base_model_used?: string;
  mode?: string;
  // Local explanation, attached when client sets `explain=true`.
  // Shape depends on model type — див. Explanation union.
  explanation?: Explanation;
}

export interface AnalyzeV2ModelUsed {
  id: number;
  name: string;
  type: string;
  f1_score?: number | null;
}

export interface AnalyzeV2SimilarPost {
  post: NewsItem;
  extraction?: AnalyzeV2Extraction;
  classification?: AnalyzeV2Classification;
}

export interface AnalyzeV2Aggregated {
  total_posts: number;
  total_claims?: number;
  stance_distribution: {
    supports: number;
    refutes: number;
    neutral: number;
  };
  classification_distribution: {
    FAKE: number;
    REAL: number;
    UNCERTAIN: number;
  };
  majority_verdict: "FAKE" | "REAL" | "UNCERTAIN" | "UNKNOWN";
  majority_confidence: number;
  consensus_strength: number;
  spread_warning?: string | null;
}

export interface PropagationStats {
  n_tweets?: number;
  n_retweets?: number;
  n_replies?: number;
  synthetic_retweets?: number;
  platforms?: string[];
  warnings?: string[];
}

export interface InferenceContext {
  text?: string;
  claim?: string | null;
  related_posts?: NewsItem[] | null;
  aggregates?: Record<string, number> | null;
  // Phase 2: справжні propagation inputs (article+tweets+retweets+replies)
  graph_data?: Record<string, unknown> | null;
  // Узагальнена статистика по graph_data — для UI без розпакування деталей
  propagation_stats?: PropagationStats | null;
  metadata?: {
    build_time_ms?: number;
    sources_used?: string[];
    n_posts_found?: number;
    warnings?: string[];
  };
}

export interface AnalyzeV2Response {
  input_mode: AnalyzeInputMode;
  original_text: string;
  fetched_post?: NewsItem | null;
  extraction?: AnalyzeV2Extraction | null;
  classification?: AnalyzeV2Classification | null;
  classified_text?: string | null;
  model_used?: AnalyzeV2ModelUsed | null;
  fact_check?: FactCheckResult | null;
  similar_posts?: AnalyzeV2SimilarPost[] | null;
  aggregated?: AnalyzeV2Aggregated | null;
  inference_context?: InferenceContext | null;
  timing_ms: Record<string, number>;
  warnings: string[];
}


// ── Local explanation (NB log-odds / IG / GNNExplainer / LLM reasoning) ─

export interface TokenAttribution {
  token: string;
  attribution: number;
  // NB:
  count?: number;
  log_odds_diff?: number;
  // DistilBERT (IG):
  position?: number;
  is_subword?: boolean;
}

export interface FeatureAttribution {
  feature: string;
  raw_value: number;
  log_odds_diff: number;
  attribution: number;
}

export interface NbExplanation {
  method: "log_odds";
  mode?: "A" | "B" | "C";
  method_params?: {
    use_text?: boolean;
    use_features?: boolean;
    n_text_features?: number;
    n_handcrafted_features?: number;
    classifier?: string;
  };
  tokens: TokenAttribution[];
  all_tokens?: TokenAttribution[];
  feature_attributions?: FeatureAttribution[];
  total_log_odds: number;
  prediction: "FAKE" | "REAL";
  n_features_used?: number;
}

export interface IgExplanation {
  method: "integrated_gradients";
  method_params?: { n_steps?: number; baseline?: string };
  tokens: TokenAttribution[];
  all_tokens_in_order?: TokenAttribution[];
  predicted_class: 0 | 1;
  predicted_label: "FAKE" | "REAL";
  confidence: number;
}

export interface GraphImportantNode {
  node_id: number;
  importance: number;
  metadata?: {
    type?: "article" | "tweet" | "retweet" | "reply" | string;
    text?: string;
    author?: string;
    [k: string]: unknown;
  };
}

export interface GraphImportantEdge {
  source: number;
  target: number;
  importance: number;
}

export interface GnnExplanation {
  method: "gnn_explainer";
  method_params?: { epochs?: number; algorithm?: string };
  important_nodes: GraphImportantNode[];
  important_edges: GraphImportantEdge[];
  n_nodes_total: number;
  n_edges_total: number;
  predicted_class: 0 | 1;
  predicted_label: "FAKE" | "REAL";
  confidence: number;
  architecture?: string;
  cached?: boolean;
}

export interface LlmExplanation {
  method: "llm_reasoning" | string;
  reasoning?: string;
  key_indicators?: string[];
  confidence_factors?: Array<{ name: string; value: number }>;
  uncertainty_factors?: string[];
}

export type Explanation =
  | NbExplanation
  | IgExplanation
  | GnnExplanation
  | LlmExplanation;
