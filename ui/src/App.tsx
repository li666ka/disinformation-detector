import React, { useState, useEffect } from "react";
import "./App.css";
import AuthPage from "./AuthPage";
import ModelsPage from "./ModelsPage";
import ClassificationWizard from "./ClassificationWizard";
import AnalysisPage from "./AnalysisPage";
import SourcesPage from "./SourcesPage";
import DatasetsPage from "./DatasetsPage";
import LLMPresetsPage from "./LLMPresetsPage";
import EnsemblesPage from "./EnsemblesPage";
import VerificationPage from "./VerificationPage";
import api from "./api";
import { useTheme } from "./ThemeProvider";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Progress } from "./components/ui/progress";
import { Separator } from "./components/ui/separator";
import { Avatar, AvatarFallback } from "./components/ui/avatar";
import {
  Brain, FileText, Globe, Database, Sparkles,
  Boxes, LogOut, Sun, Moon, Target, Crosshair,
  Activity, Award, Loader2, AlertTriangle, CheckCircle2, ShieldCheck,
  Layers, BarChart3,
} from "lucide-react";
import { toast } from "sonner";
import type {
  User as UserType,
  TrainRequest,
  TrainResponse,
  Dataset,
} from "./types";

// ── Feature group definitions (shared across training and prediction) ────────
const FEATURE_GROUP_DEFS: Record<string, string[]> = {
  emotional: [
    "anger_score", "fear_score", "anticipation_score", "trust_score",
    "surprise_score", "sadness_score", "joy_score", "disgust_score",
    "positive_score", "negative_score", "sentiment_score",
    "emotion_intensity", "emoji_count", "exclamation_count",
  ],
  stylistic: [
    // Form (4) + rhetorical/manipulation (4) — об'єднано в одну групу
    "caps_ratio", "ttr", "repetition_score", "avg_word_length",
    "clickbait_score", "authority_refs", "pronoun_ratio", "question_count",
  ],
  social: [
    // Profile counts (6)
    "followers_count_norm", "friends_count_norm", "ff_ratio",
    "statuses_count_norm", "account_age_norm", "statuses_per_day",
    // Profile flags + strings (6)
    "verified", "has_description", "has_location",
    "description_length_norm", "screen_name_length_norm", "screen_name_digits_ratio",
    // Engagement (5)
    "like_count_norm", "retweet_count_norm", "reply_count_norm",
    "like_to_retweet_ratio", "engagement_rate",
    // Graph cascade (6)
    "cascade_depth_norm", "cascade_breadth_norm", "lifetime_hours_norm",
    "retweets_per_tweet", "replies_per_tweet", "unique_users_norm",
  ],
};

// ── Nav items config ─────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { icon: Database, label: "Датасети", tab: "datasets" },
  { icon: Brain, label: "Навчання моделі", tab: "training" },
  { icon: Boxes, label: "Моделі", tab: "models" },
  { icon: Sparkles, label: "LLM пресети", tab: "llm-presets" },
  { icon: Layers, label: "Ансамблі", tab: "ensembles" },
  { icon: FileText, label: "Аналіз тексту", tab: "prediction" },
  { icon: ShieldCheck, label: "Верифікація", tab: "verification" },
  { icon: Globe, label: "Моніторинг соцмереж", tab: "sources" },
];

function ActiveDatasetBanner() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    api.get<Dataset[]>("/datasets")
      .then((res) => {
        const active = res.data.find((d) => d.is_active) || null;
        setDataset(active);
      })
      .catch(() => { })
      .finally(() => setLoaded(true));
  }, []);

  if (!loaded) return null;

  if (!dataset) {
    return (
      <Card className="border-amber-300 bg-amber-50 dark:bg-amber-950/20 dark:border-amber-900">
        <CardContent className="py-3 flex items-center gap-3">
          <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-400 shrink-0" />
          <div className="flex-1 text-sm">
            <span className="font-medium text-amber-800 dark:text-amber-300">
              Немає активного датасету.
            </span>{" "}
            <span className="text-amber-700 dark:text-amber-400">
              Завантажте датасет та активуйте його у розділі "Датасети".
            </span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="py-3 flex items-center gap-3">
        <Database className="h-4 w-4 text-primary shrink-0" />
        <div className="flex-1 text-sm">
          <span className="text-muted-foreground">Активний датасет:</span>{" "}
          <span className="font-medium">{dataset.name}</span>{" "}
          <span className="text-muted-foreground">
            ({dataset.total_news.toLocaleString()} новин &middot; {dataset.fake_count} FAKE / {dataset.real_count} REAL)
          </span>
        </div>
        <Badge variant="outline" className="text-green-700 border-green-300 bg-green-50 dark:bg-green-950/20 dark:text-green-400">
          <CheckCircle2 className="h-3 w-3 mr-1" /> Активний
        </Badge>
      </CardContent>
    </Card>
  );
}

function App() {
  // Auth state
  const [currentUser, setCurrentUser] = useState<UserType | null>(() => {
    const stored = localStorage.getItem("user");
    return stored ? JSON.parse(stored) : null;
  });
  const isLoggedIn =
    currentUser !== null && localStorage.getItem("token") !== null;

  const handleLogin = (user: UserType) => setCurrentUser(user);
  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setCurrentUser(null);
  };

  const { theme, setTheme } = useTheme();

  const [activeTab, setActiveTab] = useState<string>("training");
  const [prefilledVerifyText, setPrefilledVerifyText] = useState<string>("");

  // ── Training state ─────────────────────────────────────────────────────
  const [wizardConfig, setWizardConfig] = useState<TrainRequest | null>(null);
  const [modelName, setModelName] = useState<string>("");
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [trainingResults, setTrainingResults] = useState<TrainResponse | null>(null);

  // ── Training ───────────────────────────────────────────────────────────
  const [trainingError, setTrainingError] = useState<string | null>(null);

  const generateModelName = () => {
    const models = wizardConfig?.models || [];
    if (models.length === 0) return;

    const GROUP_SHORT: Record<string, string> = {
      emotional: "emo",
      stylistic: "style",
      social: "soc",
    };

    const describeFeatures = (cfg: any): string => {
      const mask = cfg?.additional_features?.mask || {};
      const chunks: string[] = [];
      for (const [group, keys] of Object.entries(FEATURE_GROUP_DEFS)) {
        const enabled = keys.filter((k) => mask[k]).length;
        if (enabled === 0) continue;
        const short = GROUP_SHORT[group] || group;
        chunks.push(enabled === keys.length ? `+${short}` : `+${short}(${enabled})`);
      }
      return chunks.join(" ");
    };

    const describeModel = (cfg: any): string => {
      const mtype: string = cfg.model || "nb";
      const parts: string[] = [];
      if (mtype === "nb") {
        const variant = cfg.variant === "complement" ? "CNB" : "MNB";
        const vec = cfg.vectorizer === "count" ? "Count" : "TFIDF";
        const ng = cfg.ngram_range || cfg.ngram || "1,1";
        parts.push(`${variant}-${vec}-ng(${ng})`);
        const alpha = cfg.alpha;
        if (alpha && alpha !== "1.0" && alpha !== "1") parts.push(`α${alpha}`);
      } else if (mtype === "distilbert") {
        const integ = cfg.integration_mode === "multiview" ? "MV" : "concat";
        parts.push(`DistilBERT-${integ}`);
      } else if (mtype === "gin" || mtype === "sage") {
        const label = mtype === "gin" ? "GIN" : "SAGE";
        const h = cfg.hidden_dim || "128";
        const l = cfg.num_layers || (mtype === "gin" ? "3" : "2");
        parts.push(`${label}-h${h}-L${l}`);
      } else {
        parts.push(mtype);
      }
      const isGNN = mtype === "gin" || mtype === "sage";
      if (!isGNN) {
        const feats = describeFeatures(cfg);
        if (feats) parts.push(feats);
      }
      return parts.join(" ");
    };

    let name: string;
    if (wizardConfig?.mode === "ensemble" && models.length > 1) {
      const shortLabels: Record<string, string> = { nb: "NB", distilbert: "DBERT", gin: "GIN", sage: "SAGE" };
      const ids = models.map((m: any) => shortLabels[m.model] || m.model).join("+");
      const strategy = (wizardConfig as any)?.ensemble?.strategy || "soft";
      name = `Ensemble[${ids}] ${strategy}`;
    } else {
      name = describeModel(models[0]);
    }

    const ts = new Date();
    const stamp = `${String(ts.getMonth() + 1).padStart(2, "0")}${String(ts.getDate()).padStart(2, "0")}-${String(ts.getHours()).padStart(2, "0")}${String(ts.getMinutes()).padStart(2, "0")}`;
    setModelName(`${name} · ${stamp}`);
  };

  const startTraining = async () => {
    const modelType = wizardConfig?.models?.[0]?.model;
    if (modelType === "llm") {
      toast.info("LLM моделі налаштовуються у розділі 'LLM пресети'");
      setActiveTab("llm-presets");
      return;
    }

    const isGraphModel = modelType === "gin" || modelType === "sage";
    if (modelType !== "nb" && modelType !== "distilbert" && !isGraphModel) {
      toast.error("Підтримуються лише NB, DistilBERT, GIN та GraphSAGE");
      return;
    }

    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingResults(null);
    setTrainingError(null);

    try {
      const progressInterval = setInterval(() => {
        setTrainingProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + Math.random() * 15;
        });
      }, 500);

      let trainData: any;

      try {
        if (isGraphModel) {
          // Новий Colab сервер обʼєднав GIN/SAGE під model_type="gnn",
          // а архітектура передається як model_params.architecture.
          const cfg: any = wizardConfig?.models?.[0] || {};
          const architecture = modelType; // "gin" або "sage"
          const graphPayload = {
            model_name: modelName.trim() || undefined,
            model_type: "gnn",
            mode: "single",
            models: wizardConfig?.models || [],
            preprocessing: wizardConfig?.preprocessing || {},
            model_params: {
              architecture,
              hidden_dim: parseInt(cfg.hidden_dim || "128", 10),
              num_layers: parseInt(cfg.num_layers || (architecture === "gin" ? "3" : "2"), 10),
              dropout: parseFloat(cfg.dropout || "0.5"),
              learning_rate: parseFloat(cfg.learning_rate || "0.001"),
              epochs: parseInt(cfg.epochs || "50", 10),
              ...(architecture === "gin" ? { pooling: cfg.pooling || "mean" } : {}),
              ...(architecture === "sage" ? { aggregator: cfg.aggregator || "mean" } : {}),
              additional_features: cfg.additional_features || null,
            },
          };

          const { data } = await api.post("/train", graphPayload);
          trainData = data;
          clearInterval(progressInterval);
          setTrainingProgress(100);
          const flat = data.metrics ? { ...data, ...data.metrics } : data;
          setTrainingResults(flat);
          toast.success(`${architecture.toUpperCase()} навчання завершено!`);
          return;
        }

        if (modelType === "distilbert") {
          // Article-level DistilBERT — посилаємо на /train (Colab /run_training_async),
          // НЕ через /train_aggregated, бо новий Colab сервер для distilbert
          // використовує тільки article_title + article_text.
          const cfg: any = wizardConfig?.models?.[0] || {};
          const distilbertPayload = {
            model_name: modelName.trim() || undefined,
            model_type: "distilbert",
            mode: "single",
            models: wizardConfig?.models || [],
            preprocessing: wizardConfig?.preprocessing || {},
            model_params: {
              integration_mode: cfg.integration_mode || "concat",
              additional_features: cfg.additional_features || null,
              // From UI
              epochs: cfg.epochs ?? 3,
              max_length: cfg.max_length ?? 256,
              freeze_base: cfg.freeze_base ?? true,
              // Hardcoded literature defaults — не через UI
              base_model: "distilbert-base-uncased",
              batch_size: 16,
              learning_rate: 2e-5,
              weight_decay: 0.01,
              warmup_ratio: 0.1,
            },
          };

          const { data } = await api.post("/train", distilbertPayload);
          trainData = data;
          clearInterval(progressInterval);
          setTrainingProgress(100);
          const flat = data.metrics ? { ...data, ...data.metrics } : data;
          setTrainingResults(flat);
          toast.success("DistilBERT навчання завершено!");
          return;
        }

        // NB — article-level pipeline через /train (apples-to-apples з DistilBERT/GNN).
        // Endpoint /train_aggregated більше не використовується для NB.
        const cfg: any = wizardConfig?.models?.[0] || {};
        const useText = cfg.use_text ?? true;
        const mask = cfg.additional_features?.mask || {};
        // "text" — це маркер lexical-групи, не справжня feature. Виключаємо
        // його з hasFeatures, щоб ablation-валідація працювала коректно.
        const hasFeatures = Object.entries(mask).some(
          ([k, v]) => k !== "text" && v === true
        );
        if (!useText && !hasFeatures) {
          clearInterval(progressInterval);
          setIsTraining(false);
          setTrainingProgress(0);
          toast.error(
            "Виберіть хоча б одну групу features або увімкніть 'Лексичні'"
          );
          return;
        }
        const nbPayload = {
          model_name: modelName.trim() || undefined,
          model_type: "nb",
          mode: "single",
          models: wizardConfig?.models || [],
          // preprocessing має сенс тільки коли працюємо з текстом
          preprocessing: useText ? (wizardConfig?.preprocessing || {}) : {},
          model_params: {
            nb_variant: cfg.variant || cfg.nb_variant || "complement",
            // alpha: undefined → Colab робить auto-tuning по validation set.
            ...(cfg.alpha != null && cfg.alpha !== ""
              ? { alpha: parseFloat(cfg.alpha) }
              : {}),
            tfidf_max_features: parseInt(cfg.tfidf_max_features || "50000", 10),
            additional_features: cfg.additional_features || null,
            use_text: useText,
            // vectorizer/ngram передаємо тільки в text-режимі (інакше
            // ML server їх ігнорує — clutter).
            ...(useText
              ? {
                vectorizer_type: cfg.vectorizer || cfg.vectorizer_type || "tfidf",
                ngram_range: cfg.ngram_range || cfg.ngram || "1,2",
              }
              : {}),
          },
        };

        const { data } = await api.post("/train", nbPayload);
        trainData = data;
        clearInterval(progressInterval);
        setTrainingProgress(100);
        const flat = data.metrics ? { ...data, ...data.metrics } : data;
        setTrainingResults(flat);
        toast.success("NB навчання завершено!");
        return;
      } catch (axiosErr: any) {
        clearInterval(progressInterval);
        setTrainingProgress(100);
        const status = axiosErr.response?.status;
        if (status === 503) {
          throw new Error(
            "Colab ML-сервер недоступний. Запустіть ноутбук у Google Colab та вставте COLAB_NGROK_URL у .env",
          );
        }
        const detail = axiosErr.response?.data?.detail;
        throw new Error(
          typeof detail === "string" ? detail : "Помилка навчання",
        );
      }

      clearInterval(progressInterval);
      setTrainingProgress(100);

      const flat = trainData.metrics ? { ...trainData, ...trainData.metrics } : trainData;
      setTrainingResults(flat);
      toast.success("Навчання завершено успішно!");
    } catch (err: any) {
      setTrainingError(err.message || "Помилка під час навчання моделі.");
      toast.error(err.message || "Помилка під час навчання моделі.");
    } finally {
      setIsTraining(false);
    }
  };

  if (!isLoggedIn) {
    return <AuthPage onLogin={handleLogin} />;
  }

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  return (
    <div className="flex h-screen bg-background">
      {/* ── Sidebar ────────────────────────────────────────────────────── */}
      <aside className="w-64 border-r border-border bg-card flex flex-col shrink-0">
        {/* Logo */}
        <div className="p-6 border-b border-border">
          <h1 className="text-lg font-bold text-foreground">Fake News Detector</h1>
          <p className="text-xs text-muted-foreground mt-1">ML Research Platform</p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.tab;
            return (
              <React.Fragment key={item.tab}>
                {item.tab === "prediction" && <Separator className="my-3" />}
                <button
                  onClick={() => setActiveTab(item.tab)}
                  className={cn(
                    "flex items-center gap-3 w-full rounded-md px-3 py-2 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
                  )}
                >
                  <Icon className="h-4 w-4 shrink-0" />
                  {item.label}
                </button>
              </React.Fragment>
            );
          })}
        </nav>

        {/* Footer: user + theme toggle */}
        <div className="p-3 border-t border-border">
          <div className="flex items-center gap-2 mb-2">
            <Avatar className="h-8 w-8">
              <AvatarFallback className="text-xs bg-primary text-primary-foreground">
                {currentUser?.username?.charAt(0).toUpperCase()}
              </AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{currentUser?.username}</p>
              <p className="text-xs text-muted-foreground truncate">{currentUser?.email}</p>
            </div>
            <Button size="icon" variant="ghost" onClick={handleLogout} className="h-8 w-8 shrink-0">
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
          <Button
            size="sm"
            variant="outline"
            className="w-full"
            onClick={toggleTheme}
          >
            {theme === "dark" ? <Sun className="h-4 w-4 mr-2" /> : <Moon className="h-4 w-4 mr-2" />}
            {theme === "dark" ? "Світла тема" : "Темна тема"}
          </Button>
        </div>
      </aside>

      {/* ── Main content area ──────────────────────────────────────────── */}
      <main className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto p-8">
          {activeTab === "training" && (
            <div className="space-y-6">
              {/* Page Header */}
              <div>
                <h2 className="text-2xl font-bold tracking-tight">Навчання моделі</h2>
                <p className="text-muted-foreground">Налаштуйте та навчіть модель класифікації</p>
              </div>

              <ActiveDatasetBanner />

              {/* Section 1: Configuration */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">1. Конфігурація моделі</CardTitle>
                  <CardDescription>Оберіть режим, алгоритм та параметри</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <ClassificationWizard
                    trainingMode
                    onConfigChange={setWizardConfig}
                  />
                </CardContent>
              </Card>

              {/* Section 2: Training */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">2. Навчання моделі</CardTitle>
                  <CardDescription>
                    Дата навчання на активному датасеті користувача
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <label className="text-sm font-medium mb-2 block">Назва моделі</label>
                    <div className="flex gap-2 items-center">
                      <Input
                        value={modelName}
                        onChange={(e) => setModelName(e.target.value)}
                        placeholder="Назва для ідентифікації моделі..."
                        className="max-w-sm"
                      />
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={generateModelName}
                        disabled={!wizardConfig}
                      >
                        Авто-назва
                      </Button>
                    </div>
                  </div>

                  <Button
                    size="lg"
                    className="w-full bg-green-600 hover:bg-green-700 text-white"
                    onClick={startTraining}
                    disabled={isTraining}
                  >
                    {isTraining && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    {isTraining ? "Завантаження..." : "Почати навчання"}
                  </Button>

                  {isTraining && (
                    <div className="flex items-center gap-4">
                      <Progress value={trainingProgress} className="flex-1" />
                      <span className="text-sm font-medium text-muted-foreground min-w-[3rem]">
                        {Math.round(trainingProgress)}%
                      </span>
                    </div>
                  )}

                  {trainingError && (
                    <div className="rounded-lg bg-destructive/10 text-destructive p-3 text-sm">
                      {trainingError}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Section 3: Results */}
              {trainingResults && (
                <div className="space-y-4">
                  <Card className="border-green-200 dark:border-green-900">
                    <CardHeader>
                      <CardTitle className="text-lg">Результати тестування</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Metrics Grid */}
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                        {[
                          { icon: Target, label: "Accuracy", value: trainingResults.accuracy },
                          { icon: Crosshair, label: "Precision (FAKE)", value: trainingResults.precision },
                          { icon: Activity, label: "Recall (FAKE)", value: trainingResults.recall },
                          { icon: Award, label: "F1 (FAKE)", value: trainingResults.f1_score },
                          ...(trainingResults.f1_macro != null
                            ? [{ icon: Award, label: "F1 Macro", value: trainingResults.f1_macro }]
                            : []),
                          ...(trainingResults.roc_auc != null
                            ? [{ icon: BarChart3, label: "ROC AUC", value: trainingResults.roc_auc }]
                            : []),
                        ].map((metric) => (
                          <Card key={metric.label}>
                            <CardContent className="p-4 text-center">
                              <metric.icon className="h-5 w-5 mx-auto mb-2 text-muted-foreground" />
                              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                                {(metric.value * 100).toFixed(1)}%
                              </div>
                              <div className="text-xs text-muted-foreground mt-1">{metric.label}</div>
                            </CardContent>
                          </Card>
                        ))}
                      </div>

                      {/* Confusion Matrix */}
                      <div>
                        <h3 className="text-sm font-semibold mb-1">Матриця помилок</h3>
                        <p className="text-xs text-muted-foreground mb-3">FAKE = позитивний клас</p>
                        <div className="overflow-hidden rounded-lg border">
                          <table className="w-full text-sm">
                            <thead>
                              <tr className="bg-muted/50">
                                <th className="p-3 text-left font-medium"></th>
                                <th className="p-3 text-center font-medium">Передбачено: Правда</th>
                                <th className="p-3 text-center font-medium">Передбачено: Фейк</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr className="border-t">
                                <th className="p-3 text-left font-medium bg-muted/50">Фактично: Правда</th>
                                <td className="p-3 text-center font-semibold text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950/20">
                                  {trainingResults.confusion_matrix?.tn || 0}
                                </td>
                                <td className="p-3 text-center font-semibold text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/20">
                                  {trainingResults.confusion_matrix?.fp || 0}
                                </td>
                              </tr>
                              <tr className="border-t">
                                <th className="p-3 text-left font-medium bg-muted/50">Фактично: Фейк</th>
                                <td className="p-3 text-center font-semibold text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/20">
                                  {trainingResults.confusion_matrix?.fn || 0}
                                </td>
                                <td className="p-3 text-center font-semibold text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950/20">
                                  {trainingResults.confusion_matrix?.tp || 0}
                                </td>
                              </tr>
                            </tbody>
                          </table>
                        </div>
                      </div>

                      {/* Word Cloud */}
                      {trainingResults.top_words && trainingResults.top_words.fake?.length > 0 && (() => {
                        const allWords = [
                          ...(trainingResults.top_words.fake || []).map((w: any) => ({ ...w, type: "fake" as const })),
                          ...(trainingResults.top_words.real || []).map((w: any) => ({ ...w, type: "real" as const })),
                        ];
                        const scores = allWords.map((w) => w.score);
                        const minScore = Math.min(...scores);
                        const maxScore = Math.max(...scores);
                        const range = maxScore - minScore || 1;
                        const fontSize = (score: number) => 13 + ((score - minScore) / range) * 30;
                        const opacity = (score: number) => 0.55 + ((score - minScore) / range) * 0.45;
                        const shuffled = [...allWords].sort(() => Math.random() - 0.5);

                        return (
                          <div>
                            <h3 className="text-sm font-semibold mb-2">Хмара дискримінативних слів</h3>
                            <p className="text-xs text-muted-foreground mb-3">
                              Розмір слова пропорційний його дискримінативній вазі.{" "}
                              <span className="text-red-500">Червоні</span> — маркери фейку,{" "}
                              <span className="text-green-500">зелені</span> — маркери достовірності.
                            </p>
                            <div className="flex flex-wrap gap-x-3 gap-y-1.5 items-center justify-center p-5 rounded-lg bg-muted/50 min-h-[120px]">
                              {shuffled.map((w) => (
                                <span
                                  key={`${w.type}-${w.word}`}
                                  title={`${w.word}: ${w.score.toFixed(3)} (${w.type === "fake" ? "FAKE" : "REAL"})`}
                                  style={{
                                    fontSize: fontSize(w.score),
                                    fontWeight: w.score > (minScore + range * 0.6) ? 700 : 500,
                                    opacity: opacity(w.score),
                                    lineHeight: 1.3,
                                    transition: "transform 0.15s",
                                  }}
                                  className={cn(
                                    "cursor-default hover:scale-110",
                                    w.type === "fake" ? "text-red-500" : "text-green-500"
                                  )}
                                >
                                  {w.word}
                                </span>
                              ))}
                            </div>
                          </div>
                        );
                      })()}

                      {/* Training Info */}
                      <Card>
                        <CardContent className="p-4 space-y-1 text-sm">
                          <p>
                            <span className="font-medium">Модель:</span>{" "}
                            {wizardConfig
                              ? `${wizardConfig.mode === "ensemble" ? "Ансамбль" : wizardConfig.mode === "single" ? "Одна модель" : "Мультівью"}${wizardConfig.models?.length ? ` [${wizardConfig.models.map((m: any) => m.model).join(" + ")}]` : ""}${wizardConfig.ensemble?.strategy ? ` — ${wizardConfig.ensemble.strategy.replace(/_/g, " ")}` : ""}`
                              : "N/A"}
                          </p>
                          <p>
                            <span className="font-medium">Розмір тренувального набору:</span>{" "}
                            {trainingResults.train_size || "N/A"}
                          </p>
                          <p>
                            <span className="font-medium">Розмір тестового набору:</span>{" "}
                            {trainingResults.test_size || "N/A"}
                          </p>
                          <p>
                            <span className="font-medium">Час навчання:</span>{" "}
                            {trainingResults.training_time ? `${trainingResults.training_time} s` : "N/A"}
                          </p>
                        </CardContent>
                      </Card>
                    </CardContent>
                  </Card>

                  {trainingResults.feature_samples && trainingResults.feature_samples.length > 0 && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base flex items-center gap-2">
                          🔍 Приклади обчислених ознак
                        </CardTitle>
                        <CardDescription>
                          Breakdown для 3 FAKE та 3 REAL прикладів з тренувального набору
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        {trainingResults.feature_samples.map((s: any, idx: number) => (
                          <div
                            key={idx}
                            className={cn(
                              "rounded-lg border p-3 space-y-2",
                              s.label === "FAKE"
                                ? "border-red-200 bg-red-50/30 dark:bg-red-950/20"
                                : "border-green-200 bg-green-50/30 dark:bg-green-950/20",
                            )}
                          >
                            <div className="flex items-center gap-2">
                              <Badge
                                variant={s.label === "FAKE" ? "destructive" : "default"}
                                className={s.label === "REAL" ? "bg-green-600" : ""}
                              >
                                {s.label}
                              </Badge>
                              <span className="text-xs text-muted-foreground">
                                source: {s.source}
                              </span>
                            </div>

                            <div className="space-y-1 text-xs">
                              <div>
                                <span className="font-semibold">BEFORE:</span>{" "}
                                <span className="text-muted-foreground line-clamp-2">
                                  {s.text_raw}
                                </span>
                              </div>
                              <div>
                                <span className="font-semibold">AFTER:</span>{" "}
                                <span className="text-muted-foreground line-clamp-2 italic">
                                  {s.text_processed}
                                </span>
                              </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-4 gap-2 pt-2 border-t border-border/50">
                              {s.emotional && Object.keys(s.emotional).length > 0 && (
                                <div>
                                  <p className="text-[10px] font-semibold uppercase tracking-wide text-purple-600 dark:text-purple-400 mb-1">
                                    Емоційні
                                  </p>
                                  <div className="space-y-0.5">
                                    {Object.entries(s.emotional)
                                      .sort((a: any, b: any) => Math.abs(b[1]) - Math.abs(a[1]))
                                      .slice(0, 6)
                                      .map(([k, v]: any) => (
                                        <div key={k} className="flex justify-between text-[11px] font-mono">
                                          <span className="text-muted-foreground">{k.replace("_score", "")}</span>
                                          <span className={v !== 0 ? "font-semibold" : "text-muted-foreground"}>
                                            {Number(v).toFixed(3)}
                                          </span>
                                        </div>
                                      ))}
                                  </div>
                                </div>
                              )}

                              {s.stylistic && Object.keys(s.stylistic).length > 0 && (
                                <div>
                                  <p className="text-[10px] font-semibold uppercase tracking-wide text-blue-600 dark:text-blue-400 mb-1">
                                    Стилістичні
                                  </p>
                                  <div className="space-y-0.5">
                                    {Object.entries(s.stylistic).map(([k, v]: any) => (
                                      <div key={k} className="flex justify-between text-[11px] font-mono">
                                        <span className="text-muted-foreground">{k}</span>
                                        <span className={v !== 0 ? "font-semibold" : "text-muted-foreground"}>
                                          {Number(v).toFixed(3)}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {s.social && Object.keys(s.social).length > 0 && (
                                <div>
                                  <p className="text-[10px] font-semibold uppercase tracking-wide text-teal-600 dark:text-teal-400 mb-1">
                                    Соціальні
                                  </p>
                                  <div className="space-y-0.5">
                                    {Object.entries(s.social).map(([k, v]: any) => (
                                      <div key={k} className="flex justify-between text-[11px] font-mono">
                                        <span className="text-muted-foreground">{k.replace("_count_norm", "").replace("_norm", "")}</span>
                                        <span className={v !== 0 ? "font-semibold" : "text-muted-foreground"}>
                                          {Number(v).toFixed(3)}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === "prediction" && (
            <AnalysisPage
              onDeepCheckRequest={(text) => {
                setPrefilledVerifyText(text);
                setActiveTab("verification");
              }}
            />
          )}
          {activeTab === "verification" && <VerificationPage initialText={prefilledVerifyText} />}
          {activeTab === "sources" && <SourcesPage />}
          {activeTab === "datasets" && <DatasetsPage />}
          {activeTab === "llm-presets" && <LLMPresetsPage />}
          {activeTab === "ensembles" && <EnsemblesPage />}
          {activeTab === "models" && <ModelsPage />}
        </div>
      </main>
    </div>
  );
}

export default App;
