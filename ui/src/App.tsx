import React, { useState, useEffect } from "react";
import "./App.css";
import AuthPage from "./AuthPage";
import ProfilePage from "./ProfilePage";
import ModelsPage from "./ModelsPage";
import ExperimentsPage from "./ExperimentsPage";
import ClassificationWizard from "./ClassificationWizard";
import AnalysisPage from "./AnalysisPage";
import SourcesPage from "./SourcesPage";
import DatasetsPage from "./DatasetsPage";
import LLMPresetsPage from "./LLMPresetsPage";
import VerifyPage from "./VerifyPage";
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
  Brain, FileText, Globe, Database, Sparkles, BarChart3,
  Boxes, User, LogOut, Sun, Moon, Target, Crosshair,
  Activity, Award, Loader2, AlertTriangle, CheckCircle2, ShieldCheck,
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
  stylistic: ["caps_ratio", "ttr", "repetition_score", "avg_word_length"],
  rhetorical: ["clickbait_score", "authority_refs", "pronoun_ratio", "question_count"],
  social: ["upvote_ratio", "score_normalized", "num_comments_norm", "domain_credibility", "account_age_norm", "has_url"],
};

// ── Nav items config ─────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { icon: Brain, label: "Навчання моделі", tab: "training" },
  { icon: FileText, label: "Аналіз тексту", tab: "prediction" },
  { icon: Globe, label: "Реальні дані", tab: "sources" },
  { icon: ShieldCheck, label: "Cross-verify", tab: "verify" },
  { icon: Database, label: "Датасети", tab: "datasets" },
  { icon: Sparkles, label: "LLM пресети", tab: "llm-presets" },
  { icon: BarChart3, label: "Експерименти", tab: "experiments" },
  { icon: Boxes, label: "Моделі", tab: "models" },
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
      .catch(() => {})
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

  // ── Training state ─────────────────────────────────────────────────────
  const [wizardConfig, setWizardConfig] = useState<TrainRequest | null>(null);
  const [modelName, setModelName] = useState<string>("");
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [trainingResults, setTrainingResults] = useState<TrainResponse | null>(null);

  // ── Training ───────────────────────────────────────────────────────────
  const [trainingError, setTrainingError] = useState<string | null>(null);

  const generateModelName = () => {
    const cfg = wizardConfig?.models?.[0] as any;
    if (!cfg) return;
    const parts: string[] = [];
    const mtype: string = cfg.model || "nb";
    if (mtype === "nb") {
      const variant = cfg.variant === "complement" ? "CNB" : "MNB";
      const vec = cfg.vectorizer === "count" ? "Count" : "TFIDF";
      const ng = cfg.ngram_range || cfg.ngram || "1,1";
      parts.push(`${variant}-${vec}-ng(${ng})`);
      const mask = cfg.additional_features?.mask || {};
      const emoKeys = ["sentiment_score","emotion_intensity","emoji_count","exclamation_count",
        "anger_score","fear_score","anticipation_score","trust_score","surprise_score",
        "sadness_score","joy_score","disgust_score","positive_score","negative_score"];
      const emoCount = emoKeys.filter((k) => mask[k]).length;
      if (emoCount > 0) parts.push(`+emo(${emoCount})`);
    } else if (mtype === "deberta") {
      parts.push("DistilBERT");
    } else if (mtype === "llm") {
      parts.push("Gemini");
    }
    setModelName(parts.join(" "));
  };

  const startTraining = async () => {
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
      const isLLM = wizardConfig?.models?.length === 1 && wizardConfig.models[0]?.model === "llm";

      try {
        let payload: any = { ...(wizardConfig || {}), model_name: modelName.trim() || undefined };

        if (!isLLM && wizardConfig?.models?.[0]?.model === "nb") {
          const modelParams = wizardConfig.models[0] as any || {};
          const oldMask = (modelParams?.additional_features as any)?.mask || {};

          const cleanedMask: Record<string, boolean> = {};
          FEATURE_GROUP_DEFS.emotional.forEach((key) => {
            cleanedMask[key] = oldMask[key] === true;
          });

          if (payload.models && payload.models[0]) {
            if (!payload.models[0].additional_features) {
              payload.models[0].additional_features = {};
            }
            payload.models[0].additional_features.mask = cleanedMask;
            payload.models[0].additional_features.groups = ["emotional"];
          }
        }

        const { data } = isLLM
          ? await api.post("/evaluate", { max_samples: 50, test_size: 0.2 })
          : await api.post("/train", payload);
        trainData = data;
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
              <button
                key={item.tab}
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
            );
          })}
          <Separator className="my-3" />
          <button
            onClick={() => setActiveTab("profile")}
            className={cn(
              "flex items-center gap-3 w-full rounded-md px-3 py-2 text-sm font-medium transition-colors",
              activeTab === "profile"
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
            )}
          >
            <User className="h-4 w-4 shrink-0" />
            Профіль
          </button>
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
                <CardContent>
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

                  {(() => {
                    const isLLM = wizardConfig?.models?.length === 1 && wizardConfig.models[0]?.model === "llm";
                    return (
                      <Button
                        size="lg"
                        className="w-full bg-green-600 hover:bg-green-700 text-white"
                        onClick={startTraining}
                        disabled={isTraining}
                      >
                        {isTraining && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                        {isTraining ? "Завантаження..." : isLLM ? "Результат" : "Почати навчання"}
                      </Button>
                    );
                  })()}

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
                <Card className="border-green-200 dark:border-green-900">
                  <CardHeader>
                    <CardTitle className="text-lg">Результати тестування</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Metrics Grid */}
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                      {[
                        { icon: Target, label: "Accuracy", value: trainingResults.accuracy },
                        { icon: Crosshair, label: "Precision", value: trainingResults.precision },
                        { icon: Activity, label: "Recall", value: trainingResults.recall },
                        { icon: Award, label: "F1 Score", value: trainingResults.f1_score },
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
                      <h3 className="text-sm font-semibold mb-3">Матриця помилок</h3>
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
              )}
            </div>
          )}

          {activeTab === "prediction" && <AnalysisPage />}
          {activeTab === "sources" && <SourcesPage />}
          {activeTab === "verify" && <VerifyPage />}
          {activeTab === "datasets" && <DatasetsPage />}
          {activeTab === "llm-presets" && <LLMPresetsPage />}
          {activeTab === "experiments" && <ExperimentsPage />}
          {activeTab === "models" && <ModelsPage />}
          {activeTab === "profile" && <ProfilePage user={currentUser} />}
        </div>
      </main>
    </div>
  );
}

export default App;
