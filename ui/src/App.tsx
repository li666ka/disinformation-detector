import React, { useEffect, useState } from "react";
import "./App.css";
import AuthPage from "./AuthPage";
import ProfilePage from "./ProfilePage";
import ModelsPage from "./ModelsPage";
import ExperimentsPage from "./ExperimentsPage";
import ClassificationWizard from "./ClassificationWizard";
import AnalysisPage from "./AnalysisPage";
import SourcesPage from "./SourcesPage";
import api from "./api";
import type {
  User,
  TrainRequest,
  TrainResponse,
  PredictResponse,
  ModelRecord,
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

function App() {
  // Auth state
  const [currentUser, setCurrentUser] = useState<User | null>(() => {
    const stored = localStorage.getItem("user");
    return stored ? JSON.parse(stored) : null;
  });
  const isLoggedIn =
    currentUser !== null && localStorage.getItem("token") !== null;

  const handleLogin = (user: User) => setCurrentUser(user);
  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setCurrentUser(null);
  };

  const [activeTab, setActiveTab] = useState<string>("training");

  // ── Training state ─────────────────────────────────────────────────────
  const [wizardConfig, setWizardConfig] = useState<TrainRequest | null>(null);
  const [modelName, setModelName] = useState<string>("");
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [trainingResults, setTrainingResults] = useState<TrainResponse | null>(null);

  // Prediction state
  const [text, setText] = useState<string>("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Prediction model selection state
  const [predModelType, setPredModelType] = useState<string>("nb");
  const [predFeatureGroups, setPredFeatureGroups] = useState<Set<string>>(new Set(["semantic"]));
  const [predictionModels, setPredictionModels] = useState<ModelRecord[]>([]);
  const [selectedPredictionModelId, setSelectedPredictionModelId] =
    useState<number | null>(null);
  const [predictionModelsLoading, setPredictionModelsLoading] =
    useState<boolean>(false);
  const [activatingModel, setActivatingModel] = useState<boolean>(false);

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

        // If NB model — filter mask to only emotional features
        if (!isLLM && wizardConfig?.models?.[0]?.model === "nb") {
          const modelParams = wizardConfig.models[0] as any || {};
          const oldMask = (modelParams?.additional_features as any)?.mask || {};

          // Keep only emotional features in the mask
          const cleanedMask: Record<string, boolean> = {};
          FEATURE_GROUP_DEFS.emotional.forEach((key) => {
            cleanedMask[key] = oldMask[key] === true;
          });

          // Update models array with cleaned mask
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

      // API returns { status, metrics: {...}, top_words: {...}, path, download_url } — flatten metrics to top level
      const flat = trainData.metrics ? { ...trainData, ...trainData.metrics } : trainData;
      console.log("[Training] top_words:", flat.top_words);
      setTrainingResults(flat);
    } catch (err: any) {
      setTrainingError(err.message || "Помилка під час навчання моделі.");
    } finally {
      setIsTraining(false);
    }
  };

  // ── Prediction ─────────────────────────────────────────────────────────
  const analyzeText = async () => {
    if (!text.trim()) {
      setError("Будь ласка, введіть текст для аналізу");
      return;
    }

    if (predModelType === "nb" && !selectedPredictionModelId) {
      setError("Оберіть модель для аналізу");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Build feature mask from selected groups
      const mask: Record<string, boolean> = {};
      predFeatureGroups.forEach((groupId) => {
        if (FEATURE_GROUP_DEFS[groupId]) {
          FEATURE_GROUP_DEFS[groupId].forEach((key) => {
            mask[key] = true;
          });
        }
      });

      const groups = Array.from(predFeatureGroups).filter((g) => g !== "semantic");
      const hasAdditional = groups.length > 0;

      let modelCfg: any;
      if (predModelType === "deberta") {
        modelCfg = { model: "deberta", integration_mode: "concat" };
      } else if (predModelType === "llm") {
        modelCfg = { model: "llm", mode: "single" };
      } else {
        modelCfg = { model: "nb" };
      }
      if (hasAdditional) {
        modelCfg.additional_features = { groups, mask };
      }

      const payload: any = {
        text,
        mode: "single",
        models: [modelCfg],
      };

      const { data } = await api.post("/predict", payload);
      setResult(data);
    } catch (err) {
      setError(
        "Не вдалося підключитися до сервера. Переконайтесь, що API запущено.",
      );
    } finally {
      setLoading(false);
    }
  };

  const fetchPredictionModels = async () => {
    setPredictionModelsLoading(true);
    try {
      const { data } = await api.get("/models");
      setPredictionModels(data);
      const active = data.find((m: any) => m.is_active);
      if (active) setSelectedPredictionModelId(active.id);
      else if (data.length > 0) setSelectedPredictionModelId(data[0].id);
    } catch (err) {
      // silently fail
    } finally {
      setPredictionModelsLoading(false);
    }
  };

  const handlePredictionModelChange = async (modelId: any) => {
    const numId = parseInt(modelId);
    setSelectedPredictionModelId(numId);
    const model = predictionModels.find((m) => m.id === numId);
    if (model && model.is_active) return;

    setActivatingModel(true);
    setError(null);
    try {
      await api.patch(`/models/${numId}/activate`);
      setPredictionModels((prev) =>
        prev.map((m) => ({ ...m, is_active: m.id === numId })),
      );
    } catch (err) {
      setError("Не вдалося переключити модель");
    } finally {
      setActivatingModel(false);
    }
  };

  const getConfidenceColor = (confidence: number, isFake: boolean) => {
    if (isFake) {
      return confidence > 0.7 ? "#dc3545" : "#ffc107";
    }
    return confidence > 0.7 ? "#28a745" : "#ffc107";
  };

  useEffect(() => {
    if (activeTab === "prediction") {
      fetchPredictionModels();
    }
  }, [activeTab]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!isLoggedIn) {
    return <AuthPage onLogin={handleLogin} />;
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Disinformation Detector</h1>
        <p>Система виявлення дезінформації на основі машинного навчання</p>
        <div className="header-user">
          <span>
            Вітаємо, <strong>{currentUser.username}</strong>
          </span>
          <button className="logout-btn" onClick={handleLogout}>
            Вихід
          </button>
        </div>
      </header>

      <div className="tabs">
        <button
          className={`tab ${activeTab === "training" ? "active" : ""}`}
          onClick={() => setActiveTab("training")}
        >
          Навчання моделі
        </button>
        <button
          className={`tab ${activeTab === "prediction" ? "active" : ""}`}
          onClick={() => setActiveTab("prediction")}
        >
          Аналіз тексту
        </button>
        <button
          className={`tab ${activeTab === "sources" ? "active" : ""}`}
          onClick={() => setActiveTab("sources")}
        >
          Реальні дані
        </button>
        <button
          className={`tab ${activeTab === "experiments" ? "active" : ""}`}
          onClick={() => setActiveTab("experiments")}
        >
          Експерименти
        </button>
        <button
          className={`tab ${activeTab === "models" ? "active" : ""}`}
          onClick={() => setActiveTab("models")}
        >
          Моделі
        </button>
        <button
          className={`tab ${activeTab === "profile" ? "active" : ""}`}
          onClick={() => setActiveTab("profile")}
        >
          Профіль
        </button>
      </div>

      <main className="main">
        {activeTab === "training" && (
          <div className="training-section">
            {/* ════ Section 1: Model Configuration ════ */}
            <div className="section-block">
              <h2>1. Конфігурація моделі</h2>
              <ClassificationWizard
                trainingMode
                onConfigChange={setWizardConfig}
              />
            </div>

            {/* ════ Section 2: Training ════ */}
            <div className="section-block">
              <h2>2. Навчання моделі</h2>
              <p style={{ fontSize: 13, color: "#666", marginBottom: 12 }}>
                Датасет: <strong>mdepak/fakenewsnet</strong> (завантажується автоматично в Colab)
              </p>

              {/* Model name field */}
              <div style={{ marginBottom: 16 }}>
                <label style={{ display: "block", fontSize: 13, fontWeight: 600, marginBottom: 6, color: "#333" }}>
                  Назва моделі
                </label>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <input
                    type="text"
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    placeholder="Назва для ідентифікації моделі..."
                    style={{
                      flex: 1, maxWidth: 360, padding: "8px 12px",
                      borderRadius: 8, border: "1px solid #ccc",
                      fontSize: 13, outline: "none",
                    }}
                  />
                  <button
                    onClick={generateModelName}
                    disabled={!wizardConfig}
                    style={{
                      padding: "8px 14px", borderRadius: 8, fontSize: 12, fontWeight: 600,
                      border: "1px solid #4a90d9", background: "#f0f7ff", color: "#4a90d9",
                      cursor: wizardConfig ? "pointer" : "not-allowed", whiteSpace: "nowrap",
                    }}
                  >
                    Авто-назва
                  </button>
                </div>
              </div>

              {(() => {
                const isLLM = wizardConfig?.models?.length === 1 && wizardConfig.models[0]?.model === "llm";
                return (
                  <button className="train-btn" onClick={startTraining} disabled={isTraining}>
                    {isTraining ? "Завантаження..." : isLLM ? "Результат" : "Почати навчання"}
                  </button>
                );
              })()}

              {isTraining && (
                <div className="progress-container">
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{ width: `${trainingProgress}%` }}
                    />
                  </div>
                  <span className="progress-text">
                    {Math.round(trainingProgress)}%
                  </span>
                </div>
              )}
              {trainingError && (
                <div className="upload-status error" style={{ marginTop: 10 }}>
                  {trainingError}
                </div>
              )}
            </div>

            {trainingResults && (
              <div className="section-block results-block">
                <h2>Результати тестування</h2>
                <div className="metrics-grid">
                  <div className="metric-card">
                    <span className="metric-value">
                      {(trainingResults.accuracy * 100).toFixed(1)}%
                    </span>
                    <span className="metric-label">Accuracy</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-value">
                      {(trainingResults.precision * 100).toFixed(1)}%
                    </span>
                    <span className="metric-label">Precision</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-value">
                      {(trainingResults.recall * 100).toFixed(1)}%
                    </span>
                    <span className="metric-label">Recall</span>
                  </div>
                  <div className="metric-card">
                    <span className="metric-value">
                      {(trainingResults.f1_score * 100).toFixed(1)}%
                    </span>
                    <span className="metric-label">F1 Score</span>
                  </div>
                </div>

                <div className="confusion-matrix">
                  <h3>Матриця помилок</h3>
                  <table className="matrix-table">
                    <thead>
                      <tr>
                        <th></th>
                        <th>Передбачено: Правда</th>
                        <th>Передбачено: Фейк</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th>Фактично: Правда</th>
                        <td className="tn">
                          {trainingResults.confusion_matrix?.tn || 0}
                        </td>
                        <td className="fp">
                          {trainingResults.confusion_matrix?.fp || 0}
                        </td>
                      </tr>
                      <tr>
                        <th>Фактично: Фейк</th>
                        <td className="fn">
                          {trainingResults.confusion_matrix?.fn || 0}
                        </td>
                        <td className="tp">
                          {trainingResults.confusion_matrix?.tp || 0}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

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
                    <div className="top-words-section" style={{ marginTop: 18 }}>
                      <h3>Хмара слів</h3>
                      <p style={{ fontSize: 12, color: "#888", marginBottom: 12 }}>
                        Розмір слова пропорційний його дискримінативній вазі.{" "}
                        <span style={{ color: "#dc3545" }}>Червоні</span> — маркери фейку,{" "}
                        <span style={{ color: "#28a745" }}>зелені</span> — маркери достовірності.
                      </p>
                      <div style={{
                        display: "flex", flexWrap: "wrap", gap: "6px 12px",
                        alignItems: "center", justifyContent: "center",
                        padding: "20px 16px",
                        background: "#fafafa", borderRadius: 12, border: "1px solid #eee",
                        minHeight: 120,
                      }}>
                        {shuffled.map((w) => (
                          <span
                            key={`${w.type}-${w.word}`}
                            title={`${w.word}: ${w.score.toFixed(3)} (${w.type === "fake" ? "FAKE" : "REAL"})`}
                            style={{
                              fontSize: fontSize(w.score),
                              fontWeight: w.score > (minScore + range * 0.6) ? 700 : 500,
                              color: w.type === "fake" ? "#dc3545" : "#28a745",
                              opacity: opacity(w.score),
                              cursor: "default",
                              lineHeight: 1.3,
                              transition: "transform 0.15s",
                            }}
                            onMouseEnter={(e) => (e.currentTarget.style.transform = "scale(1.15)")}
                            onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
                          >
                            {w.word}
                          </span>
                        ))}
                      </div>
                    </div>
                  );
                })()}

                <div className="training-info">
                  <p>
                    <strong>Модель:</strong>{" "}
                    {wizardConfig
                      ? `${wizardConfig.mode === "ensemble" ? "Ансамбль" : wizardConfig.mode === "single" ? "Одна модель" : "Мультівью"}${wizardConfig.models?.length ? ` [${wizardConfig.models.map((m: any) => m.model).join(" + ")}]` : ""}${wizardConfig.ensemble?.strategy ? ` — ${wizardConfig.ensemble.strategy.replace(/_/g, " ")}` : ""}`
                      : "N/A"}
                  </p>
                  <p>
                    <strong>Розмір тренувального набору:</strong>{" "}
                    {trainingResults.train_size || "N/A"}
                  </p>
                  <p>
                    <strong>Розмір тестового набору:</strong>{" "}
                    {trainingResults.test_size || "N/A"}
                  </p>
                  <p>
                    <strong>Час навчання:</strong>{" "}
                    {trainingResults.training_time ? `${trainingResults.training_time} s` : "N/A"}
                  </p>
                </div>

              </div>
            )}
          </div>
        )}

        {activeTab === "prediction" && <AnalysisPage />}

        {activeTab === "sources" && <SourcesPage />}
        {activeTab === "experiments" && <ExperimentsPage />}
        {activeTab === "models" && <ModelsPage />}
        {activeTab === "profile" && <ProfilePage user={currentUser} />}
      </main>

      <footer className="footer"></footer>
    </div>
  );
}

export default App;

