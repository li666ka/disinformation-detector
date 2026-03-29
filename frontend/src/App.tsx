import React, { useEffect, useState } from "react";
import "./App.css";
import AuthPage from "./AuthPage";
import ProfilePage from "./ProfilePage";
import ModelsPage from "./ModelsPage";
import ExperimentsPage from "./ExperimentsPage";
import ClassificationWizard from "./ClassificationWizard";
import api from "./api";

function App() {
  // Auth state
  const [currentUser, setCurrentUser] = useState<any>(() => {
    const stored = localStorage.getItem("user");
    return stored ? JSON.parse(stored) : null;
  });
  const isLoggedIn =
    currentUser !== null && localStorage.getItem("token") !== null;

  const handleLogin = (user: any) => setCurrentUser(user);
  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setCurrentUser(null);
  };

  const [activeTab, setActiveTab] = useState<string>("training");

  // ── Training state ─────────────────────────────────────────────────────
  const [wizardConfig, setWizardConfig] = useState<any>(null);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [trainingResults, setTrainingResults] = useState<any>(null);

  // Prediction state
  const [text, setText] = useState<string>("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<any>(null);

  // Prediction model selection state
  const [predictionModels, setPredictionModels] = useState<any[]>([]);
  const [selectedPredictionModelId, setSelectedPredictionModelId] =
    useState<any>(null);
  const [predictionModelsLoading, setPredictionModelsLoading] =
    useState<boolean>(false);
  const [activatingModel, setActivatingModel] = useState<boolean>(false);

  // ── Training ───────────────────────────────────────────────────────────
  const [trainingError, setTrainingError] = useState<string | null>(null);

  const startTraining = async () => {
    setIsTraining(true);
    setTrainingProgress(0);
    setTrainingResults(null);

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
        const { data } = await api.post("/train", {
          ...(wizardConfig || {}),
        });
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

      setTrainingResults(trainData);
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

    if (!selectedPredictionModelId) {
      setError("Оберіть модель для аналізу");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: any = {
        text,
        mode: "single",
        models: [{ model: "nb" }],
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
              <button
                className="train-btn"
                onClick={startTraining}
                disabled={isTraining}
              >
                {isTraining ? "Навчання..." : "Почати навчання"}
              </button>

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

        {activeTab === "prediction" && (
          <div className="prediction-section">
            <div className="section-block">
              <h2>Активна модель</h2>
              {predictionModelsLoading ? (
                <p style={{ color: "#666" }}>Завантаження моделей...</p>
              ) : predictionModels.length > 0 ? (
                <div className="model-selector-row">
                  <select
                    className="translator-select"
                    value={selectedPredictionModelId || ""}
                    onChange={(e) =>
                      handlePredictionModelChange((e.target as HTMLSelectElement).value)
                    }
                    disabled={activatingModel}
                  >
                    {predictionModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.model_type}
                        {m.accuracy != null
                          ? ` — ${(m.accuracy * 100).toFixed(1)}%`
                          : ""}
                        {m.is_active ? " ✓" : ""}
                      </option>
                    ))}
                  </select>
                  {activatingModel && (
                    <span style={{ marginLeft: 10, color: "#666" }}>
                      Завантаження...
                    </span>
                  )}
                </div>
              ) : (
                <p style={{ color: "#666" }}>
                  Навчені моделі відсутні. Спочатку навчіть модель.
                </p>
              )}
            </div>

            <div className="section-block">
              <h2>Текст для аналізу</h2>
              <textarea
                className="text-input"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Вставте текст для перевірки..."
                rows={6}
              />


              <button
                className="analyze-btn"
                onClick={analyzeText}
                disabled={
                  loading || activatingModel || !selectedPredictionModelId
                }
              >
                {loading ? "Аналіз..." : "Аналізувати"}
              </button>
            </div>

            {error && <div className="error">{error}</div>}

            {result && (
              <div className="result">
                <div
                  className="verdict"
                  style={{
                    backgroundColor: getConfidenceColor(result.confidence, result.is_fake),
                  }}
                >
                  <h2>{result.is_fake ? "ДЕЗІНФОРМАЦІЯ" : "ДОСТОВІРНО"}</h2>
                  <p>Впевненість: {(result.confidence * 100).toFixed(1)}%</p>
                </div>
                <div className="details">
                  <h3>Деталі аналізу:</h3>
                  <ul>
                    <li>
                      Модель:{" "}
                      {predictionModels.find(
                        (m) => m.id === selectedPredictionModelId,
                      )?.model_type || "—"}
                    </li>
                    {result.mode && (
                      <li>
                        Режим:{" "}
                        {result.mode === "ensemble" ? "Ансамбль" : "Одна модель"}
                      </li>
                    )}
                  </ul>
                </div>

                {result.individual_results &&
                  result.individual_results.length > 0 && (
                    <div className="section-block" style={{ marginTop: 14 }}>
                      <h3>Результати моделей:</h3>
                      {result.individual_results.map((r: any, i: number) => (
                        <div
                          key={i}
                          style={{
                            padding: "10px 14px",
                            marginBottom: 8,
                            borderRadius: 8,
                            background: r.label === "FAKE" ? "#fff5f5" : "#f0fff0",
                            border: `1px solid ${r.label === "FAKE" ? "#f5c6cb" : "#c3e6cb"
                              }`,
                          }}
                        >
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "center",
                            }}
                          >
                            <strong>
                              {r.model === "nb"
                                ? "Naive Bayes"
                                : r.model === "xlm_r"
                                  ? "XLM-RoBERTa"
                                  : r.model === "llm"
                                    ? "GPT-4o-mini"
                                    : r.model}
                            </strong>
                            <span
                              style={{
                                fontWeight: 600,
                                color:
                                  r.label === "FAKE" ? "#dc3545" : "#28a745",
                              }}
                            >
                              {r.label}{" "}
                              {r.probability != null
                                ? `(${(r.probability * 100).toFixed(1)}%)`
                                : ""}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                {result.ensemble_result && (
                  <div className="section-block" style={{ marginTop: 14 }}>
                    <h3>Результат ансамблю:</h3>
                    <p>
                      <strong>Стратегія:</strong> {result.ensemble_result.strategy}
                    </p>
                    <p>
                      <strong>Голоси:</strong> FAKE:{" "}
                      {result.ensemble_result.votes?.FAKE || 0}, REAL:{" "}
                      {result.ensemble_result.votes?.REAL || 0}
                    </p>
                  </div>
                )}

                {result.feature_values &&
                  Object.keys(result.feature_values).some(
                    (k) => result.feature_values[k] !== 0,
                  ) && (
                    <div className="section-block" style={{ marginTop: 14 }}>
                      <h3>Обчислені ознаки:</h3>
                      <table className="data-table" style={{ fontSize: 13 }}>
                        <thead>
                          <tr>
                            <th>Ознака</th>
                            <th>Значення</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(result.feature_values)
                            .filter(([, v]: any) => v !== 0)
                            .map(([key, val]: any) => (
                              <tr key={key}>
                                <td>{key}</td>
                                <td style={{ fontFamily: "monospace" }}>
                                  {typeof val === "number"
                                    ? val.toFixed(4)
                                    : val}
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  )}
              </div>
            )}
          </div>
        )}

        {activeTab === "experiments" && <ExperimentsPage />}
        {activeTab === "models" && <ModelsPage />}
        {activeTab === "profile" && <ProfilePage user={currentUser} />}
      </main>

      <footer className="footer"></footer>
    </div>
  );
}

export default App;

