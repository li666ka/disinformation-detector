import React, { useEffect, useState } from "react";
import api from "./api";
import type { ModelRecord } from "./types";

interface AnalyzeResult {
  label: "FAKE" | "REAL";
  confidence: number;
  probability: number;
}

export default function AnalysisPage() {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState<number | null>(null);
  const [text, setText] = useState<string>("");
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingModels, setLoadingModels] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.get("/models")
      .then(({ data }) => {
        setModels(data);
        if (data.length > 0) setModelId(data[0].id);
      })
      .catch(() => {})
      .finally(() => setLoadingModels(false));
  }, []);

  const handleSubmit = async () => {
    if (!text.trim()) { setError("Введіть текст для аналізу"); return; }
    if (!modelId) { setError("Оберіть модель"); return; }

    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const { data } = await api.post("/analyze", { text, model_id: modelId });
      setResult(data);
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Помилка під час аналізу");
    } finally {
      setLoading(false);
    }
  };

  const isFake = result?.label === "FAKE";

  const selectedModel = models.find((m) => m.id === modelId);

  return (
    <div style={{ maxWidth: 680, margin: "0 auto" }}>
      <div className="section-block">
        <h2>Модель</h2>
        {loadingModels ? (
          <p style={{ color: "#888", fontSize: 13 }}>Завантаження моделей...</p>
        ) : models.length === 0 ? (
          <p style={{ color: "#888", fontSize: 13 }}>
            Навчених моделей немає. Спочатку навчіть модель на вкладці "Навчання моделі".
          </p>
        ) : (
          <select
            className="translator-select"
            value={modelId ?? ""}
            onChange={(e) => { setModelId(Number(e.target.value)); setResult(null); setError(null); }}
            style={{ maxWidth: 400 }}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name || m.model_type}
                {m.accuracy != null ? ` — ${(m.accuracy * 100).toFixed(1)}%` : ""}
              </option>
            ))}
          </select>
        )}
        {selectedModel && (
          <p style={{ fontSize: 11, color: "#888", marginTop: 6 }}>
            Тип: <strong>{selectedModel.model_type}</strong>
            {selectedModel.accuracy != null && <> · Accuracy: <strong>{(selectedModel.accuracy * 100).toFixed(1)}%</strong></>}
          </p>
        )}
      </div>

      <div className="section-block">
        <h2>Текст для аналізу</h2>
        <textarea
          className="text-input"
          rows={7}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Вставте текст новини для перевірки..."
        />
        <button
          className="analyze-btn"
          onClick={handleSubmit}
          disabled={loading || models.length === 0}
          style={{ marginTop: 12 }}
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
              backgroundColor: isFake
                ? result.confidence > 0.7 ? "#dc3545" : "#ffc107"
                : result.confidence > 0.7 ? "#28a745" : "#ffc107",
            }}
          >
            <h2>{isFake ? "ДЕЗІНФОРМАЦІЯ" : "ДОСТОВІРНО"}</h2>
            <p>Впевненість: {(result.confidence * 100).toFixed(1)}%</p>
          </div>
          <div className="details">
            <ul>
              <li><strong>Модель:</strong> {selectedModel?.name || selectedModel?.model_type}</li>
              <li><strong>Ймовірність FAKE:</strong> {(result.probability * 100).toFixed(1)}%</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
