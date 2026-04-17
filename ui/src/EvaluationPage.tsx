import React, { useState } from "react";
import api from "./api";
import type { EvalResponse, EvalSample } from "./types";

function EvaluationPage() {
  const [maxSamples, setMaxSamples] = useState<number>(50);
  const [loading, setLoading]       = useState(false);
  const [result, setResult]         = useState<EvalResponse | null>(null);
  const [error, setError]           = useState<string | null>(null);
  const [filter, setFilter]         = useState<"all" | "correct" | "wrong">("all");
  const [searchText, setSearchText] = useState("");
  const [expanded, setExpanded]     = useState<Set<number>>(new Set());

  const run = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setExpanded(new Set());
    try {
      const { data } = await api.post<EvalResponse>("/evaluate", {
        max_samples: maxSamples,
        test_size: 0.2,
      });
      setResult(data);
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Помилка оцінювання");
    } finally {
      setLoading(false);
    }
  };

  const toggleExpand = (index: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(index) ? next.delete(index) : next.add(index);
      return next;
    });
  };

  const visibleSamples = (result?.samples ?? []).filter((s) => {
    if (filter === "correct" && !s.correct) return false;
    if (filter === "wrong"   &&  s.correct) return false;
    if (searchText && !s.text.toLowerCase().includes(searchText.toLowerCase())) return false;
    return true;
  });

  const m = result?.metrics;

  return (
    <div className="section-block">
      <h2>Оцінювання Gemini (Zero-Shot)</h2>
      <p style={{ color: "#666", marginBottom: 20 }}>
        Запускає Gemini zero-shot класифікацію на тестовій вибірці датасету та повертає метрики.
        При активному Colab — виконується на стороні Colab (GEMINI_API_KEY з Secrets).
      </p>

      {/* Controls */}
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 20, flexWrap: "wrap" }}>
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>Кількість зразків:</span>
          <input
            type="number"
            min={10}
            max={500}
            value={maxSamples}
            onChange={(e) => setMaxSamples(Number(e.target.value))}
            style={{ width: 80, padding: "6px 8px", border: "1px solid #ddd", borderRadius: 6 }}
            disabled={loading}
          />
        </label>
        <button
          className="train-btn"
          onClick={run}
          disabled={loading}
          style={{ minWidth: 180 }}
        >
          {loading ? "Оцінювання…" : "Запустити оцінювання"}
        </button>
      </div>

      {loading && (
        <div style={{ padding: "20px 0", color: "#666" }}>
          ⏳ Класифікація {maxSamples} зразків через Gemini API (~{Math.round(maxSamples * 1.1 / 60)} хв)…
        </div>
      )}

      {error && (
        <div className="upload-status error" style={{ marginBottom: 16 }}>{error}</div>
      )}

      {/* Metrics */}
      {m && (
        <>
          <div className="metrics-grid" style={{ marginBottom: 24 }}>
            {(["accuracy", "precision", "recall", "f1_score"] as const).map((key) => (
              <div className="metric-card" key={key}>
                <span className="metric-value">{(m[key] * 100).toFixed(1)}%</span>
                <span className="metric-label">
                  {key === "f1_score" ? "F1 Score" : key.charAt(0).toUpperCase() + key.slice(1)}
                </span>
              </div>
            ))}
          </div>

          <div className="confusion-matrix" style={{ marginBottom: 24 }}>
            <h3>
              Матриця помилок{" "}
              <small style={{ fontWeight: 400, color: "#888" }}>({m.samples_evaluated} зразків)</small>
            </h3>
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
                  <td className="tn">{m.confusion_matrix.tn}</td>
                  <td className="fp">{m.confusion_matrix.fp}</td>
                </tr>
                <tr>
                  <th>Фактично: Фейк</th>
                  <td className="fn">{m.confusion_matrix.fn}</td>
                  <td className="tp">{m.confusion_matrix.tp}</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Filter + search */}
          <div style={{ display: "flex", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
            {(["all", "correct", "wrong"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                style={{
                  padding: "5px 14px",
                  borderRadius: 20,
                  border: "1px solid #ddd",
                  background: filter === f ? "#4f46e5" : "#fff",
                  color: filter === f ? "#fff" : "#333",
                  cursor: "pointer",
                  fontSize: 13,
                }}
              >
                {f === "all"
                  ? `Усі (${result!.samples.length})`
                  : f === "correct"
                  ? `✓ Правильні (${result!.samples.filter((s) => s.correct).length})`
                  : `✗ Помилкові (${result!.samples.filter((s) => !s.correct).length})`}
              </button>
            ))}
            <input
              type="text"
              placeholder="Пошук у тексті…"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              style={{
                padding: "5px 10px",
                border: "1px solid #ddd",
                borderRadius: 6,
                flex: 1,
                minWidth: 180,
                fontSize: 13,
              }}
            />
          </div>

          {/* Sample table */}
          <div className="table-wrapper">
            <table className="data-table" style={{ fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ width: 40 }}>#</th>
                  <th>Текст</th>
                  <th style={{ width: 90 }}>Факт</th>
                  <th style={{ width: 90 }}>Gemini</th>
                  <th style={{ width: 80 }}>Conf.</th>
                  <th style={{ width: 70 }}>Вердикт</th>
                  <th style={{ width: 100 }}></th>
                </tr>
              </thead>
              <tbody>
                {visibleSamples.map((s: EvalSample) => (
                  <React.Fragment key={s.index}>
                    <tr style={{ background: s.correct ? "inherit" : "#fff5f5" }}>
                      <td className="mono">{s.index}</td>
                      <td style={{ maxWidth: 340, wordBreak: "break-word" }}>{s.text}</td>
                      <td>
                        <span className={`badge badge-${s.true_str === "FAKE" ? "failed" : "active"}`}>
                          {s.true_str}
                        </span>
                      </td>
                      <td>
                        <span className={`badge badge-${s.pred_str === "FAKE" ? "failed" : s.pred_str === "UNCERTAIN" ? "" : "active"}`}>
                          {s.pred_str}
                        </span>
                      </td>
                      <td className="mono">{(s.confidence * 100).toFixed(0)}%</td>
                      <td style={{ fontSize: 16, textAlign: "center" }}>
                        {s.correct ? "✓" : "✗"}
                      </td>
                      <td>
                        {s.reason && (
                          <button
                            onClick={() => toggleExpand(s.index)}
                            style={{
                              padding: "3px 10px",
                              borderRadius: 12,
                              border: "1px solid #4f46e5",
                              background: expanded.has(s.index) ? "#4f46e5" : "#fff",
                              color: expanded.has(s.index) ? "#fff" : "#4f46e5",
                              cursor: "pointer",
                              fontSize: 12,
                              whiteSpace: "nowrap",
                            }}
                          >
                            {expanded.has(s.index) ? "Сховати" : "Результат"}
                          </button>
                        )}
                      </td>
                    </tr>
                    {expanded.has(s.index) && s.reason && (
                      <tr style={{ background: "#f8f7ff" }}>
                        <td></td>
                        <td colSpan={6} style={{ padding: "8px 12px", color: "#444", fontSize: 13, fontStyle: "italic" }}>
                          💬 <strong>Пояснення Gemini:</strong> {s.reason}
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

export default EvaluationPage;
