import React, { useEffect, useState } from "react";
import api from "./api";

function ExperimentsPage() {
  const [experiments, setExperiments] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<any>(null);

  const loadExperiments = () => {
    setLoading(true);
    api
      .get("/experiments")
      .then(({ data }) => setExperiments(data))
      .catch(() => setError("Не вдалося завантажити експерименти"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadExperiments();
  }, []);

  const deleteExperiment = async (id: number) => {
    try {
      await api.delete(`/experiments/${id}`);
      setExperiments((prev) => prev.filter((e) => e.id !== id));
    } catch {
      setError("Не вдалося видалити експеримент");
    }
  };

  const deleteAll = async () => {
    if (!window.confirm("Видалити всі експерименти?")) return;
    try {
      await api.delete("/experiments");
      setExperiments([]);
    } catch {
      setError("Не вдалося видалити експерименти");
    }
  };

  const formatDate = (iso: any) => new Date(iso).toLocaleString("uk-UA");

  if (loading)
    return (
      <div className="section-block">
        <p>Завантаження...</p>
      </div>
    );

  return (
    <div className="section-block">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h2 style={{ margin: 0 }}>Журнал навчання</h2>
        {experiments.length > 0 && (
          <button
            className="train-btn"
            style={{ background: "#dc3545", fontSize: 13, padding: "6px 14px" }}
            onClick={deleteAll}
          >
            Видалити всі
          </button>
        )}
      </div>
      {error && (
        <div
          className="upload-status error"
          style={{ marginBottom: 16 }}
        >
          {error}
        </div>
      )}
      {experiments.length === 0 ? (
        <p style={{ color: "#666" }}>
          Експериментів ще немає. Запустіть перше навчання у вкладці
          "Навчання моделі".
        </p>
      ) : (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Тип моделі</th>
                <th>Accuracy</th>
                <th>F1</th>
                <th>Статус</th>
                <th>Дата</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((exp) => (
                <tr key={exp.id}>
                  <td className="mono" title={exp.experiment_id}>
                    {String(exp.experiment_id).slice(0, 8)}...
                  </td>
                  <td>{exp.model_type}</td>
                  <td>
                    {exp.accuracy != null
                      ? `${(exp.accuracy * 100).toFixed(1)}%`
                      : "—"}
                  </td>
                  <td>
                    {exp.f1_score != null
                      ? `${(exp.f1_score * 100).toFixed(1)}%`
                      : "—"}
                  </td>
                  <td>
                    <span
                      className={`badge badge-${
                        exp.status === "success" ? "active" : "failed"
                      }`}
                    >
                      {exp.status === "success" ? "Успішно" : "Помилка"}
                    </span>
                  </td>
                  <td>{formatDate(exp.created_at)}</td>
                  <td>
                    <button
                      style={{
                        background: "none",
                        border: "none",
                        color: "#dc3545",
                        cursor: "pointer",
                        fontSize: 16,
                        padding: "2px 6px",
                      }}
                      title="Видалити"
                      onClick={() => deleteExperiment(exp.id)}
                    >
                      &times;
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default ExperimentsPage;

