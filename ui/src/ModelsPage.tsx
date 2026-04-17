import React, { useEffect, useState } from "react";
import api from "./api";
import type { ModelRecord } from "./types";

function ModelsPage() {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [activating, setActivating] = useState<number | null>(null);
  const [deletingAll, setDeletingAll] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = async () => {
    try {
      const { data } = await api.get("/models");
      setModels(data);
    } catch (err) {
      setError("Не вдалося завантажити список моделей");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleActivate = async (modelId: number) => {
    setActivating(modelId);
    setError(null);
    try {
      await api.patch(`/models/${modelId}/activate`);
      await fetchModels();
    } catch (err: any) {
      setError(err.response?.data?.detail || "Помилка активації моделі");
    } finally {
      setActivating(null);
    }
  };

  const handleDeleteAll = async () => {
    if (!window.confirm("Видалити всі моделі? Цю дію не можна скасувати.")) return;
    setDeletingAll(true);
    setError(null);
    try {
      await api.delete("/models");
      await fetchModels();
    } catch (err: any) {
      setError(err.response?.data?.detail || "Помилка видалення моделей");
    } finally {
      setDeletingAll(false);
    }
  };

  const formatDate = (iso: string) => new Date(iso).toLocaleString("uk-UA");

  if (loading)
    return (
      <div className="section-block">
        <p>Завантаження...</p>
      </div>
    );

  return (
    <div className="section-block">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0 }}>Збережені моделі</h2>
        {models.length > 0 && (
          <button
            className="activate-btn"
            style={{ background: "#e74c3c", color: "#fff", border: "none" }}
            onClick={handleDeleteAll}
            disabled={deletingAll}
          >
            {deletingAll ? "Видалення..." : "Видалити всі"}
          </button>
        )}
      </div>
      {error && (
        <div className="upload-status error" style={{ marginBottom: 16 }}>
          {error}
        </div>
      )}
      {models.length === 0 ? (
        <p style={{ color: "#666" }}>
          Моделей ще немає. Навчіть першу модель у вкладці "Навчання моделі".
        </p>
      ) : (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Файл</th>
                <th>Тип моделі</th>
                <th>Accuracy</th>
                <th>Статус</th>
                <th>Дата</th>
                <th>Дія</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr
                  key={m.id}
                  className={m.is_active ? "row-active" : ""}
                >
                  <td className="mono">{m.filename}</td>
                  <td>{m.model_type}</td>
                  <td>
                    {m.accuracy != null
                      ? `${(m.accuracy * 100).toFixed(1)}%`
                      : "—"}
                  </td>
                  <td>
                    {m.is_active ? (
                      <span className="badge badge-active">Активна</span>
                    ) : (
                      <span className="badge badge-inactive">Неактивна</span>
                    )}
                  </td>
                  <td>{formatDate(m.created_at)}</td>
                  <td>
                    {!m.is_active && (
                      <button
                        className="activate-btn"
                        onClick={() => handleActivate(m.id)}
                        disabled={activating === m.id}
                      >
                        {activating === m.id ? "..." : "Активувати"}
                      </button>
                    )}
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

export default ModelsPage;

