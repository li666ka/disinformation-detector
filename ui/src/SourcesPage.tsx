import React, { useEffect, useState } from "react";
import api from "./api";
import type { ModelRecord } from "./types";
import PostCard from "./PostCard";
import type { NewsItem, ClassifiedPost, SourceType } from "./types";

type Mode = "search" | "recent" | "url";

const SOURCE_LABELS: Record<SourceType, string> = {
  bluesky: "Bluesky",
  mastodon: "Mastodon",
  rss: "RSS",
};

export default function SourcesPage() {
  // Mode selection
  const [mode, setMode] = useState<Mode>("search");

  // Common state
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState<number | null>(null);
  const [loadingModels, setLoadingModels] = useState<boolean>(true);

  // Posts state
  const [posts, setPosts] = useState<ClassifiedPost[]>([]);
  const [loadingPosts, setLoadingPosts] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Search mode state
  const [query, setQuery] = useState<string>("");
  const [sources, setSources] = useState<Set<SourceType>>(
    new Set<SourceType>(["bluesky", "mastodon", "rss"]),
  );
  const [limit, setLimit] = useState<number>(20);

  // URL fetch mode state
  const [postUrl, setPostUrl] = useState<string>("");

  // Classification state (per-post loading flags)
  const [classifyingIds, setClassifyingIds] = useState<Set<string>>(new Set());
  const [batchClassifying, setBatchClassifying] = useState<boolean>(false);

  // Load available models for classification
  useEffect(() => {
    api
      .get("/models")
      .then(({ data }) => {
        setModels(data);
        const active = data.find((m: ModelRecord) => m.is_active);
        if (active) setModelId(active.id);
        else if (data.length > 0) setModelId(data[0].id);
      })
      .catch(() => setError("Не вдалося завантажити моделі"))
      .finally(() => setLoadingModels(false));
  }, []);

  const toggleSource = (src: SourceType) => {
    setSources((prev) => {
      const next = new Set(prev);
      if (next.has(src)) next.delete(src);
      else next.add(src);
      return next;
    });
  };

  // ── Search posts by query ─────────────────────────────────────────────
  const handleSearch = async () => {
    if (!query.trim()) {
      setError("Введіть пошуковий запит");
      return;
    }
    if (sources.size === 0) {
      setError("Оберіть хоча б одне джерело");
      return;
    }

    setLoadingPosts(true);
    setError(null);
    setPosts([]);

    try {
      const { data } = await api.get<{ posts: NewsItem[] }>("/sources/search", {
        params: {
          query,
          sources: Array.from(sources).join(","),
          limit,
        },
      });
      setPosts(data.posts.map((p) => ({ ...p, classification: null })));
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Помилка пошуку");
    } finally {
      setLoadingPosts(false);
    }
  };

  // ── Get recent posts ──────────────────────────────────────────────────
  const handleFetchRecent = async () => {
    if (sources.size === 0) {
      setError("Оберіть хоча б одне джерело");
      return;
    }

    setLoadingPosts(true);
    setError(null);
    setPosts([]);

    try {
      const { data } = await api.get<{ posts: NewsItem[] }>("/sources/recent", {
        params: {
          sources: Array.from(sources).join(","),
          limit,
        },
      });
      setPosts(data.posts.map((p) => ({ ...p, classification: null })));
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === "string" ? detail : "Не вдалося отримати пости",
      );
    } finally {
      setLoadingPosts(false);
    }
  };

  // ── Fetch single post by URL ──────────────────────────────────────────
  const handleFetchUrl = async () => {
    if (!postUrl.trim()) {
      setError("Введіть URL поста");
      return;
    }

    setLoadingPosts(true);
    setError(null);
    setPosts([]);

    try {
      const { data } = await api.post<NewsItem>("/sources/fetch-url", {
        url: postUrl,
      });
      setPosts([{ ...data, classification: null }]);
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === "string" ? detail : "Не вдалося завантажити пост",
      );
    } finally {
      setLoadingPosts(false);
    }
  };

  // ── Classify single post ──────────────────────────────────────────────
  const classifyPost = async (post: ClassifiedPost) => {
    if (!modelId) {
      setError("Оберіть модель для класифікації");
      return;
    }

    setClassifyingIds((prev) => new Set(prev).add(post.id));

    try {
      const { data } = await api.post("/analyze", {
        text: post.text,
        model_id: modelId,
      });
      setPosts((prev) =>
        prev.map((p) =>
          p.id === post.id
            ? {
                ...p,
                classification: {
                  label: data.label,
                  confidence: data.confidence,
                  probability: data.probability,
                },
              }
            : p,
        ),
      );
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      setError(
        typeof detail === "string" ? detail : "Помилка класифікації",
      );
    } finally {
      setClassifyingIds((prev) => {
        const next = new Set(prev);
        next.delete(post.id);
        return next;
      });
    }
  };

  // ── Classify all unclassified posts ───────────────────────────────────
  const classifyAll = async () => {
    if (!modelId) {
      setError("Оберіть модель для класифікації");
      return;
    }
    const unclassified = posts.filter((p) => !p.classification);
    if (unclassified.length === 0) return;

    setBatchClassifying(true);
    setError(null);

    // Classify sequentially to respect rate limits
    for (const post of unclassified) {
      await classifyPost(post);
    }

    setBatchClassifying(false);
  };

  // ── Stats for classified posts ────────────────────────────────────────
  const classifiedPosts = posts.filter((p) => p.classification);
  const fakeCount = classifiedPosts.filter(
    (p) => p.classification?.label === "FAKE",
  ).length;
  const realCount = classifiedPosts.filter(
    (p) => p.classification?.label === "REAL",
  ).length;

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      {/* ── Mode tabs ────────────────────────────────────────────────── */}
      <div className="section-block">
        <h2>Режим роботи з джерелами</h2>
        <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
          {[
            {
              id: "search" as Mode,
              title: "Пошук за словами",
              desc: "Знайти пости за ключовими словами в обраних джерелах",
            },
            {
              id: "recent" as Mode,
              title: "Свіжі пости",
              desc: "Останні публікації в обраних джерелах",
            },
            {
              id: "url" as Mode,
              title: "За URL",
              desc: "Завантажити конкретний пост/статтю за посиланням",
            },
          ].map((m) => (
            <div
              key={m.id}
              onClick={() => {
                setMode(m.id);
                setPosts([]);
                setError(null);
              }}
              style={{
                flex: 1,
                padding: 16,
                borderRadius: 10,
                border: `2px solid ${mode === m.id ? "#4a90d9" : "#e0e0e0"}`,
                background: mode === m.id ? "#f0f7ff" : "white",
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 4 }}>
                {m.title}
              </div>
              <div style={{ fontSize: 12, color: "#666" }}>{m.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Sources selection (for search & recent) ──────────────────── */}
      {mode !== "url" && (
        <div className="section-block">
          <h2>Джерела даних</h2>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {(Object.keys(SOURCE_LABELS) as SourceType[]).map((src) => {
              const active = sources.has(src);
              return (
                <label
                  key={src}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "8px 16px",
                    borderRadius: 20,
                    border: `1px solid ${active ? "#4a90d9" : "#ddd"}`,
                    background: active ? "#eef6ff" : "white",
                    cursor: "pointer",
                    fontSize: 13,
                    userSelect: "none",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={active}
                    onChange={() => toggleSource(src)}
                    style={{ accentColor: "#4a90d9" }}
                  />
                  {SOURCE_LABELS[src]}
                </label>
              );
            })}
          </div>

          <div style={{ marginTop: 16, display: "flex", gap: 16, alignItems: "center" }}>
            <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 13 }}>Кількість постів:</span>
              <select
                className="translator-select"
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
                style={{ width: 80 }}
              >
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </label>
          </div>
        </div>
      )}

      {/* ── Search input ─────────────────────────────────────────────── */}
      {mode === "search" && (
        <div className="section-block">
          <h2>Пошуковий запит</h2>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              type="text"
              className="text-input"
              placeholder="Наприклад: election, COVID, climate change..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleSearch()}
              style={{
                flex: 1,
                padding: "10px 14px",
                border: "2px solid #e0e0e0",
                borderRadius: 8,
                fontSize: 14,
              }}
            />
            <button
              className="analyze-btn"
              onClick={handleSearch}
              disabled={loadingPosts || !query.trim()}
              style={{ padding: "10px 24px" }}
            >
              {loadingPosts ? "Пошук..." : "Шукати"}
            </button>
          </div>
        </div>
      )}

      {/* ── Recent mode: just a fetch button ─────────────────────────── */}
      {mode === "recent" && (
        <div className="section-block">
          <h2>Отримати свіжі публікації</h2>
          <button
            className="analyze-btn"
            onClick={handleFetchRecent}
            disabled={loadingPosts}
            style={{ width: "auto", padding: "10px 24px" }}
          >
            {loadingPosts ? "Завантаження..." : "Завантажити"}
          </button>
        </div>
      )}

      {/* ── URL mode ─────────────────────────────────────────────────── */}
      {mode === "url" && (
        <div className="section-block">
          <h2>URL поста або статті</h2>
          <p style={{ fontSize: 12, color: "#888", marginBottom: 10 }}>
            Підтримуються: Bluesky (bsky.app/profile/.../post/...), Mastodon URLs
            та посилання на статті з RSS-стрічок
          </p>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              type="url"
              className="text-input"
              placeholder="https://bsky.app/profile/user.bsky.social/post/..."
              value={postUrl}
              onChange={(e) => setPostUrl(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleFetchUrl()}
              style={{
                flex: 1,
                padding: "10px 14px",
                border: "2px solid #e0e0e0",
                borderRadius: 8,
                fontSize: 14,
              }}
            />
            <button
              className="analyze-btn"
              onClick={handleFetchUrl}
              disabled={loadingPosts || !postUrl.trim()}
              style={{ padding: "10px 24px" }}
            >
              {loadingPosts ? "..." : "Завантажити"}
            </button>
          </div>
        </div>
      )}

      {/* ── Model selector for classification ────────────────────────── */}
      <div className="section-block">
        <h2>Модель для класифікації</h2>
        {loadingModels ? (
          <p style={{ color: "#888", fontSize: 13 }}>Завантаження моделей...</p>
        ) : models.length === 0 ? (
          <p style={{ color: "#888", fontSize: 13 }}>
            Навчених моделей немає. Спочатку навчіть модель у вкладці "Навчання
            моделі".
          </p>
        ) : (
          <select
            className="translator-select"
            value={modelId ?? ""}
            onChange={(e) => setModelId(Number(e.target.value))}
            style={{ maxWidth: 400 }}
          >
            {models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name || m.model_type}
                {m.accuracy != null
                  ? ` — ${(m.accuracy * 100).toFixed(1)}%`
                  : ""}
                {m.is_active ? " ✓ (активна)" : ""}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* ── Error ─────────────────────────────────────────────────────── */}
      {error && <div className="error">{error}</div>}

      {/* ── Results header + batch classify ──────────────────────────── */}
      {posts.length > 0 && (
        <div className="section-block">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 14,
            }}
          >
            <h2 style={{ margin: 0 }}>
              Знайдено постів: {posts.length}
              {classifiedPosts.length > 0 && (
                <span style={{ fontSize: 13, fontWeight: 400, marginLeft: 12 }}>
                  (проаналізовано: {classifiedPosts.length})
                </span>
              )}
            </h2>
            {modelId && (
              <button
                className="analyze-btn"
                onClick={classifyAll}
                disabled={batchClassifying || classifyingIds.size > 0}
                style={{ width: "auto", padding: "8px 20px", fontSize: 13 }}
              >
                {batchClassifying
                  ? `Аналіз... (${classifiedPosts.length}/${posts.length})`
                  : "Аналізувати всі"}
              </button>
            )}
          </div>

          {/* Aggregated stats */}
          {classifiedPosts.length > 0 && (
            <div
              style={{
                display: "flex",
                gap: 12,
                padding: 12,
                background: "#f8f9fa",
                borderRadius: 8,
                marginBottom: 16,
              }}
            >
              <div style={{ flex: 1, textAlign: "center" }}>
                <div
                  style={{
                    fontSize: 24,
                    fontWeight: 700,
                    color: "#dc3545",
                  }}
                >
                  {fakeCount}
                </div>
                <div style={{ fontSize: 12, color: "#666" }}>Фейки</div>
              </div>
              <div style={{ flex: 1, textAlign: "center" }}>
                <div
                  style={{
                    fontSize: 24,
                    fontWeight: 700,
                    color: "#28a745",
                  }}
                >
                  {realCount}
                </div>
                <div style={{ fontSize: 12, color: "#666" }}>Достовірні</div>
              </div>
              <div style={{ flex: 1, textAlign: "center" }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: "#4a90d9" }}>
                  {classifiedPosts.length > 0
                    ? `${((fakeCount / classifiedPosts.length) * 100).toFixed(0)}%`
                    : "—"}
                </div>
                <div style={{ fontSize: 12, color: "#666" }}>% фейків</div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Posts list ───────────────────────────────────────────────── */}
      {posts.map((post) => (
        <PostCard
          key={post.id}
          post={post}
          classifying={classifyingIds.has(post.id)}
          canClassify={!!modelId}
          onClassify={() => classifyPost(post)}
        />
      ))}

      {/* ── Empty state ──────────────────────────────────────────────── */}
      {posts.length === 0 && !loadingPosts && !error && (
        <div
          className="section-block"
          style={{ textAlign: "center", color: "#888" }}
        >
          <p style={{ margin: 0 }}>
            {mode === "search" && "Введіть запит і натисніть «Шукати»"}
            {mode === "recent" && "Натисніть «Завантажити» щоб отримати свіжі пости"}
            {mode === "url" && "Вставте URL поста і натисніть «Завантажити»"}
          </p>
        </div>
      )}
    </div>
  );
}
