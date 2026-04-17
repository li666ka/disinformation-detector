import React, { useState } from "react";
import type { ClassifiedPost } from "./types";

interface PostCardProps {
  post: ClassifiedPost;
  classifying: boolean;
  canClassify: boolean;
  onClassify: () => void;
}

const SOURCE_COLORS: Record<string, { bg: string; text: string }> = {
  bluesky: { bg: "#dff1ff", text: "#1185fe" },
  mastodon: { bg: "#e8e5ff", text: "#6364ff" },
  rss: { bg: "#fff3e0", text: "#e67e22" },
};

export default function PostCard({
  post,
  classifying,
  canClassify,
  onClassify,
}: PostCardProps) {
  const [expanded, setExpanded] = useState<boolean>(false);

  const sourceStyle = SOURCE_COLORS[post.source] || {
    bg: "#f0f0f0",
    text: "#666",
  };

  const isFake = post.classification?.label === "FAKE";
  const confidence = post.classification?.confidence ?? 0;

  // Truncate long text unless expanded
  const TRUNCATE_LEN = 300;
  const needsTruncation = post.text.length > TRUNCATE_LEN;
  const displayText =
    expanded || !needsTruncation
      ? post.text
      : post.text.slice(0, TRUNCATE_LEN) + "…";

  const formatDate = (iso?: string) => {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      return d.toLocaleString("uk-UA", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return iso;
    }
  };

  return (
    <div
      style={{
        background: "white",
        border: "1px solid #e0e0e0",
        borderRadius: 12,
        padding: 16,
        marginBottom: 12,
        transition: "box-shadow 0.15s",
      }}
    >
      {/* Header: source badge + author + date */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 10,
          flexWrap: "wrap",
          gap: 8,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span
            style={{
              padding: "3px 10px",
              borderRadius: 12,
              background: sourceStyle.bg,
              color: sourceStyle.text,
              fontSize: 11,
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.5px",
            }}
          >
            {post.source}
          </span>
          {post.author && (
            <span style={{ fontSize: 13, fontWeight: 600, color: "#333" }}>
              {post.author_handle || post.author}
            </span>
          )}
          {post.created_at && (
            <span style={{ fontSize: 12, color: "#999" }}>
              {formatDate(post.created_at)}
            </span>
          )}
        </div>

        {post.url && (
          <a
            href={post.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{
              fontSize: 12,
              color: "#4a90d9",
              textDecoration: "none",
            }}
          >
            Переглянути →
          </a>
        )}
      </div>

      {/* Title (if present — RSS articles have titles) */}
      {post.title && (
        <h3
          style={{
            fontSize: 15,
            fontWeight: 600,
            color: "#222",
            marginBottom: 8,
            lineHeight: 1.4,
          }}
        >
          {post.title}
        </h3>
      )}

      {/* Text body */}
      <div
        style={{
          fontSize: 14,
          lineHeight: 1.5,
          color: "#444",
          marginBottom: 12,
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {displayText}
      </div>

      {needsTruncation && (
        <button
          onClick={() => setExpanded(!expanded)}
          style={{
            background: "none",
            border: "none",
            color: "#4a90d9",
            cursor: "pointer",
            fontSize: 12,
            padding: 0,
            marginBottom: 10,
          }}
        >
          {expanded ? "Згорнути" : "Показати повністю"}
        </button>
      )}

      {/* Metadata row (likes, reposts, replies) */}
      {(post.likes_count != null ||
        post.reposts_count != null ||
        post.replies_count != null) && (
          <div
            style={{
              display: "flex",
              gap: 16,
              fontSize: 12,
              color: "#888",
              marginBottom: 10,
              paddingBottom: 10,
              borderBottom: "1px solid #f0f0f0",
            }}
          >
            {post.likes_count != null && (
              <span>♥ {post.likes_count.toLocaleString()}</span>
            )}
            {post.reposts_count != null && (
              <span>⟲ {post.reposts_count.toLocaleString()}</span>
            )}
            {post.replies_count != null && (
              <span>💬 {post.replies_count.toLocaleString()}</span>
            )}
          </div>
        )}

      {/* Footer: classify button OR result */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 10,
        }}
      >
        {!post.classification ? (
          <button
            onClick={onClassify}
            disabled={classifying || !canClassify}
            style={{
              padding: "8px 18px",
              fontSize: 13,
              background: classifying ? "#ccc" : "#4a90d9",
              color: "white",
              border: "none",
              borderRadius: 8,
              cursor: classifying || !canClassify ? "not-allowed" : "pointer",
              fontWeight: 500,
            }}
          >
            {classifying ? "Аналіз..." : "Класифікувати"}
          </button>
        ) : (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              flex: 1,
            }}
          >
            <div
              style={{
                padding: "6px 14px",
                borderRadius: 8,
                background: isFake
                  ? confidence > 0.7
                    ? "#ffe0e0"
                    : "#fff4e0"
                  : confidence > 0.7
                    ? "#e0f7e0"
                    : "#fff4e0",
                color: isFake
                  ? confidence > 0.7
                    ? "#dc3545"
                    : "#e67e22"
                  : confidence > 0.7
                    ? "#28a745"
                    : "#e67e22",
                fontWeight: 600,
                fontSize: 13,
              }}
            >
              {isFake ? "⚠ ДЕЗІНФОРМАЦІЯ" : "✓ ДОСТОВІРНО"}
            </div>
            <div
              style={{ fontSize: 12, color: "#666" }}
            >
              Впевненість:{" "}
              <strong>{(confidence * 100).toFixed(1)}%</strong>
            </div>
            <button
              onClick={onClassify}
              disabled={classifying}
              style={{
                marginLeft: "auto",
                padding: "4px 10px",
                fontSize: 11,
                background: "none",
                border: "1px solid #ddd",
                borderRadius: 6,
                color: "#666",
                cursor: "pointer",
              }}
              title="Повторно класифікувати"
            >
              {classifying ? "..." : "↻"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
