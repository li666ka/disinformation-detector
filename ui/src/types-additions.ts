// ────────────────────────────────────────────────────────────────────────────
// ДОДАТИ ЦІ ТИПИ В КІНЕЦЬ ФАЙЛУ ui/src/types.ts
// ────────────────────────────────────────────────────────────────────────────

export type SourceType = "bluesky" | "mastodon" | "rss";

/**
 * Нормалізований формат поста/статті з будь-якого джерела.
 * Бекенд повертає саме цю структуру незалежно від оригінального джерела.
 */
export interface NewsItem {
  /** Унікальний ID (source + native_id), напр. "bluesky:at://did:plc:xyz/post/abc" */
  id: string;
  source: SourceType;

  /** URL оригінального поста — для переходу "Переглянути" */
  url: string;

  /** Заголовок (зазвичай є тільки у RSS-статей, у соцмережах null) */
  title: string | null;

  /** Текст посту (обов'язково) */
  text: string;

  /** Автор — display name */
  author: string | null;

  /** Автор — handle/username (@alice.bsky.social, @user@mastodon.social) */
  author_handle: string | null;

  /** ISO timestamp */
  created_at: string | null;

  /** Метадані взаємодії (опціонально, залежно від джерела) */
  likes_count: number | null;
  reposts_count: number | null;
  replies_count: number | null;

  /** Мова (якщо визначена — "uk", "en", etc.) */
  language: string | null;
}

/**
 * NewsItem з опціональним результатом класифікації — для відображення в UI.
 */
export interface ClassifiedPost extends NewsItem {
  classification: {
    label: "FAKE" | "REAL" | "UNCERTAIN";
    confidence: number;
    probability: number | null; // null when UNCERTAIN
    reason?: string; // optional explanation from LLM
  } | null;
}

/**
 * Запит на пошук постів.
 * GET /sources/search?query=...&sources=bluesky,mastodon&limit=20
 */
export interface SourceSearchParams {
  query: string;
  sources: string; // comma-separated
  limit: number;
}

/**
 * Запит на отримання свіжих постів.
 * GET /sources/recent?sources=bluesky&limit=20
 */
export interface SourceRecentParams {
  sources: string;
  limit: number;
}

/**
 * Запит на отримання поста за URL.
 * POST /sources/fetch-url { url: string }
 */
export interface SourceFetchUrlRequest {
  url: string;
}

/**
 * Відповідь бекенду для пошуку/recent.
 */
export interface SourceSearchResponse {
  posts: NewsItem[];
  total: number;
  sources_used: SourceType[];
}
