import React, { useState } from "react";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import {
  AlertTriangle, CheckCircle2, HelpCircle, ExternalLink, Loader2,
  RefreshCw, Heart, Repeat2, MessageCircle, ShieldCheck, Users,
  Clock, Globe, Bot, Info, Rss, Brain,
} from "lucide-react";
import type { ClassifiedPost } from "./types";
import { FactCheckBadge } from "./components/FactCheckBadge";

interface PostCardProps {
  post: ClassifiedPost;
  classifying: boolean;
  canClassify: boolean;
  onClassify: () => void;
  onVerify?: () => void;
  onDetails?: () => void;
  onExtract?: () => void;
  extracting?: boolean;
}

const SOURCE_STYLES: Record<string, string> = {
  bluesky: "bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300",
  mastodon: "bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300",
  rss: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
};

const BlueskyIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 568 501" className={className} fill="currentColor">
    <path d="M123.121 33.664C188.241 82.553 258.281 181.68 284 234.873c25.719-53.192 95.759-152.32 160.879-201.21C491.866-1.611 568-28.906 568 57.947c0 17.346-9.945 145.713-15.778 166.555-20.275 72.453-94.155 90.933-159.875 79.748C507.222 323.8 536.444 388.56 473.333 453.32c-119.86 122.992-172.272-30.859-185.702-70.281-2.462-7.227-3.614-10.608-3.631-7.733-.017-2.875-1.169.506-3.631 7.733-13.43 39.422-65.842 193.273-185.702 70.281-63.111-64.76-33.889-129.52 80.986-149.071-65.72 11.185-139.6-7.295-159.875-79.748C9.945 203.659 0 75.291 0 57.946 0-28.906 76.135-1.612 123.121 33.664Z" />
  </svg>
);

const MastodonIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 448 512" className={className} fill="currentColor">
    <path d="M433 179.11c0-97.2-63.71-125.7-63.71-125.7-62.52-28.7-228.56-28.4-290.48 0 0 0-63.72 28.5-63.72 125.7 0 115.7-6.6 259.4 105.63 289.1 40.51 10.7 75.32 13 103.33 11.4 50.81-2.8 79.32-18.1 79.32-18.1l-1.7-36.9s-36.31 11.4-77.12 10.1c-40.41-1.4-83.13-4.4-89.63-54a102.54 102.54 0 0 1-.9-13.9c85.63 20.9 158.65 9.1 178.75 6.7 56.12-6.7 105-41.3 111.23-72.9 9.8-49.8 9-121.5 9-121.5zm-75.12 125.2h-46.63v-114.2c0-49.7-64-51.6-64 6.9v62.5h-46.33V197c0-58.5-64-56.6-64-6.9v114.2H90.19c0-122.1-5.2-147.9 18.41-175 25.9-28.9 79.82-30.8 103.83 6.1l11.6 19.5 11.6-19.5c24.11-37.1 78.12-34.8 103.83-6.1 23.71 27.3 18.4 53 18.4 175z" />
  </svg>
);

function SourceIcon({ source }: { source: string }) {
  switch (source) {
    case "bluesky":
      return <BlueskyIcon className="h-3 w-3 shrink-0" />;
    case "mastodon":
      return <MastodonIcon className="h-3 w-3 shrink-0" />;
    case "rss":
      return <Rss className="h-3 w-3 shrink-0" />;
    default:
      return <Globe className="h-3 w-3 shrink-0" />;
  }
}

export default function PostCard({
  post,
  classifying,
  canClassify,
  onClassify,
  onVerify,
  onDetails,
  onExtract,
  extracting,
}: PostCardProps) {
  const [expanded, setExpanded] = useState<boolean>(false);

  const classification = post.classification;

  const TRUNCATE_LEN = 300;
  const needsTruncation = post.text.length > TRUNCATE_LEN;
  const displayText = expanded || !needsTruncation
    ? post.text
    : post.text.slice(0, TRUNCATE_LEN) + "...";

  const formatDate = (iso?: string) => {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      return d.toLocaleString("uk-UA", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
    } catch { return iso; }
  };

  // Detect if author is "trusted" for visual emphasis
  const hasTrustSignals =
    post.author_is_verified || post.author_has_custom_domain || (post.author_followers_count ?? 0) > 10000;

  // Format account age nicely
  const formatAge = (days?: number | null) => {
    if (days == null) return "";
    if (days < 30) return `${days}д`;
    if (days < 365) return `${Math.round(days / 30)}міс.`;
    return `${Math.round(days / 365)}р`;
  };

  // Warning signals for the post author
  const hasYoungAuthor = post.author_account_age_days != null && post.author_account_age_days < 30;
  const hasLowFollowers =
    post.author_followers_count != null && post.author_followers_count < 50 && post.source !== "rss";

  return (
    <Card className={cn("mb-3", hasTrustSignals && "border-emerald-200 dark:border-emerald-900/50")}>
      <CardContent className="p-4">
        {/* Header: source + author row */}
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge variant="secondary" className={cn("text-[10px] uppercase tracking-wider flex items-center gap-1", SOURCE_STYLES[post.source])}>
              <SourceIcon source={post.source} />
              {post.source}
            </Badge>

            <RelevanceBadge score={post._relevance_score} />

            {post.author && (
              <span className="text-sm font-medium">
                {post.author_handle || post.author}
              </span>
            )}

            {/* Trust badges */}
            {post.author_is_verified && (
              <Badge className="bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-950/40 dark:text-blue-300 text-[10px]">
                <CheckCircle2 className="h-2.5 w-2.5 mr-0.5" />
                verified
              </Badge>
            )}
            {post.author_has_custom_domain && (
              <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/40 dark:text-emerald-300 text-[10px]">
                <Globe className="h-2.5 w-2.5 mr-0.5" />
                domain
              </Badge>
            )}
            {post.raw_metadata?.is_bot && (
              <Badge className="bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 text-[10px]">
                <Bot className="h-2.5 w-2.5 mr-0.5" />
                bot
              </Badge>
            )}

            {/* Warning badges */}
            {hasYoungAuthor && (
              <Badge className="bg-red-50 text-red-700 border-red-200 dark:bg-red-950/40 dark:text-red-300 text-[10px]">
                <AlertTriangle className="h-2.5 w-2.5 mr-0.5" />
                новий ({post.author_account_age_days}д)
              </Badge>
            )}

            {post.created_at && (
              <span className="text-xs text-muted-foreground">
                {formatDate(post.created_at)}
              </span>
            )}
          </div>

          {post.url && (
            <a href={post.url} target="_blank" rel="noopener noreferrer"
              className="text-xs text-primary hover:underline flex items-center gap-1">
              <ExternalLink className="h-3 w-3" />
              Переглянути
            </a>
          )}
        </div>

        {/* Author metadata row (if available) */}
        {(post.author_followers_count != null || post.author_account_age_days != null || post.author_posts_count != null) && (
          <div className="flex items-center gap-3 text-[11px] text-muted-foreground mb-2 flex-wrap">
            {post.author_followers_count != null && (
              <span className={cn("flex items-center gap-1", hasLowFollowers && "text-amber-700 dark:text-amber-400")}>
                <Users className="h-3 w-3" />
                {post.author_followers_count.toLocaleString()} фоловерів
                {hasLowFollowers && (
                  <span title="Мало фоловерів — можливий сигнал"> ⚠</span>
                )}
              </span>
            )}
            {post.author_account_age_days != null && (
              <span className={cn("flex items-center gap-1", hasYoungAuthor && "text-amber-700 dark:text-amber-400")}>
                <Clock className="h-3 w-3" />
                {formatAge(post.author_account_age_days)}
              </span>
            )}
            {post.author_posts_count != null && (
              <span>{post.author_posts_count.toLocaleString()} постів</span>
            )}
            {post.author_following_count != null && post.author_followers_count != null &&
              post.author_followers_count > 0 &&
              (post.author_following_count / post.author_followers_count) > 10 && (
                <span className="flex items-center gap-1 text-amber-700 dark:text-amber-400">
                  <AlertTriangle className="h-3 w-3" />
                  підозрілий ratio
                </span>
              )}
          </div>
        )}

        {/* Title */}
        {post.title && (
          <h3 className="text-sm font-semibold mb-2 leading-snug">{post.title}</h3>
        )}

        {/* Text */}
        <div className="text-sm leading-relaxed whitespace-pre-wrap break-words mb-2">
          {displayText}
        </div>

        {needsTruncation && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-primary hover:underline mb-2"
          >
            {expanded ? "Згорнути" : "Показати повністю"}
          </button>
        )}

        {/* Engagement row */}
        {(post.likes_count != null || post.reposts_count != null || post.replies_count != null) && (
          <div className="flex items-center gap-4 text-xs text-muted-foreground mb-3 pb-3 border-b">
            {post.likes_count != null && (
              <span className="flex items-center gap-1">
                <Heart className="h-3 w-3" />
                {post.likes_count.toLocaleString()}
              </span>
            )}
            {post.reposts_count != null && (
              <span className="flex items-center gap-1">
                <Repeat2 className="h-3 w-3" />
                {post.reposts_count.toLocaleString()}
              </span>
            )}
            {post.replies_count != null && (
              <span className="flex items-center gap-1">
                <MessageCircle className="h-3 w-3" />
                {post.replies_count.toLocaleString()}
              </span>
            )}
            {post.quote_count != null && post.quote_count > 0 && (
              <span className="flex items-center gap-1">
                <MessageCircle className="h-3 w-3" />
                {post.quote_count.toLocaleString()} quotes
              </span>
            )}
            {(post.labels && post.labels.length > 0) && (
              <span className="flex items-center gap-1 text-amber-700 dark:text-amber-400 ml-auto">
                <Info className="h-3 w-3" />
                {post.labels.join(", ")}
              </span>
            )}
          </div>
        )}

        {/* LLM Extraction — окремий етап перед класифікацією */}
        {onExtract && (
          <div className="mt-1">
            {!post.extraction || post.extraction.status === "idle" ? (
              <Button
                variant="outline"
                size="sm"
                onClick={onExtract}
                disabled={extracting}
                className="gap-1.5"
              >
                {extracting ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Brain className="h-3.5 w-3.5" />
                )}
                {extracting ? "Розпаковую…" : "Розпакувати claim"}
              </Button>
            ) : post.extraction.status === "loading" ? (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                LLM розпаковує claim…
              </div>
            ) : post.extraction.status === "error" ? (
              <div className="text-xs text-red-600 dark:text-red-400">
                Помилка extraction: {post.extraction.error}
              </div>
            ) : (
              <ExtractionBlock extraction={post.extraction} />
            )}
          </div>
        )}

        {/* Footer: action buttons + classification */}
        <div className="flex items-center gap-2 flex-wrap">
          {!classification ? (
            <Button
              size="sm"
              onClick={onClassify}
              disabled={classifying || !canClassify}
            >
              {classifying && <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />}
              {classifying ? "Аналіз..." : "Класифікувати"}
            </Button>
          ) : (
            <ClassificationBadge classification={classification} onReclassify={onClassify} classifying={classifying} />
          )}

          {classification && classification.label !== "UNCERTAIN" && (
            <FactCheckBadge
              factCheck={post.factCheck}
              loading={post.factCheckLoading}
            />
          )}

          {/* Details button (only for social posts, not RSS) */}
          {onDetails && post.source !== "rss" && (
            <Button variant="outline" size="sm" onClick={onDetails} className="gap-1.5">
              <Users className="h-3.5 w-3.5" />
              Деталі
            </Button>
          )}

          {/* Verify button (useful for RSS or news-like social posts) */}
          {onVerify && (
            <Button variant="outline" size="sm" onClick={onVerify} className="gap-1.5">
              <ShieldCheck className="h-3.5 w-3.5" />
              Verify
            </Button>
          )}
        </div>

        {post.factCheck && (
          <details className="text-xs text-muted-foreground mt-2">
            <summary className="cursor-pointer hover:text-foreground select-none">
              Fact-check деталі ({post.factCheck.claims_total ?? 0} claim
              {(post.factCheck.claims_total ?? 0) === 1 ? "" : "s"},{" "}
              {post.factCheck.claims_found ?? 0} перевірено)
            </summary>
            <div className="mt-1.5 space-y-2 pl-2 border-l-2 border-muted">
              <p className="text-[10px] text-muted-foreground italic">
                Метод витягування:{" "}
                {post.factCheck.extraction_method === "llm"
                  ? "LLM (Gemini)"
                  : "Простий (regex)"}
              </p>

              {post.factCheck.claims_results && post.factCheck.claims_results.length > 0 ? (
                post.factCheck.claims_results.map((cr, idx) => {
                  const stanceLabel =
                    cr.stance === "supports" ? "стверджує" :
                    cr.stance === "refutes" ? "спростовує" : "нейтрально";

                  return (
                    <div key={idx} className="space-y-0.5">
                      <div className="font-medium">
                        {idx + 1}. "{cr.claim}"
                      </div>
                      <div className="pl-3 space-y-0.5">
                        <div className="text-[10px] text-muted-foreground">
                          Позиція: {stanceLabel}
                        </div>
                        {cr.found ? (
                          <>
                            {cr.claim_text_matched && cr.claim_text_matched !== cr.claim && (
                              <div className="text-muted-foreground">
                                Знайдено:{" "}
                                <span className="italic">"{cr.claim_text_matched}"</span>
                              </div>
                            )}
                            <div>
                              Fact-check:{" "}
                              <span
                                className={cn(
                                  cr.verdict_normalized === "FAKE" && "text-red-600 dark:text-red-400",
                                  cr.verdict_normalized === "REAL" && "text-green-600 dark:text-green-400",
                                  cr.verdict_normalized === "MIXED" && "text-amber-600 dark:text-amber-400",
                                )}
                              >
                                {cr.verdict || cr.verdict_normalized}
                              </span>{" "}
                              ({cr.publisher})
                              {cr.url && (
                                <a
                                  href={cr.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-blue-500 ml-1 hover:underline"
                                >
                                  →
                                </a>
                              )}
                            </div>
                            {cr.stance !== "neutral" && (
                              <div>
                                Висновок: автор{" "}
                                <span
                                  className={cn(
                                    "font-medium",
                                    cr.effective_author_verdict === "FAKE" && "text-red-600 dark:text-red-400",
                                    cr.effective_author_verdict === "REAL" && "text-green-600 dark:text-green-400",
                                    cr.effective_author_verdict === "MIXED" && "text-amber-600 dark:text-amber-400",
                                  )}
                                >
                                  {cr.effective_author_verdict === "FAKE" && "spreading misinfo (FAKE)"}
                                  {cr.effective_author_verdict === "REAL" && "factually correct (REAL)"}
                                  {cr.effective_author_verdict === "MIXED" && "mixed/ambiguous"}
                                  {cr.effective_author_verdict === "UNKNOWN" && "unknown"}
                                </span>
                              </div>
                            )}
                          </>
                        ) : (
                          <div className="text-amber-600 dark:text-amber-400">
                            Не знайдено fact-check
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })
              ) : (
                <>
                  {post.factCheck.claim_query_used && (
                    <div>
                      <span className="font-medium">Запит:</span>{" "}
                      <span className="italic">"{post.factCheck.claim_query_used}"</span>
                    </div>
                  )}
                  {post.factCheck.fact_check_found ? (
                    <div>
                      <span className="font-medium">{post.factCheck.publisher}:</span>{" "}
                      {post.factCheck.verdict}
                      {post.factCheck.url && (
                        <a
                          href={post.factCheck.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-500 ml-1 hover:underline"
                        >
                          [link]
                        </a>
                      )}
                    </div>
                  ) : (
                    <div className="text-amber-600 dark:text-amber-400">
                      Перевірок не знайдено
                    </div>
                  )}
                </>
              )}
            </div>
          </details>
        )}
      </CardContent>
    </Card>
  );
}

// ── Classification badge ─────────────────────────────────────────────────

function ClassificationBadge({
  classification,
  onReclassify,
  classifying,
}: {
  classification: NonNullable<ClassifiedPost["classification"]>;
  onReclassify: () => void;
  classifying: boolean;
}) {
  const { label, confidence } = classification;

  if (label === "UNCERTAIN") {
    return (
      <div className="flex items-center gap-2 flex-1">
        <Badge className="bg-muted text-muted-foreground">
          <HelpCircle className="h-3 w-3 mr-1" />
          НЕВПЕВНЕНО
        </Badge>
        <span className="text-xs text-muted-foreground italic">
          Модель не змогла класифікувати
        </span>
        <Button variant="ghost" size="sm" onClick={onReclassify} disabled={classifying} className="ml-auto h-7 px-2">
          <RefreshCw className={cn("h-3 w-3", classifying && "animate-spin")} />
        </Button>
      </div>
    );
  }

  const isFake = label === "FAKE";
  const isHighConf = confidence > 0.7;

  const badgeClass = isFake
    ? (isHighConf ? "bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-300"
      : "bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300")
    : (isHighConf ? "bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-300"
      : "bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300");

  return (
    <div className="flex items-center gap-2 flex-1">
      <Badge className={badgeClass}>
        {isFake ? <AlertTriangle className="h-3 w-3 mr-1" /> : <CheckCircle2 className="h-3 w-3 mr-1" />}
        {isFake ? "ДЕЗІНФОРМАЦІЯ" : "ДОСТОВІРНО"}
      </Badge>
      <span className="text-xs text-muted-foreground">
        Впевненість: <strong>{(confidence * 100).toFixed(1)}%</strong>
      </span>
      <Button variant="ghost" size="sm" onClick={onReclassify} disabled={classifying} className="ml-auto h-7 px-2">
        <RefreshCw className={cn("h-3 w-3", classifying && "animate-spin")} />
      </Button>
    </div>
  );
}


// ── Extraction block (LLM розпакування claim+stance) ─────────────────────

const STANCE_LABEL_UA: Record<string, { label: string; color: string }> = {
  supports: {
    label: "стверджує",
    color: "bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300",
  },
  refutes: {
    label: "спростовує",
    color: "bg-purple-100 text-purple-700 dark:bg-purple-950/40 dark:text-purple-300",
  },
  neutral: {
    label: "нейтрально",
    color: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  },
};

const AUTHOR_VERDICT_LABEL_UA: Record<string, { label: string; color: string }> = {
  REAL: {
    label: "автор вважає ПРАВДОЮ",
    color: "bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-300",
  },
  FAKE: {
    label: "автор вважає ФЕЙКОМ",
    color: "bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-300",
  },
  MIXED: {
    label: "автор нейтральний",
    color: "bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300",
  },
};

function ExtractionBlock({
  extraction,
}: {
  extraction: NonNullable<ClassifiedPost["extraction"]>;
}) {
  if (extraction.status !== "done" || !extraction.claims) return null;

  if (extraction.claims.length === 0) {
    return (
      <div className="text-xs text-muted-foreground bg-muted/40 px-3 py-2 rounded-md border">
        <div className="flex items-center gap-1.5">
          <Brain className="h-3.5 w-3.5" />
          <span className="font-medium">LLM не знайшов перевіряємих claims</span>
          {extraction.method && (
            <Badge variant="outline" className="text-[10px] h-4 px-1.5 ml-auto">
              {extraction.method === "llm" ? "Claude" : "regex"}
            </Badge>
          )}
        </div>
        <p className="mt-1 italic">
          Пост, ймовірно, виражає особисту думку/емоцію без конкретного твердження.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2 bg-muted/40 px-3 py-2.5 rounded-md border">
      <div className="flex items-center gap-1.5 text-xs font-medium">
        <Brain className="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" />
        <span>LLM Extraction</span>
        <Badge variant="outline" className="text-[10px] h-4 px-1.5">
          {extraction.method === "llm" ? "Claude" : "regex"}
        </Badge>
        <span className="text-muted-foreground">·</span>
        <span className="text-muted-foreground">
          {extraction.claims.length} claim{extraction.claims.length !== 1 ? "s" : ""}
        </span>
      </div>

      {extraction.claims.map((c, idx) => {
        const stance = STANCE_LABEL_UA[c.stance] || STANCE_LABEL_UA.neutral;
        const author =
          AUTHOR_VERDICT_LABEL_UA[c.author_verdict] || AUTHOR_VERDICT_LABEL_UA.MIXED;
        return (
          <div
            key={idx}
            className="space-y-1.5 border-l-2 border-blue-300 dark:border-blue-700 pl-2"
          >
            <div className="text-sm font-medium leading-snug">
              {idx + 1}. «{c.claim}»
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <Badge className={cn("text-[10px] h-5", stance.color)}>
                {stance.label}
              </Badge>
              <Badge className={cn("text-[10px] h-5", author.color)}>
                {author.label}
              </Badge>
            </div>
          </div>
        );
      })}
    </div>
  );
}


// ── Relevance badge (token-overlap score from /sources/search) ───────────

function RelevanceBadge({ score }: { score?: number }) {
  if (score == null) return null;
  const pct = Math.round(score * 100);
  let cls: string;
  let label: string;
  if (score >= 0.7) {
    cls = "bg-green-100 text-green-700 border-green-200 dark:bg-green-950/40 dark:text-green-300";
    label = "Точний збіг";
  } else if (score >= 0.5) {
    cls = "bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-950/40 dark:text-blue-300";
    label = "Релевантне";
  } else if (score >= 0.3) {
    cls = "bg-amber-100 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300";
    label = "Можливо релевантне";
  } else {
    return null; // нижчий за поріг — взагалі не показуємо
  }
  return (
    <Badge
      className={cn("text-[10px] border", cls)}
      title={`Relevance: ${pct}%`}
    >
      {label} · {pct}%
    </Badge>
  );
}