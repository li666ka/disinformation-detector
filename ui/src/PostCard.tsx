import React, { useState } from "react";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import {
  AlertTriangle, CheckCircle2, HelpCircle, ExternalLink, Loader2,
  RefreshCw, Heart, Repeat2, MessageCircle, ShieldCheck, Users,
  Clock, Globe, Bot, Info,
} from "lucide-react";
import type { ClassifiedPost } from "./types";

interface PostCardProps {
  post: ClassifiedPost;
  classifying: boolean;
  canClassify: boolean;
  onClassify: () => void;
  onVerify?: () => void;
  onDetails?: () => void;
}

const SOURCE_STYLES: Record<string, string> = {
  bluesky: "bg-sky-100 text-sky-700 dark:bg-sky-950 dark:text-sky-300",
  mastodon: "bg-violet-100 text-violet-700 dark:bg-violet-950 dark:text-violet-300",
  rss: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
};

export default function PostCard({
  post,
  classifying,
  canClassify,
  onClassify,
  onVerify,
  onDetails,
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
            <Badge variant="secondary" className={cn("text-[10px] uppercase tracking-wider", SOURCE_STYLES[post.source])}>
              {post.source}
            </Badge>

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