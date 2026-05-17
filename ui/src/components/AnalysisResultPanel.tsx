import React from "react";
import { Badge } from "./ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Alert, AlertDescription } from "./ui/alert";
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Brain,
  Sparkles,
  ShieldCheck,
  ExternalLink,
  Clock,
  Network,
} from "lucide-react";
import { cn } from "../lib/utils";
import { ExplanationPanel, type ExplanationModelType } from "./ExplanationPanel";
import type {
  AnalyzeV2Aggregated,
  AnalyzeV2Classification,
  AnalyzeV2Extraction,
  AnalyzeV2ModelUsed,
  AnalyzeV2Response,
  AnalyzeV2SimilarPost,
  FactCheckResult,
  InferenceContext,
  NewsItem,
} from "../types";

const STANCE_LABEL: Record<string, string> = {
  supports: "стверджує",
  refutes: "спростовує",
  neutral: "нейтрально",
};

const STANCE_COLOR: Record<string, string> = {
  supports:
    "bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300",
  refutes:
    "bg-purple-100 text-purple-700 dark:bg-purple-950/40 dark:text-purple-300",
  neutral: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
};

const VERDICT_BADGE: Record<string, string> = {
  FAKE:
    "bg-red-100 text-red-800 border-red-300 dark:bg-red-950/40 dark:text-red-200 dark:border-red-900",
  REAL:
    "bg-green-100 text-green-800 border-green-300 dark:bg-green-950/40 dark:text-green-200 dark:border-green-900",
  UNCERTAIN:
    "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-950/40 dark:text-amber-200 dark:border-amber-900",
  UNKNOWN:
    "bg-gray-100 text-gray-800 border-gray-300 dark:bg-gray-900 dark:text-gray-200 dark:border-gray-700",
};

const VERDICT_BORDER: Record<string, string> = {
  FAKE: "border-red-300 dark:border-red-900",
  REAL: "border-green-300 dark:border-green-900",
  UNCERTAIN: "border-amber-300 dark:border-amber-900",
  UNKNOWN: "border-gray-300 dark:border-gray-700",
};

export function AnalysisResultPanel({ result }: { result: AnalyzeV2Response }) {
  return (
    <div className="space-y-4">
      {result.warnings.length > 0 && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <ul className="text-sm space-y-1 mt-1">
              {result.warnings.map((w, i) => (
                <li key={i}>• {w}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {result.fetched_post && <FetchedPostCard post={result.fetched_post} />}

      {result.extraction && <ExtractionCard extraction={result.extraction} />}

      {result.classification && (
        <ClassificationCard
          classification={result.classification}
          modelUsed={result.model_used ?? null}
          classifiedText={result.classified_text ?? null}
          originalText={result.original_text}
        />
      )}

      {result.classification?.explanation && (
        <ExplanationPanel
          modelType={_resolveExplanationModelType(
            result.model_used?.type,
            result.classification.explanation,
          )}
          explanation={result.classification.explanation}
          originalText={result.classified_text || result.original_text}
        />
      )}

      {result.fact_check && <FactCheckCard factCheck={result.fact_check} />}

      {result.inference_context && (
        <InferenceContextCard context={result.inference_context} />
      )}

      {result.aggregated && (
        <AggregatedSpreadCard
          aggregated={result.aggregated}
          similarPosts={result.similar_posts || []}
        />
      )}

      {Object.keys(result.timing_ms).length > 0 && (
        <details className="text-xs text-muted-foreground">
          <summary className="cursor-pointer">Час виконання</summary>
          <div className="mt-1 space-y-0.5 pl-2 border-l-2 border-muted">
            {Object.entries(result.timing_ms).map(([key, ms]) => (
              <div key={key}>
                <span className="font-mono">{key}</span>: {ms} ms
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function FetchedPostCard({ post }: { post: NewsItem }) {
  const handle = (post as any).author_handle || post.author;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base flex items-center justify-between gap-2 flex-wrap">
          <span className="flex items-center gap-2">
            <Badge variant="outline">{post.source}</Badge>
            <span className="text-sm">{handle}</span>
          </span>
          {post.url && (
            <a
              href={post.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-blue-600 hover:underline inline-flex items-center gap-1"
            >
              Відкрити <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm whitespace-pre-wrap">{post.text}</p>
        <div className="mt-3 flex flex-wrap gap-3 text-xs text-muted-foreground">
          {post.likes_count != null && <span>♥ {post.likes_count}</span>}
          {post.reposts_count != null && <span>🔁 {post.reposts_count}</span>}
          {post.replies_count != null && <span>💬 {post.replies_count}</span>}
          {post.created_at && (
            <span className="inline-flex items-center gap-1">
              <Clock className="h-3 w-3" />
              {new Date(post.created_at).toLocaleDateString("uk-UA")}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function ExtractionCard({ extraction }: { extraction: AnalyzeV2Extraction }) {
  if (!extraction.claims || extraction.claims.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <Brain className="h-4 w-4 text-blue-600" />
            LLM Extraction
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Alert>
            <HelpCircle className="h-4 w-4" />
            <AlertDescription className="text-sm">
              LLM не знайшов перевіряємих claims. Текст, ймовірно, містить лише
              думку, емоцію, або питання без конкретного твердження.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Brain className="h-4 w-4 text-blue-600" />
          LLM Extraction
          <Badge variant="outline" className="text-[10px] ml-1">
            {extraction.method === "llm" ? "Claude" : extraction.method}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {extraction.claims.map((c, idx) => (
          <div
            key={idx}
            className="space-y-1.5 pl-2 border-l-2 border-blue-300 dark:border-blue-700"
          >
            <div className="text-sm font-medium leading-snug">
              {idx + 1}. «{c.claim}»
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <Badge
                className={cn(
                  "text-[10px] h-5",
                  STANCE_COLOR[c.stance] || STANCE_COLOR.neutral,
                )}
              >
                {STANCE_LABEL[c.stance] || c.stance}
              </Badge>
              <span className="text-xs text-muted-foreground">→</span>
              <Badge variant="outline" className="text-[10px] h-5">
                автор вважає{" "}
                {c.author_verdict === "REAL"
                  ? "ПРАВДОЮ"
                  : c.author_verdict === "FAKE"
                  ? "ФЕЙКОМ"
                  : "ЗМІШАНО"}
              </Badge>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function ClassificationCard({
  classification,
  modelUsed,
  classifiedText,
  originalText,
}: {
  classification: AnalyzeV2Classification;
  modelUsed: AnalyzeV2ModelUsed | null;
  classifiedText: string | null;
  originalText: string;
}) {
  const label = classification.label;
  const Icon =
    label === "FAKE" ? AlertTriangle : label === "REAL" ? CheckCircle2 : HelpCircle;
  const labelUk =
    label === "FAKE"
      ? "ДЕЗІНФОРМАЦІЯ"
      : label === "REAL"
      ? "ДОСТОВІРНО"
      : "НЕВПЕВНЕНО";

  return (
    <Card className={cn("border-2", VERDICT_BORDER[label])}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Sparkles className="h-4 w-4" />
          Класифікація
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <Badge className={cn(VERDICT_BADGE[label], "text-sm px-3 py-1")}>
            <Icon className="h-3.5 w-3.5 mr-1.5 inline" />
            {labelUk}
          </Badge>
          <span className="text-sm">
            Впевненість:{" "}
            <strong>{(classification.confidence * 100).toFixed(1)}%</strong>
          </span>
        </div>

        {modelUsed && (
          <div className="text-xs text-muted-foreground">
            Модель: <strong>{modelUsed.name}</strong> ({modelUsed.type}
            {modelUsed.f1_score != null &&
              `, F1=${modelUsed.f1_score.toFixed(3)}`}
            )
          </div>
        )}

        {classifiedText && classifiedText !== originalText && (
          <details className="text-xs text-muted-foreground">
            <summary className="cursor-pointer hover:text-foreground">
              Класифіковано extracted claim замість raw text
            </summary>
            <div className="mt-1 pl-2 border-l-2 border-muted space-y-1">
              <p>
                Raw: <em>«{originalText.slice(0, 100)}…»</em>
              </p>
              <p>
                Used: <strong>«{classifiedText}»</strong>
              </p>
            </div>
          </details>
        )}

        {classification.reason && (
          <div className="text-sm border-t pt-2">
            <span className="text-xs text-muted-foreground">Обґрунтування:</span>
            <p className="mt-1">{classification.reason}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function FactCheckCard({ factCheck }: { factCheck: FactCheckResult }) {
  const statusColor: Record<string, string> = {
    MATCH: "bg-green-100 text-green-800 dark:bg-green-950/40 dark:text-green-200",
    MISMATCH: "bg-red-100 text-red-800 dark:bg-red-950/40 dark:text-red-200",
    MIXED: "bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-200",
    NO_DATA: "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200",
    NO_MODEL: "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200",
  };

  const status =
    (factCheck as any).comparison_status ||
    (factCheck.fact_check_found ? "MATCH" : "NO_DATA");

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <ShieldCheck className="h-4 w-4 text-amber-600" />
          Fact-check (Google)
          <Badge className={cn("text-xs ml-1", statusColor[status] || statusColor.NO_DATA)}>
            {status}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {factCheck.fact_check_found ? (
          <>
            {factCheck.verdict && (
              <p className="text-sm">
                <strong>{factCheck.publisher || "Fact-checker"}:</strong>{" "}
                {factCheck.verdict}
                {factCheck.url && (
                  <a
                    href={factCheck.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="ml-1 text-blue-600 hover:underline"
                  >
                    [link]
                  </a>
                )}
              </p>
            )}
            <div className="text-xs text-muted-foreground">
              Перевірено {(factCheck as any).claims_found ?? 0} із{" "}
              {(factCheck as any).claims_total ?? 0} claims
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground italic">
            Незалежних fact-checks не знайдено
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function AggregatedSpreadCard({
  aggregated,
  similarPosts,
}: {
  aggregated: AnalyzeV2Aggregated;
  similarPosts: AnalyzeV2SimilarPost[];
}) {
  const total = aggregated.total_posts;
  if (total === 0) {
    return (
      <Card>
        <CardContent className="pt-6">
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Не знайдено схожих постів у соцмережах. Спробуйте інші ключові слова.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const pct = (n: number) => (total > 0 ? Math.round((n / total) * 100) : 0);
  const sd = aggregated.stance_distribution;
  const cd = aggregated.classification_distribution;

  return (
    <Card className={cn("border-2", VERDICT_BORDER[aggregated.majority_verdict])}>
      <CardHeader>
        <CardTitle className="text-base">Поширення твердження</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-sm">
          Знайдено <strong>{total}</strong> пов'язаних постів
        </div>

        <div className="space-y-2">
          <div className="text-xs text-muted-foreground font-medium">
            Stance автора:
          </div>
          <BarRow label="Поширюють" count={sd.supports} pct={pct(sd.supports)} color="blue" />
          <BarRow label="Спростовують" count={sd.refutes} pct={pct(sd.refutes)} color="purple" />
          <BarRow label="Нейтрально" count={sd.neutral} pct={pct(sd.neutral)} color="gray" />
        </div>

        <div className="space-y-2 pt-2 border-t">
          <div className="text-xs text-muted-foreground font-medium">
            Класифікація моделі:
          </div>
          <BarRow label="FAKE" count={cd.FAKE} pct={pct(cd.FAKE)} color="red" />
          <BarRow label="REAL" count={cd.REAL} pct={pct(cd.REAL)} color="green" />
          {cd.UNCERTAIN > 0 && (
            <BarRow label="UNCERTAIN" count={cd.UNCERTAIN} pct={pct(cd.UNCERTAIN)} color="amber" />
          )}
        </div>

        <div className="pt-2 border-t space-y-2">
          <div className="flex flex-wrap items-center gap-3">
            <Badge
              className={cn(VERDICT_BADGE[aggregated.majority_verdict], "text-sm px-3 py-1")}
            >
              ВЕРДИКТ: {aggregated.majority_verdict}
            </Badge>
            <span className="text-xs text-muted-foreground">
              consensus {(aggregated.consensus_strength * 100).toFixed(0)}%, avg conf{" "}
              {(aggregated.majority_confidence * 100).toFixed(0)}%
            </span>
          </div>

          {aggregated.spread_warning && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription className="text-sm">
                {aggregated.spread_warning}
              </AlertDescription>
            </Alert>
          )}
        </div>

        <details className="text-sm">
          <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
            Показати всі {total} постів
          </summary>
          <div className="mt-2 space-y-2 pl-2 border-l-2 border-muted">
            {similarPosts.map((sp, i) => (
              <SimilarPostMini key={i} item={sp} idx={i + 1} />
            ))}
          </div>
        </details>
      </CardContent>
    </Card>
  );
}

function BarRow({
  label,
  count,
  pct,
  color,
}: {
  label: string;
  count: number;
  pct: number;
  color: string;
}) {
  const colorMap: Record<string, string> = {
    blue: "bg-blue-400 dark:bg-blue-600",
    purple: "bg-purple-400 dark:bg-purple-600",
    gray: "bg-gray-400 dark:bg-gray-600",
    red: "bg-red-400 dark:bg-red-600",
    green: "bg-green-400 dark:bg-green-600",
    amber: "bg-amber-400 dark:bg-amber-600",
  };

  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-24 shrink-0 text-muted-foreground">{label}</span>
      <div className="flex-1 h-4 bg-muted/40 rounded overflow-hidden">
        <div
          className={cn("h-full transition-[width]", colorMap[color])}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-20 text-right text-muted-foreground tabular-nums">
        {count} ({pct}%)
      </span>
    </div>
  );
}

function SimilarPostMini({
  item,
  idx,
}: {
  item: AnalyzeV2SimilarPost;
  idx: number;
}) {
  const post = item.post;
  const cls = item.classification;
  const verdict = cls?.label || "UNCERTAIN";
  const handle = (post as any)?.author_handle || post?.author || "—";

  return (
    <div className="text-xs space-y-1 py-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-muted-foreground">{idx}.</span>
        <Badge variant="outline" className="text-[9px] h-4 px-1">
          {post?.source}
        </Badge>
        <span className="text-muted-foreground">{handle}</span>
        {cls && (
          <Badge
            className={cn(
              "text-[9px] h-4 px-1.5",
              VERDICT_BADGE[verdict] || VERDICT_BADGE.UNKNOWN,
            )}
          >
            {verdict} {(cls.confidence * 100).toFixed(0)}%
          </Badge>
        )}
      </div>
      <p className="text-foreground line-clamp-2 leading-snug">{post?.text}</p>
    </div>
  );
}


// Резолвимо `ExplanationModelType` за model_type з ModelRecord. Якщо це
// поле відсутнє — пробуємо вгадати за shape самого explanation (saliency
// над токенами → distilbert/nb; graph nodes → gin/sage; reasoning → llm).
function _resolveExplanationModelType(
  modelType: string | undefined,
  explanation: any,
): ExplanationModelType {
  const known = ["nb", "distilbert", "gin", "sage", "llm"] as const;
  if (modelType && (known as readonly string[]).includes(modelType)) {
    return modelType as ExplanationModelType;
  }
  if (Array.isArray(explanation?.tokens)) {
    return explanation.method === "log_odds" ? "nb" : "distilbert";
  }
  if (Array.isArray(explanation?.important_nodes)) return "gin";
  return "llm";
}


// ── InferenceContextBuilder (Phase 1) ────────────────────────────────────

function InferenceContextCard({ context }: { context: InferenceContext }) {
  const meta = context.metadata || {};
  const warnings = meta.warnings || [];
  const fewPosts = warnings.find((w) => w.startsWith("few_posts_found"));
  const placeholder = warnings.includes("graph_construction_phase1_placeholder");
  const aggregates = context.aggregates || {};
  const nPosts = meta.n_posts_found ?? 0;

  // Phase 2: propagation_stats може бути або вкладеним у context.graph_data.metadata,
  // або плоским полем (FastAPI GIN branch шле обидва шляхи).
  const propStats: any = (context as any).propagation_stats
    ?? ((context.graph_data as any)?.metadata as any)
    ?? null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Network className="h-4 w-4 text-indigo-600" />
          Inference context
          {meta.sources_used && meta.sources_used.length > 0 && (
            <Badge variant="outline" className="text-[10px] ml-1">
              {meta.sources_used.join(" · ")}
            </Badge>
          )}
          {meta.build_time_ms != null && (
            <span className="text-xs text-muted-foreground ml-auto font-mono">
              {meta.build_time_ms} ms
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {context.claim && (
          <div>
            <div className="text-xs text-muted-foreground">Витягнутий claim</div>
            <p className="text-sm font-medium leading-snug">«{context.claim}»</p>
          </div>
        )}

        <div className="text-sm">
          Пов'язаних постів: <strong>{nPosts}</strong>
        </div>

        {Object.keys(aggregates).length > 0 && (
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-1.5">
              Social aggregates:
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-1 text-xs">
              {Object.entries(aggregates).map(([k, v]) => (
                <div
                  key={k}
                  className="flex justify-between border rounded px-2 py-1 bg-muted/30"
                >
                  <span className="font-mono truncate">{k}</span>
                  <span className="font-mono tabular-nums text-muted-foreground">
                    {Number(v).toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {propStats && (propStats.n_tweets || propStats.n_replies || propStats.n_retweets) && (
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-1.5">
              Граф поширення:
            </div>
            <ul className="text-xs space-y-0.5 pl-3 list-disc">
              <li>1 центральна стаття</li>
              <li><strong>{propStats.n_tweets ?? 0}</strong> твітів</li>
              <li>
                <strong>{propStats.n_retweets ?? 0}</strong> retweet-вузлів
                {propStats.synthetic_retweets ? (
                  <span className="text-muted-foreground">
                    {" "}({propStats.synthetic_retweets} синтетичних)
                  </span>
                ) : null}
              </li>
              <li><strong>{propStats.n_replies ?? 0}</strong> відповідей</li>
              {propStats.platforms && propStats.platforms.length > 0 && (
                <li className="text-muted-foreground">
                  Платформи: {propStats.platforms.join(", ")}
                </li>
              )}
            </ul>
          </div>
        )}

        {fewPosts && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="text-sm">
              Знайдено мало релевантних постів. Передбачення моделі з
              social/graph features матиме низьку надійність — потрібен
              ширший контекст у соцмережах.
            </AlertDescription>
          </Alert>
        )}

        {placeholder && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription className="text-sm">
              GIN/SAGE інтеграція — Phase 1 placeholder. Граф для inference
              ще не будується; модель отримує текст без cascade structure.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}
