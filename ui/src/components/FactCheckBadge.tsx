import React from "react";
import { Badge } from "./ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import {
  CheckCircle2,
  XCircle,
  HelpCircle,
  AlertCircle,
  Loader2,
  ExternalLink,
} from "lucide-react";
import { cn } from "../lib/utils";
import type { FactCheckResult } from "../types";

interface Props {
  factCheck?: FactCheckResult;
  loading?: boolean;
  compact?: boolean;
}

type StatusConfig = {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  className: string;
  tooltip: string;
};

export function FactCheckBadge({ factCheck, loading, compact = false }: Props) {
  if (loading) {
    return (
      <Badge variant="outline" className="gap-1">
        <Loader2 className="h-3 w-3 animate-spin" />
        Перевірка fact-check...
      </Badge>
    );
  }

  if (!factCheck) return null;

  const { comparison_status, verdict, publisher, url, error } = factCheck;

  const configs: Record<string, StatusConfig> = {
    MATCH: {
      icon: CheckCircle2,
      label: "Збігається",
      className:
        "border-green-500 bg-green-50 text-green-700 dark:bg-green-950/30 dark:text-green-400",
      tooltip: "Класифікація моделі збігається з fact-checker",
    },
    MISMATCH: {
      icon: XCircle,
      label: "Не збігається",
      className:
        "border-red-500 bg-red-50 text-red-700 dark:bg-red-950/30 dark:text-red-400",
      tooltip: "Модель і fact-checker дають різні відповіді",
    },
    MIXED: {
      icon: AlertCircle,
      label: "Змішано",
      className:
        "border-amber-500 bg-amber-50 text-amber-700 dark:bg-amber-950/30 dark:text-amber-400",
      tooltip: "Fact-checker дав mixed verdict (half-true, partly false)",
    },
    NO_DATA: {
      icon: HelpCircle,
      label: "Нема fact-check",
      className:
        "border-gray-300 bg-gray-50 text-gray-600 dark:bg-gray-900/30 dark:text-gray-400",
      tooltip: "Fact-check для цього посту не знайдено",
    },
    NO_MODEL: {
      icon: HelpCircle,
      label: "Нема моделі",
      className: "border-gray-300",
      tooltip: "Класифікації моделі немає для порівняння",
    },
  };

  const config: StatusConfig =
    configs[comparison_status] ?? {
      icon: HelpCircle,
      label: comparison_status,
      className: "",
      tooltip: error || "Невідомий статус",
    };

  const Icon = config.icon;

  const badge = (
    <Badge variant="outline" className={cn("gap-1", config.className)}>
      <Icon className="h-3 w-3" />
      {!compact && config.label}
    </Badge>
  );

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span>{badge}</span>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-md space-y-2">
          <p className="font-medium">{config.tooltip}</p>

          {(factCheck.claims_total ?? 0) > 0 && (
            <div className="text-xs text-muted-foreground border-b pb-1.5">
              Витягнуто {factCheck.claims_total} claim
              {(factCheck.claims_total ?? 0) === 1 ? "" : "s"}{" "}
              ({factCheck.extraction_method === "llm" ? "LLM" : "auto"}),{" "}
              {factCheck.claims_found ?? 0} з fact-check
            </div>
          )}

          {factCheck.claims_results && factCheck.claims_results.length > 0 ? (
            factCheck.claims_results.map((cr, idx) => {
              const stanceIcon =
                cr.stance === "supports" ? "↑" :
                cr.stance === "refutes" ? "↓" : "→";
              const stanceLabel =
                cr.stance === "supports" ? "автор стверджує" :
                cr.stance === "refutes" ? "автор спростовує" : "автор згадує";

              return (
                <div
                  key={idx}
                  className="border-l-2 border-muted pl-2 space-y-1"
                >
                  <p className="text-xs italic">"{cr.claim}"</p>
                  <p className="text-xs text-muted-foreground">
                    {stanceIcon} {stanceLabel}
                  </p>

                  {cr.found ? (
                    <div className="text-xs space-y-0.5">
                      {cr.claim_text_matched && cr.claim_text_matched !== cr.claim && (
                        <p className="text-muted-foreground">
                          Знайдено: <span className="italic">"{cr.claim_text_matched}"</span>
                        </p>
                      )}
                      <div>
                        <span className="text-muted-foreground">Fact-check:</span>{" "}
                        <span
                          className={cn(
                            "font-medium",
                            cr.verdict_normalized === "FAKE" && "text-red-600",
                            cr.verdict_normalized === "REAL" && "text-green-600",
                            cr.verdict_normalized === "MIXED" && "text-amber-600",
                          )}
                        >
                          {cr.verdict || cr.verdict_normalized}
                        </span>{" "}
                        <span className="text-muted-foreground">
                          ({cr.publisher})
                        </span>
                      </div>

                      {cr.stance !== "neutral" && (
                        <div>
                          <span className="text-muted-foreground">Позиція автора:</span>{" "}
                          <span
                            className={cn(
                              "font-medium",
                              cr.effective_author_verdict === "FAKE" && "text-red-600",
                              cr.effective_author_verdict === "REAL" && "text-green-600",
                              cr.effective_author_verdict === "MIXED" && "text-amber-600",
                            )}
                          >
                            {cr.effective_author_verdict === "FAKE" && "spreading misinfo"}
                            {cr.effective_author_verdict === "REAL" && "factually correct"}
                            {cr.effective_author_verdict === "MIXED" && "mixed"}
                            {cr.effective_author_verdict === "UNKNOWN" && "unknown"}
                          </span>
                        </div>
                      )}

                      {cr.url && (
                        <a
                          href={cr.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-500 hover:underline inline-flex items-center gap-0.5"
                          onClick={(e) => e.stopPropagation()}
                        >
                          {cr.review_title || "View review"}
                          <ExternalLink className="h-2.5 w-2.5" />
                        </a>
                      )}
                    </div>
                  ) : (
                    <p className="text-xs text-muted-foreground">
                      Не знайдено fact-check
                    </p>
                  )}
                </div>
              );
            })
          ) : (
            <>
              {factCheck.claim_query_used && (
                <div className="border-t pt-1.5">
                  <p className="text-xs text-muted-foreground mb-0.5">
                    Запит у Google Fact Check:
                  </p>
                  <p className="text-xs italic bg-muted px-2 py-1 rounded">
                    "{factCheck.claim_query_used}"
                  </p>
                </div>
              )}

              {factCheck.fact_check_found && (
                <>
                  {factCheck.claim_text_matched && (
                    <div>
                      <p className="text-xs text-muted-foreground mb-0.5">
                        Знайдений claim:
                      </p>
                      <p className="text-xs italic bg-blue-50 dark:bg-blue-950/30 px-2 py-1 rounded">
                        "{factCheck.claim_text_matched}"
                      </p>
                    </div>
                  )}

                  <div className="border-t pt-1.5 space-y-1">
                    {publisher && (
                      <p className="text-xs">
                        <span className="text-muted-foreground">Джерело:</span>{" "}
                        <span className="font-medium">{publisher}</span>
                      </p>
                    )}
                    {verdict && (
                      <p className="text-xs">
                        <span className="text-muted-foreground">Verdict:</span>{" "}
                        <span className="font-medium">{verdict}</span>
                      </p>
                    )}
                    {factCheck.review_date && (
                      <p className="text-xs text-muted-foreground">
                        {factCheck.review_date.split("T")[0]}
                      </p>
                    )}
                    {url && (
                      <a
                        href={url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-blue-500 hover:underline inline-flex items-center gap-0.5"
                        onClick={(e) => e.stopPropagation()}
                      >
                        Відкрити повний review{" "}
                        <ExternalLink className="h-2.5 w-2.5" />
                      </a>
                    )}
                  </div>
                </>
              )}

              {!factCheck.fact_check_found && (
                <div className="border-t pt-1.5">
                  <p className="text-xs text-muted-foreground">
                    {error ||
                      "Google Fact Check API не знайшов перевірок на цей claim. Це не означає що твердження невірне — просто ніхто з 300+ fact-checker організацій не публікував review саме на нього."}
                  </p>
                </div>
              )}
            </>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
