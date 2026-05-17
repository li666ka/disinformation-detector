import React, { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Alert, AlertDescription } from "./ui/alert";
import {
  AlertTriangle,
  Download,
  Lightbulb,
  Microscope,
  Network,
  Sparkles,
} from "lucide-react";
import { cn } from "../lib/utils";
import type {
  Explanation,
  GnnExplanation,
  IgExplanation,
  LlmExplanation,
  NbExplanation,
  TokenAttribution,
} from "../types";

export type ExplanationModelType = "nb" | "distilbert" | "gin" | "sage" | "llm";

interface ExplanationPanelProps {
  modelType: ExplanationModelType;
  explanation: Explanation;
  originalText?: string;
}

// ── Type guards (структура response від трьох різних backend) ────────────

function isTokenExplanation(e: Explanation): e is NbExplanation | IgExplanation {
  return Array.isArray((e as any).tokens);
}
function isGnnExplanation(e: Explanation): e is GnnExplanation {
  return Array.isArray((e as any).important_nodes);
}
function isLlmExplanation(e: Explanation): e is LlmExplanation {
  const ll = e as any;
  return typeof ll.reasoning === "string"
    || Array.isArray(ll.key_indicators)
    || Array.isArray(ll.confidence_factors);
}

// ── Public component ────────────────────────────────────────────────────

export function ExplanationPanel({
  modelType,
  explanation,
  originalText,
}: ExplanationPanelProps) {
  const body = useMemo(() => {
    if (isTokenExplanation(explanation)) {
      return (
        <TokenAttributionView
          explanation={explanation}
          originalText={originalText}
        />
      );
    }
    if (isGnnExplanation(explanation)) {
      return <GraphSubgraphView explanation={explanation} />;
    }
    if (isLlmExplanation(explanation)) {
      return <LlmReasoningView explanation={explanation} />;
    }
    return (
      <div className="text-sm text-muted-foreground italic">
        No explanation available for model type «{modelType}».
      </div>
    );
  }, [explanation, originalText, modelType]);

  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(explanation, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `explanation_${modelType}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const method = (explanation as any).method ?? "—";

  return (
    <Card>
      <CardHeader className="pb-3 flex flex-row items-center justify-between gap-2">
        <CardTitle className="text-base flex items-center gap-2">
          <Microscope className="h-4 w-4 text-violet-600" />
          Пояснення
          <Badge variant="outline" className="text-[10px] ml-1">
            {modelType} · {method}
          </Badge>
        </CardTitle>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleDownload}
          className="h-7 px-2 text-xs"
          title="Зберегти explanation як JSON"
        >
          <Download className="h-3.5 w-3.5 mr-1" /> JSON
        </Button>
      </CardHeader>
      <CardContent className="space-y-3">
        {body}
        <ExplanationMeta explanation={explanation} />
      </CardContent>
    </Card>
  );
}

// ── Token attribution (NB, DistilBERT) ───────────────────────────────────

function TokenAttributionView({
  explanation,
  originalText,
}: {
  explanation: NbExplanation | IgExplanation;
  originalText?: string;
}) {
  const allTokens = explanation.tokens || [];
  const [showAll, setShowAll] = useState(false);
  const visibleTokens = showAll ? allTokens : allTokens.slice(0, 10);

  // У bar chart показуємо до 10 — інакше нечитабельно
  const chartData = useMemo(
    () =>
      allTokens.slice(0, 10).map((t) => ({
        token: t.token,
        attribution: t.attribution,
        fillKey: t.attribution >= 0 ? "fake" : "real",
      })),
    [allTokens],
  );

  const direction =
    "prediction" in explanation
      ? explanation.prediction
      : (explanation as IgExplanation).predicted_label;

  return (
    <>
      <div className="text-xs text-muted-foreground">
        Сильні токени з тексту, що зсунули prediction у{" "}
        <strong>{direction}</strong>. Позитивна attribution → FAKE; від'ємна → REAL.
      </div>

      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ left: 8, right: 16, top: 4, bottom: 4 }}
          >
            <XAxis type="number" tick={{ fontSize: 10 }} />
            <YAxis
              type="category"
              dataKey="token"
              tick={{ fontSize: 11 }}
              width={120}
            />
            <Tooltip
              formatter={(v: number) => v.toFixed(3)}
              cursor={{ fill: "transparent" }}
            />
            <Bar dataKey="attribution" radius={[2, 2, 2, 2]}>
              {chartData.map((d, i) => (
                <Cell
                  key={i}
                  fill={d.fillKey === "fake" ? "#ef4444" : "#22c55e"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {originalText && (
        <HighlightedTextView text={originalText} tokens={allTokens} />
      )}

      {allTokens.length > 10 && (
        <div className="flex items-center justify-between border-t pt-2">
          <span className="text-xs text-muted-foreground">
            Показано {visibleTokens.length} з {allTokens.length} токенів
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowAll((v) => !v)}
            className="h-7 text-xs"
          >
            {showAll ? "Згорнути" : "Розгорнути всі токени"}
          </Button>
        </div>
      )}

      {showAll && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-1 text-xs">
          {visibleTokens.map((t, i) => (
            <div
              key={i}
              className="flex items-center justify-between px-2 py-1 rounded border bg-muted/30"
            >
              <span className="font-mono truncate">{t.token}</span>
              <span
                className={cn(
                  "font-mono tabular-nums",
                  t.attribution >= 0
                    ? "text-red-600 dark:text-red-400"
                    : "text-green-600 dark:text-green-400",
                )}
              >
                {t.attribution >= 0 ? "+" : ""}
                {t.attribution.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function HighlightedTextView({
  text,
  tokens,
}: {
  text: string;
  tokens: TokenAttribution[];
}) {
  // Для highlight: word-level match. WordPiece-токени (`is_subword`) ідуть
  // як часткові; lookup case-insensitive за нормалізованим token.
  const attrMap = useMemo(() => {
    const m = new Map<string, number>();
    let maxAbs = 0;
    for (const t of tokens) {
      const key = t.token.toLowerCase();
      const existing = m.get(key) || 0;
      // Беремо максимум за |abs| на випадок повторів
      if (Math.abs(t.attribution) > Math.abs(existing)) {
        m.set(key, t.attribution);
      }
      maxAbs = Math.max(maxAbs, Math.abs(t.attribution));
    }
    return { map: m, maxAbs: maxAbs || 1 };
  }, [tokens]);

  // Розбиваємо текст на слова + пробіли (зберігаємо пунктуацію між)
  const parts = text.split(/(\s+|[.,!?;:()"'`])/g).filter(Boolean);

  return (
    <div className="rounded-md border bg-muted/20 p-3 text-sm leading-relaxed">
      <div className="text-xs text-muted-foreground mb-2">
        Підсвітка тексту (інтенсивність = сила attribution):
      </div>
      <div className="whitespace-pre-wrap break-words">
        {parts.map((p, i) => {
          const attr = attrMap.map.get(p.toLowerCase());
          if (attr == null) {
            return <span key={i}>{p}</span>;
          }
          const intensity = Math.min(Math.abs(attr) / attrMap.maxAbs, 1);
          const opacity = 0.15 + 0.55 * intensity;
          const bg =
            attr >= 0
              ? `rgba(239, 68, 68, ${opacity})`
              : `rgba(34, 197, 94, ${opacity})`;
          return (
            <span
              key={i}
              style={{ backgroundColor: bg, borderRadius: 3, padding: "0 2px" }}
              title={`attribution: ${attr.toFixed(3)}`}
            >
              {p}
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ── Graph subgraph (GIN / SAGE) ──────────────────────────────────────────

function GraphSubgraphView({ explanation }: { explanation: GnnExplanation }) {
  const nodes = explanation.important_nodes || [];
  const edges = explanation.important_edges || [];

  return (
    <>
      <div className="text-xs text-muted-foreground flex items-center gap-1.5">
        <Network className="h-3.5 w-3.5" />
        Найважливіший підграф: <strong>{nodes.length}</strong> з{" "}
        {explanation.n_nodes_total} вузлів, <strong>{edges.length}</strong> з{" "}
        {explanation.n_edges_total} ребер
        {explanation.architecture && (
          <Badge variant="outline" className="text-[10px] ml-1">
            {explanation.architecture.toUpperCase()}
          </Badge>
        )}
      </div>

      <SubgraphSVG nodes={nodes} edges={edges} />

      <div>
        <div className="text-xs font-medium text-muted-foreground mb-1.5">
          Топ-{Math.min(5, nodes.length)} вузлів:
        </div>
        <div className="space-y-1">
          {nodes.slice(0, 5).map((n) => (
            <div
              key={n.node_id}
              className="flex items-start gap-2 text-xs border rounded px-2 py-1.5 bg-muted/20"
            >
              <Badge
                variant="outline"
                className={cn(
                  "text-[9px] h-4 px-1.5 shrink-0",
                  nodeColorClass(n.metadata?.type),
                )}
              >
                {n.metadata?.type || "node"}
              </Badge>
              <div className="flex-1 min-w-0">
                {n.metadata?.text ? (
                  <p className="line-clamp-2 leading-snug">
                    «{n.metadata.text}»
                  </p>
                ) : (
                  <p className="text-muted-foreground italic">
                    node #{n.node_id}
                  </p>
                )}
              </div>
              <span className="font-mono tabular-nums text-violet-600 dark:text-violet-400 shrink-0">
                {n.importance.toFixed(3)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

function nodeColorClass(type?: string): string {
  switch (type) {
    case "article":
      return "bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300";
    case "tweet":
      return "bg-green-100 text-green-700 dark:bg-green-950/40 dark:text-green-300";
    case "retweet":
      return "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300";
    case "reply":
      return "bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300";
    default:
      return "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300";
  }
}

function SubgraphSVG({
  nodes,
  edges,
}: {
  nodes: GnnExplanation["important_nodes"];
  edges: GnnExplanation["important_edges"];
}) {
  // Простий radial layout без зовнішніх залежностей:
  // article (якщо є) у центрі; tweet/reply розкладені по колу;
  // інші типи — на зовнішньому колі. Не оптимально для великих графів,
  // але достатньо для демо top-10 ноди.
  const W = 480;
  const H = 280;
  const cx = W / 2;
  const cy = H / 2;

  const visible = nodes.slice(0, 10);
  const visibleIds = new Set(visible.map((n) => n.node_id));
  const visibleEdges = edges.filter(
    (e) => visibleIds.has(e.source) && visibleIds.has(e.target),
  );

  const positions = useMemo(() => {
    const map = new Map<number, { x: number; y: number; r: number }>();
    const articleNode = visible.find((n) => n.metadata?.type === "article");
    const rest = visible.filter((n) => n !== articleNode);

    const maxImp = Math.max(...visible.map((n) => n.importance), 1);
    const minR = 8;
    const maxR = 22;
    const sizeFor = (imp: number) => minR + (maxR - minR) * (imp / maxImp);

    if (articleNode) {
      map.set(articleNode.node_id, {
        x: cx,
        y: cy,
        r: sizeFor(articleNode.importance),
      });
    }
    const ring = Math.min(W, H) / 2 - 30;
    rest.forEach((n, i) => {
      const angle = (2 * Math.PI * i) / Math.max(rest.length, 1) - Math.PI / 2;
      map.set(n.node_id, {
        x: cx + ring * Math.cos(angle),
        y: cy + ring * Math.sin(angle),
        r: sizeFor(n.importance),
      });
    });
    return map;
  }, [visible, cx, cy]);

  const maxEdgeImp = Math.max(...visibleEdges.map((e) => e.importance), 1);

  const nodeFill = (type?: string) => {
    switch (type) {
      case "article":
        return "#3b82f6";
      case "tweet":
        return "#22c55e";
      case "retweet":
        return "#10b981";
      case "reply":
        return "#f59e0b";
      default:
        return "#94a3b8";
    }
  };

  return (
    <div className="rounded-md border bg-muted/10 overflow-hidden">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {/* Edges */}
        {visibleEdges.map((e, i) => {
          const a = positions.get(e.source);
          const b = positions.get(e.target);
          if (!a || !b) return null;
          const strokeWidth = 0.5 + 3 * (e.importance / maxEdgeImp);
          return (
            <line
              key={`e-${i}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke="rgba(120, 113, 200, 0.7)"
              strokeWidth={strokeWidth}
            />
          );
        })}
        {/* Nodes */}
        {visible.map((n) => {
          const p = positions.get(n.node_id);
          if (!p) return null;
          const fill = nodeFill(n.metadata?.type);
          const tip = n.metadata?.text || `node #${n.node_id}`;
          return (
            <g key={n.node_id}>
              <title>{`${n.metadata?.type || "node"} (imp ${n.importance.toFixed(3)}): ${tip}`}</title>
              <circle
                cx={p.x}
                cy={p.y}
                r={p.r}
                fill={fill}
                fillOpacity={0.8}
                stroke="white"
                strokeWidth={1.5}
              />
              <text
                x={p.x}
                y={p.y + 3}
                textAnchor="middle"
                fontSize={9}
                fill="white"
                fontWeight={600}
              >
                {n.node_id}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ── LLM reasoning ────────────────────────────────────────────────────────

function LlmReasoningView({ explanation }: { explanation: LlmExplanation }) {
  return (
    <>
      {explanation.reasoning && (
        <div className="text-sm leading-relaxed">{explanation.reasoning}</div>
      )}

      {explanation.key_indicators && explanation.key_indicators.length > 0 && (
        <div>
          <div className="text-xs font-medium text-muted-foreground mb-1.5">
            Ключові індикатори:
          </div>
          <ul className="space-y-1">
            {explanation.key_indicators.map((k, i) => (
              <li key={i} className="flex items-start gap-2 text-sm">
                <Lightbulb className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-500" />
                <span>{k}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {explanation.confidence_factors && explanation.confidence_factors.length > 0 && (
        <div>
          <div className="text-xs font-medium text-muted-foreground mb-1.5">
            Чинники впевненості:
          </div>
          <div className="space-y-1.5">
            {explanation.confidence_factors.map((f, i) => {
              const pct = Math.max(0, Math.min(1, f.value)) * 100;
              return (
                <div key={i}>
                  <div className="flex items-center justify-between text-xs mb-0.5">
                    <span>{f.name}</span>
                    <span className="font-mono tabular-nums text-muted-foreground">
                      {pct.toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-muted rounded overflow-hidden">
                    <div
                      className="h-full bg-violet-500"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {explanation.uncertainty_factors && explanation.uncertainty_factors.length > 0 && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <div className="text-xs font-medium mb-1">Фактори невпевненості:</div>
            <ul className="space-y-0.5 text-sm">
              {explanation.uncertainty_factors.map((u, i) => (
                <li key={i}>• {u}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {!explanation.reasoning
        && !explanation.key_indicators?.length
        && !explanation.confidence_factors?.length && (
        <div className="text-sm text-muted-foreground italic">
          LLM не повернув reasoning для цього prediction.
        </div>
      )}
    </>
  );
}

// ── Metadata footer ──────────────────────────────────────────────────────

function ExplanationMeta({ explanation }: { explanation: Explanation }) {
  const params = (explanation as any).method_params;
  const extra: [string, unknown][] = [];
  if (params && typeof params === "object") {
    for (const [k, v] of Object.entries(params)) extra.push([k, v]);
  }
  if ("n_features_used" in explanation) {
    extra.push(["n_features_used", (explanation as NbExplanation).n_features_used]);
  }
  if ("total_log_odds" in explanation) {
    extra.push(["total_log_odds", (explanation as NbExplanation).total_log_odds]);
  }
  if ("cached" in explanation) {
    extra.push(["cached", (explanation as GnnExplanation).cached]);
  }
  if (extra.length === 0) return null;

  return (
    <details className="text-xs text-muted-foreground">
      <summary className="cursor-pointer hover:text-foreground flex items-center gap-1">
        <Sparkles className="h-3 w-3" />
        Деталі методу
      </summary>
      <div className="mt-1 pl-2 border-l-2 border-muted space-y-0.5 font-mono">
        {extra.map(([k, v]) => (
          <div key={k}>
            {k}: {typeof v === "object" ? JSON.stringify(v) : String(v)}
          </div>
        ))}
      </div>
    </details>
  );
}
