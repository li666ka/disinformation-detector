import type { Ensemble, EnsembleMemberInfo } from "./types";
import { cn } from "./lib/utils";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "./components/ui/dialog";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import {
  Brain,
  Layers,
  Network,
  Share2,
  Sparkles,
  TrendingDown,
  TrendingUp,
  Zap,
} from "lucide-react";

interface Props {
  ensemble: Ensemble;
  onClose: () => void;
}

const formatPercent = (v: number | null | undefined): string =>
  v == null ? "—" : `${(v * 100).toFixed(1)}%`;

const VOTING_LABEL: Record<string, string> = {
  hard: "Hard Voting",
  soft: "Soft Voting",
  weighted: "Weighted Voting",
};

const MODEL_ICONS: Record<string, any> = {
  nb: Brain,
  distilbert: Zap,
  deberta: Zap,
  llm: Sparkles,
  gin: Network,
  sage: Share2,
  gnn: Network,
};

const modelIconClasses = (t: string): string => {
  switch ((t || "").toLowerCase()) {
    case "nb":
      return "bg-blue-100 dark:bg-blue-950 text-blue-600";
    case "distilbert":
    case "deberta":
    case "bert":
      return "bg-violet-100 dark:bg-violet-950 text-violet-600";
    case "llm":
      return "bg-amber-100 dark:bg-amber-950 text-amber-600";
    case "gin":
      return "bg-emerald-100 dark:bg-emerald-950 text-emerald-600";
    case "sage":
      return "bg-teal-100 dark:bg-teal-950 text-teal-600";
    default:
      return "bg-muted text-muted-foreground";
  }
};

const modelTypeLabel = (t: string): string => {
  switch ((t || "").toLowerCase()) {
    case "nb":
      return "NB";
    case "distilbert":
      return "DistilBERT";
    case "deberta":
      return "DeBERTa";
    case "gin":
      return "GIN";
    case "sage":
      return "SAGE";
    case "llm":
      return "LLM";
    default:
      return (t || "").toUpperCase();
  }
};

const formatSplits = (s: string | null): string => {
  if (!s) return "auto-split";
  const low = s.toLowerCase();
  if (low.includes("cross")) return "cross-domain";
  if (low.includes("in_domain") || low === "in") return "in-domain";
  if (low.includes("mixed")) return "mixed";
  return s;
};

export default function EnsembleDetailsModal({ ensemble, onClose }: Props) {
  const cm = ensemble.confusion_matrix;
  const members: EnsembleMemberInfo[] = ensemble.member_models || [];

  const best = members.reduce<EnsembleMemberInfo | null>((acc, m) => {
    if (acc == null || (m.f1_macro ?? -1) > (acc.f1_macro ?? -1)) return m;
    return acc;
  }, null);

  const diff =
    best && ensemble.f1_macro != null && best.f1_macro != null
      ? ensemble.f1_macro - best.f1_macro
      : null;

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            {ensemble.name}
          </DialogTitle>
          <DialogDescription className="flex flex-wrap items-center gap-2 pt-1">
            <Badge variant="outline" className="text-xs">
              {VOTING_LABEL[ensemble.voting_type] || ensemble.voting_type}
            </Badge>
            <span className="text-muted-foreground">·</span>
            <span>{members.length} моделей</span>
            <span className="text-muted-foreground">·</span>
            <span>split: {formatSplits(ensemble.splits_used)}</span>
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Metric cards — 4 main */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Accuracy", value: ensemble.accuracy, color: "text-green-600" },
              { label: "Precision (FAKE)", value: ensemble.precision, color: "text-blue-600" },
              { label: "Recall (FAKE)", value: ensemble.recall, color: "text-violet-600" },
              { label: "F1 (FAKE)", value: ensemble.f1_score, color: "text-amber-600" },
            ].map((metric) => (
              <Card key={metric.label}>
                <CardContent className="pt-5 pb-4 text-center">
                  <div className={`text-2xl font-bold ${metric.color}`}>
                    {formatPercent(metric.value)}
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {metric.label}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Additional metrics */}
          {(ensemble.f1_macro != null || ensemble.roc_auc != null) && (
            <div>
              <h3 className="text-sm font-semibold mb-2">Додаткові метрики</h3>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "F1 (macro)", value: ensemble.f1_macro, color: "text-emerald-600" },
                  { label: "ROC AUC", value: ensemble.roc_auc, color: "text-indigo-600" },
                ].map((metric) => (
                  <Card key={metric.label}>
                    <CardContent className="pt-5 pb-4 text-center">
                      <div className={`text-2xl font-bold ${metric.color}`}>
                        {formatPercent(metric.value)}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {metric.label}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {/* Confusion Matrix */}
          {cm && (
            <div>
              <h3 className="text-sm font-semibold mb-1">Матриця помилок</h3>
              <p className="text-xs text-muted-foreground mb-2">
                FAKE = позитивний клас
              </p>
              <div className="overflow-hidden rounded-lg border">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-muted/50">
                      <th className="p-2 text-left font-medium"></th>
                      <th className="p-2 text-center font-medium">
                        Передбачено: REAL
                      </th>
                      <th className="p-2 text-center font-medium">
                        Передбачено: FAKE
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-t">
                      <th className="p-2 text-left font-medium bg-muted/50">
                        Фактично: REAL
                      </th>
                      <td className="p-3 text-center font-semibold text-green-600 bg-green-50 dark:bg-green-950/20">
                        {cm.tn}
                      </td>
                      <td className="p-3 text-center font-semibold text-red-600 bg-red-50 dark:bg-red-950/20">
                        {cm.fp}
                      </td>
                    </tr>
                    <tr className="border-t">
                      <th className="p-2 text-left font-medium bg-muted/50">
                        Фактично: FAKE
                      </th>
                      <td className="p-3 text-center font-semibold text-red-600 bg-red-50 dark:bg-red-950/20">
                        {cm.fn}
                      </td>
                      <td className="p-3 text-center font-semibold text-green-600 bg-green-50 dark:bg-green-950/20">
                        {cm.tp}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Members */}
          <div>
            <h3 className="text-sm font-semibold mb-2">Члени ансамблю</h3>
            <div className="space-y-2">
              {members.map((m) => {
                const Icon = MODEL_ICONS[m.model_type] || Brain;
                const isBest =
                  best != null &&
                  m.id === best.id &&
                  (m.f1_macro ?? -1) > -1;
                return (
                  <Card key={m.id} className={cn(isBest && "ring-1 ring-amber-400")}>
                    <CardContent className="p-3 flex items-center gap-3">
                      <div
                        className={cn(
                          "w-9 h-9 rounded-lg flex items-center justify-center shrink-0",
                          modelIconClasses(m.model_type)
                        )}
                      >
                        <Icon className="h-4 w-4" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium text-sm truncate">{m.name}</p>
                        <div className="flex flex-wrap gap-1 mt-0.5">
                          <Badge variant="secondary" className="text-[10px]">
                            {modelTypeLabel(m.model_type)}
                          </Badge>
                          {isBest && (
                            <Badge className="text-[10px] bg-amber-500 hover:bg-amber-500/90 text-white">
                              найкращий член
                            </Badge>
                          )}
                        </div>
                      </div>
                      <div className="text-right text-xs">
                        <div className="font-mono font-semibold text-green-600 dark:text-green-400">
                          {formatPercent(m.accuracy)}
                        </div>
                        <div className="text-muted-foreground">
                          F1-macro: {formatPercent(m.f1_macro)}
                        </div>
                        {ensemble.voting_type === "weighted" && m.weight != null && (
                          <div className="text-primary font-mono mt-0.5">
                            вага: {m.weight.toFixed(2)}
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>

          {/* Comparison vs best member */}
          {best && diff != null && (
            <Card>
              <CardContent className="p-3 text-sm">
                <div className="flex items-center gap-2">
                  {diff > 0 ? (
                    <TrendingUp className="h-4 w-4 text-green-600 shrink-0" />
                  ) : diff < 0 ? (
                    <TrendingDown className="h-4 w-4 text-amber-600 shrink-0" />
                  ) : null}
                  <span className="font-medium">Ансамбль vs найкращий член:</span>
                  {diff > 0 ? (
                    <span className="text-green-600 dark:text-green-400 font-mono">
                      +{(diff * 100).toFixed(2)} pp
                    </span>
                  ) : diff < 0 ? (
                    <span className="text-amber-600 dark:text-amber-400 font-mono">
                      {(diff * 100).toFixed(2)} pp
                    </span>
                  ) : (
                    <span className="text-muted-foreground">однаковий результат</span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Найкращий: {best.name} (F1-macro: {formatPercent(best.f1_macro)})
                </p>
              </CardContent>
            </Card>
          )}

          <div className="text-xs text-muted-foreground text-right">
            Створено: {new Date(ensemble.created_at).toLocaleString("uk-UA")}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
