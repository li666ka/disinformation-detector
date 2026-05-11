import React, { useEffect, useState } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "./components/ui/dropdown-menu";
import { Brain, Zap, Sparkles, MoreVertical, CheckCircle2, Loader2, Trash2, Boxes, Target, Network, Share2, Database } from "lucide-react";
import { toast } from "sonner";
import type { ModelRecord } from "./types";
import EvaluationResultsModal from "./EvaluationResultsModal";

const MODEL_ICONS: Record<string, any> = {
  nb: Brain,
  distilbert: Zap,
  deberta: Zap, // legacy records — keep for backward compat
  llm: Sparkles,
  gin: Network,
  sage: Share2,
  gnn: Network, // fallback for records saved without an explicit architecture
};

function ModelsPage() {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [activating, setActivating] = useState<number | null>(null);
  const [deletingAll, setDeletingAll] = useState<boolean>(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [evaluatingId, setEvaluatingId] = useState<number | null>(null);
  const [evalResult, setEvalResult] = useState<any>(null);

  const fetchModels = async () => {
    try {
      const { data } = await api.get("/models");
      setModels(data);
    } catch (err) {
      toast.error("Не вдалося завантажити список моделей");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchModels(); }, []);

  const handleActivate = async (modelId: number) => {
    setActivating(modelId);
    try {
      await api.patch(`/models/${modelId}/activate`);
      await fetchModels();
      toast.success("Модель активовано");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка активації моделі");
    } finally {
      setActivating(null);
    }
  };

  const handleEvaluate = async (modelId: number) => {
    setEvaluatingId(modelId);
    try {
      const { data } = await api.post(`/models/${modelId}/evaluate`, {
        max_samples: 200,
        test_size: 0.2,
      });
      setEvalResult(data);
      await fetchModels();
      toast.success(
        `Evaluation готова: accuracy ${(data.metrics.accuracy * 100).toFixed(1)}%`
      );
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка оцінювання");
    } finally {
      setEvaluatingId(null);
    }
  };

  const handleDelete = async (modelId: number, modelName: string) => {
    if (!window.confirm(`Видалити модель "${modelName}"? Цю дію не можна скасувати.`)) return;
    setDeletingId(modelId);
    try {
      await api.delete(`/models/${modelId}`);
      await fetchModels();
      toast.success("Модель видалено");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка видалення");
    } finally {
      setDeletingId(null);
    }
  };

  const handleDeleteAll = async () => {
    if (!window.confirm("Видалити всі моделі? Цю дію не можна скасувати.")) return;
    setDeletingAll(true);
    try {
      await api.delete("/models");
      await fetchModels();
      toast.success("Всі моделі видалено");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка видалення моделей");
    } finally {
      setDeletingAll(false);
    }
  };

  const formatDate = (iso: string) => new Date(iso).toLocaleString("uk-UA");

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Моделі</h2>
          <p className="text-muted-foreground">Збережені моделі класифікації</p>
        </div>
        {models.length > 0 && (
          <Button
            variant="destructive"
            size="sm"
            onClick={handleDeleteAll}
            disabled={deletingAll}
          >
            {deletingAll ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Trash2 className="mr-2 h-4 w-4" />}
            Видалити всі
          </Button>
        )}
      </div>

      {models.length === 0 ? (
        <div className="text-center py-16">
          <Boxes className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <p className="text-muted-foreground">
            Моделей ще немає. Навчіть першу модель у вкладці "Навчання моделі".
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models.map((m) => {
            const Icon = MODEL_ICONS[m.model_type] || Brain;
            return (
              <Card key={m.id} className={cn(m.is_active && "ring-2 ring-green-500")}>
                <CardContent className="p-5">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <div className={cn(
                        "w-10 h-10 rounded-lg flex items-center justify-center",
                        m.model_type === "nb" && "bg-blue-100 dark:bg-blue-950 text-blue-600",
                        (m.model_type === "distilbert" || m.model_type === "deberta") && "bg-violet-100 dark:bg-violet-950 text-violet-600",
                        m.model_type === "llm" && "bg-amber-100 dark:bg-amber-950 text-amber-600",
                        m.model_type === "gin" && "bg-emerald-100 dark:bg-emerald-950 text-emerald-600",
                        m.model_type === "sage" && "bg-teal-100 dark:bg-teal-950 text-teal-600",
                      )}>
                        <Icon className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="font-semibold text-sm">{m.name || m.filename}</p>
                        <div className="flex flex-wrap gap-1 mt-1">
                          <Badge variant="secondary" className="text-[10px]">
                            {m.model_type}
                          </Badge>
                        </div>
                        {(m.dataset_name || m.splits_used) && (
                          <p className="text-[11px] text-muted-foreground mt-1">
                            {m.dataset_name && (
                              <span title="Датасет, на якому тренувалась модель">
                                <Database className="inline h-3 w-3 mr-1 -mt-0.5" />
                                {m.dataset_name}
                              </span>
                            )}
                            {m.dataset_name && (m.splits_used || m.pipeline_type) && (
                              <span className="mx-1.5 text-muted-foreground/50">·</span>
                            )}
                            <span title="Split-набір використаний при тренуванні">
                              {m.splits_used && m.splits_used !== "auto"
                                ? `split: ${m.splits_used}`
                                : "auto-split"}
                            </span>
                          </p>
                        )}
                      </div>
                    </div>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        {!m.is_active && (
                          <DropdownMenuItem onClick={() => handleActivate(m.id)}>
                            <CheckCircle2 className="mr-2 h-4 w-4" />
                            Активувати
                          </DropdownMenuItem>
                        )}
                        <DropdownMenuItem
                          onClick={() => handleEvaluate(m.id)}
                          disabled={evaluatingId === m.id}
                        >
                          {evaluatingId === m.id ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <Target className="mr-2 h-4 w-4" />
                          )}
                          Оцінити на test set
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          className="text-destructive"
                          onClick={() => handleDelete(m.id, m.name || m.filename || `#${m.id}`)}
                          disabled={deletingId === m.id}
                        >
                          {deletingId === m.id ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="mr-2 h-4 w-4" />
                          )}
                          Видалити
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>

                  {m.accuracy != null ? (
                    <div className="mb-3 space-y-1">
                      <div>
                        <span className="text-2xl font-bold text-green-600 dark:text-green-400">
                          {(m.accuracy * 100).toFixed(1)}%
                        </span>
                        <span className="text-xs text-muted-foreground ml-1">accuracy</span>
                      </div>
                      {(m.precision != null || m.recall != null || m.f1_score != null) && (
                        <div className="grid grid-cols-3 gap-1 text-[11px]">
                          <div className="rounded bg-blue-50 dark:bg-blue-950/30 p-1 text-center">
                            <div className="font-semibold text-blue-600 dark:text-blue-400">
                              {m.precision != null ? (m.precision * 100).toFixed(1) + "%" : "—"}
                            </div>
                            <div className="text-muted-foreground">P</div>
                          </div>
                          <div className="rounded bg-violet-50 dark:bg-violet-950/30 p-1 text-center">
                            <div className="font-semibold text-violet-600 dark:text-violet-400">
                              {m.recall != null ? (m.recall * 100).toFixed(1) + "%" : "—"}
                            </div>
                            <div className="text-muted-foreground">R</div>
                          </div>
                          <div className="rounded bg-amber-50 dark:bg-amber-950/30 p-1 text-center">
                            <div className="font-semibold text-amber-600 dark:text-amber-400">
                              {m.f1_score != null ? (m.f1_score * 100).toFixed(1) + "%" : "—"}
                            </div>
                            <div className="text-muted-foreground">F1</div>
                          </div>
                        </div>
                      )}
                      {(m.f1_macro != null || m.roc_auc != null) && (
                        <div className="grid grid-cols-2 gap-1 text-[11px] mt-1">
                          {m.f1_macro != null && (
                            <div className="rounded bg-emerald-50 dark:bg-emerald-950/30 p-1 text-center">
                              <div className="font-semibold text-emerald-600 dark:text-emerald-400">
                                {(m.f1_macro * 100).toFixed(1)}%
                              </div>
                              <div className="text-muted-foreground">F1 macro</div>
                            </div>
                          )}
                          {m.roc_auc != null && (
                            <div className="rounded bg-indigo-50 dark:bg-indigo-950/30 p-1 text-center">
                              <div className="font-semibold text-indigo-600 dark:text-indigo-400">
                                {(m.roc_auc * 100).toFixed(1)}%
                              </div>
                              <div className="text-muted-foreground">ROC AUC</div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="mb-3 flex items-center gap-2 text-xs text-muted-foreground">
                      <Target className="h-3 w-3" />
                      Не оцінено — запустіть "Оцінити на test set"
                    </div>
                  )}

                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{formatDate(m.created_at)}</span>
                    {m.is_active && (
                      <div className="flex items-center gap-1 text-green-600 dark:text-green-400 font-medium">
                        <div className="w-2 h-2 rounded-full bg-green-500" />
                        Активна
                      </div>
                    )}
                  </div>

                  {activating === m.id && (
                    <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      Активація...
                    </div>
                  )}

                  {evaluatingId === m.id && (
                    <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                      <Loader2 className="h-3 w-3 animate-spin" />
                      Оцінювання...
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      <EvaluationResultsModal
        open={!!evalResult}
        onClose={() => setEvalResult(null)}
        result={evalResult}
      />
    </div>
  );
}

export default ModelsPage;
