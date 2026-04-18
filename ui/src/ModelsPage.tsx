import React, { useEffect, useState } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "./components/ui/dropdown-menu";
import { Brain, Zap, Sparkles, MoreVertical, CheckCircle2, Loader2, Trash2, Boxes } from "lucide-react";
import { toast } from "sonner";
import type { ModelRecord } from "./types";

const MODEL_ICONS: Record<string, any> = {
  nb: Brain,
  deberta: Zap,
  llm: Sparkles,
};

function ModelsPage() {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [activating, setActivating] = useState<number | null>(null);
  const [deletingAll, setDeletingAll] = useState<boolean>(false);

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
                        m.model_type === "deberta" && "bg-violet-100 dark:bg-violet-950 text-violet-600",
                        m.model_type === "llm" && "bg-amber-100 dark:bg-amber-950 text-amber-600",
                      )}>
                        <Icon className="h-5 w-5" />
                      </div>
                      <div>
                        <p className="font-semibold text-sm">{m.name || m.filename}</p>
                        <Badge variant="secondary" className="mt-1 text-[10px]">
                          {m.model_type}
                        </Badge>
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
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>

                  {m.accuracy != null && (
                    <div className="mb-3">
                      <span className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {(m.accuracy * 100).toFixed(1)}%
                      </span>
                      <span className="text-xs text-muted-foreground ml-1">accuracy</span>
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
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default ModelsPage;
