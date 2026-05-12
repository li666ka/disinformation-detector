import { useCallback, useEffect, useState } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import type { Ensemble, EnsembleSummary } from "./types";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./components/ui/dropdown-menu";
import {
  CheckCircle2,
  Eye,
  Layers,
  Loader2,
  MoreVertical,
  Plus,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import CreateEnsembleWizard from "./CreateEnsembleWizard";
import EnsembleDetailsModal from "./EnsembleDetailsModal";

interface Props {
  refreshTrigger?: number;
}

const formatPercent = (v: number | null | undefined): string =>
  v == null ? "—" : `${(v * 100).toFixed(1)}%`;

const VOTING_LABEL: Record<string, string> = {
  hard: "Hard",
  soft: "Soft",
  weighted: "Weighted",
};

const formatSplits = (s: string | null): string => {
  if (!s) return "auto-split";
  const low = s.toLowerCase();
  if (low.includes("cross")) return "cross-domain";
  if (low.includes("in_domain") || low === "in") return "in-domain";
  if (low.includes("mixed")) return "mixed";
  return s;
};

const formatDate = (iso: string) => new Date(iso).toLocaleString("uk-UA");

export default function EnsemblesPage({ refreshTrigger }: Props) {
  const [ensembles, setEnsembles] = useState<EnsembleSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [activatingId, setActivatingId] = useState<number | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [showWizard, setShowWizard] = useState(false);
  const [selectedEnsemble, setSelectedEnsemble] = useState<Ensemble | null>(null);

  const loadEnsembles = useCallback(async () => {
    setLoading(true);
    try {
      const { data } = await api.get<EnsembleSummary[]>("/ensembles");
      setEnsembles(data);
    } catch {
      toast.error("Не вдалося завантажити ансамблі");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadEnsembles();
  }, [loadEnsembles, refreshTrigger]);

  const handleActivate = async (id: number) => {
    setActivatingId(id);
    try {
      await api.post(`/ensembles/${id}/activate`);
      await loadEnsembles();
      toast.success("Ансамбль активовано");
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || "Не вдалося активувати");
    } finally {
      setActivatingId(null);
    }
  };

  const handleDelete = async (id: number, name: string) => {
    if (!window.confirm(`Видалити ансамбль «${name}»? Цю дію не можна скасувати.`))
      return;
    setDeletingId(id);
    try {
      await api.delete(`/ensembles/${id}`);
      await loadEnsembles();
      toast.success("Ансамбль видалено");
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || "Не вдалося видалити");
    } finally {
      setDeletingId(null);
    }
  };

  const handleViewDetails = async (id: number) => {
    try {
      const { data } = await api.get<Ensemble>(`/ensembles/${id}`);
      setSelectedEnsemble(data);
    } catch {
      toast.error("Не вдалося завантажити деталі");
    }
  };

  const handleCreated = () => {
    setShowWizard(false);
    loadEnsembles();
    toast.success("Ансамбль створено та оцінено");
  };

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
          <h2 className="text-2xl font-bold tracking-tight">Ансамблі</h2>
          <p className="text-muted-foreground">
            Об'єднання натренованих моделей через voting (hard / soft / weighted)
          </p>
        </div>
        <Button onClick={() => setShowWizard(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Створити ансамбль
        </Button>
      </div>

      {ensembles.length === 0 ? (
        <div className="text-center py-16">
          <Layers className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <p className="text-muted-foreground">
            Ансамблів ще немає. Натренуйте 2+ моделі та створіть перший.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {ensembles.map((e) => (
            <Card key={e.id} className={cn(e.is_active && "ring-2 ring-green-500")}>
              <CardContent className="p-5">
                {/* Header: icon + name + dropdown */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg flex items-center justify-center bg-violet-100 dark:bg-violet-950 text-violet-600">
                      <Layers className="h-5 w-5" />
                    </div>
                    <div>
                      <p className="font-semibold text-sm">{e.name}</p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        <Badge variant="secondary" className="text-[10px]">
                          {VOTING_LABEL[e.voting_type] || e.voting_type}
                        </Badge>
                        <Badge variant="outline" className="text-[10px]">
                          {e.member_count} моделей
                        </Badge>
                      </div>
                      <p className="text-[11px] text-muted-foreground mt-1">
                        <span title="Split-набір використаний при тренуванні">
                          split: {formatSplits(e.splits_used)}
                        </span>
                      </p>
                    </div>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" className="h-8 w-8">
                        <MoreVertical className="h-4 w-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem onClick={() => handleViewDetails(e.id)}>
                        <Eye className="mr-2 h-4 w-4" />
                        Деталі
                      </DropdownMenuItem>
                      {!e.is_active && (
                        <DropdownMenuItem
                          onClick={() => handleActivate(e.id)}
                          disabled={activatingId === e.id}
                        >
                          {activatingId === e.id ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <CheckCircle2 className="mr-2 h-4 w-4" />
                          )}
                          Активувати
                        </DropdownMenuItem>
                      )}
                      <DropdownMenuItem
                        className="text-destructive"
                        onClick={() => handleDelete(e.id, e.name)}
                        disabled={deletingId === e.id}
                      >
                        {deletingId === e.id ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                          <Trash2 className="mr-2 h-4 w-4" />
                        )}
                        Видалити
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                {/* Metrics block — у стилі ModelsPage */}
                {e.accuracy != null ? (
                  <div className="mb-3 space-y-1">
                    <div>
                      <span className="text-2xl font-bold text-green-600 dark:text-green-400">
                        {formatPercent(e.accuracy)}
                      </span>
                      <span className="text-xs text-muted-foreground ml-1">accuracy</span>
                    </div>
                    {e.f1_macro != null && (
                      <div className="grid grid-cols-1 gap-1 text-[11px]">
                        <div className="rounded bg-emerald-50 dark:bg-emerald-950/30 p-1 text-center">
                          <div className="font-semibold text-emerald-600 dark:text-emerald-400">
                            {formatPercent(e.f1_macro)}
                          </div>
                          <div className="text-muted-foreground">F1 macro</div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="mb-3 flex items-center gap-2 text-xs text-muted-foreground">
                    <Layers className="h-3 w-3" />
                    Не оцінено
                  </div>
                )}

                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>{formatDate(e.created_at)}</span>
                  {e.is_active && (
                    <div className="flex items-center gap-1 text-green-600 dark:text-green-400 font-medium">
                      <div className="w-2 h-2 rounded-full bg-green-500" />
                      Активний
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {showWizard && (
        <CreateEnsembleWizard
          onClose={() => setShowWizard(false)}
          onSuccess={handleCreated}
        />
      )}

      {selectedEnsemble && (
        <EnsembleDetailsModal
          ensemble={selectedEnsemble}
          onClose={() => setSelectedEnsemble(null)}
        />
      )}
    </div>
  );
}
