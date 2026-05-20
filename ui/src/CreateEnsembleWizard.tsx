import { useEffect, useMemo, useState } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import type {
  CreateEnsembleRequest,
  EligibleModel,
  EligibleModelsResponse,
  VotingType,
} from "./types";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "./components/ui/dialog";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Checkbox } from "./components/ui/checkbox";
import { Badge } from "./components/ui/badge";
import { Alert, AlertDescription } from "./components/ui/alert";
import {
  AlertCircle,
  Brain,
  Check,
  ChevronLeft,
  ChevronRight,
  Info,
  Layers,
  Loader2,
  Network,
  Share2,
  Sparkles,
  Zap,
} from "lucide-react";
import { toast } from "sonner";

interface Props {
  onClose: () => void;
  onSuccess: () => void;
}

type Step = 0 | 1 | 2 | 3;

const STEP_LABELS = ["Моделі", "Voting", "Ваги", "Підсумок"];

const formatPercent = (v: number | null | undefined): string =>
  v == null ? "—" : `${(v * 100).toFixed(1)}%`;

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
      return t.toUpperCase();
  }
};

const VOTING_TYPES: Array<{
  id: VotingType;
  title: string;
  desc: string;
}> = [
  {
    id: "hard",
    title: "Hard Voting",
    desc: "Більшість голосів. Не враховує впевненість моделей.",
  },
  {
    id: "soft",
    title: "Soft Voting",
    desc: "Середнє ймовірностей FAKE. Зазвичай найкращий вибір.",
  },
  {
    id: "weighted",
    title: "Weighted Voting",
    desc: "Зважена сума. Ваги задаються вручну на наступному кроці.",
  },
];

export default function CreateEnsembleWizard({ onClose, onSuccess }: Props) {
  const [step, setStep] = useState<Step>(0);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const [eligibleModels, setEligibleModels] = useState<EligibleModel[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [votingType, setVotingType] = useState<VotingType>("soft");
  const [weights, setWeights] = useState<Record<number, number>>({});
  const [name, setName] = useState("");

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const { data } = await api.get<EligibleModelsResponse>(
          "/ensembles/eligible-models"
        );
        setEligibleModels(data.models);
      } catch {
        toast.error("Не вдалося завантажити список моделей");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const selectedModels = useMemo(
    () => eligibleModels.filter((m) => selectedIds.has(m.id)),
    [eligibleModels, selectedIds]
  );

  const splitsValidation = useMemo(() => {
    const splits = new Set(
      selectedModels
        .map((m) => m.splits_used)
        .filter((s): s is string => s != null)
    );
    return {
      mixedSplits: splits.size > 1,
      uniqueSplits: Array.from(splits),
    };
  }, [selectedModels]);

  const missingPredictions = useMemo(
    () => selectedModels.filter((m) => !m.has_predictions),
    [selectedModels]
  );

  const canProceedStep0 =
    selectedIds.size >= 2 &&
    !splitsValidation.mixedSplits &&
    missingPredictions.length === 0;

  const canProceedStep2 = useMemo(() => {
    if (votingType !== "weighted") return true;
    return selectedModels.every((m) => (weights[m.id] ?? 0) > 0);
  }, [votingType, weights, selectedModels]);

  const canSubmit = name.trim().length > 0 && canProceedStep2;


  useEffect(() => {
    if (step === 2 && votingType === "weighted" && selectedModels.length > 0) {
      const equal = 1 / selectedModels.length;
      setWeights((prev) => {
        const next: Record<number, number> = {};
        selectedModels.forEach((m) => {
          next[m.id] = prev[m.id] ?? equal;
        });
        return next;
      });
    }

  }, [step, votingType, selectedModels.length]);


  useEffect(() => {
    if (step === 3 && !name) {
      const types = Array.from(
        new Set(selectedModels.map((m) => modelTypeLabel(m.model_type)))
      );
      setName(
        `${votingType.toUpperCase()} ${types.join("+")} (${selectedModels.length} моделей)`
      );
    }

  }, [step]);

  const toggleModel = (id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const updateWeight = (id: number, val: number) => {
    setWeights((prev) => ({ ...prev, [id]: val }));
  };

  const normalizeWeights = () => {
    const total = Object.values(weights).reduce((s, w) => s + w, 0);
    if (total <= 0) return;
    const normalized: Record<number, number> = {};
    Object.entries(weights).forEach(([k, v]) => {
      normalized[Number(k)] = v / total;
    });
    setWeights(normalized);
  };

  const handleNext = () => {
    if (step >= 3) return;
    setStep((s) => (s + 1) as Step);
  };

  const handleBack = () => {
    if (step <= 0) return;
    setStep((s) => (s - 1) as Step);
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      const payload: CreateEnsembleRequest = {
        name: name.trim(),
        voting_type: votingType,
        member_model_ids: Array.from(selectedIds),
      };
      if (votingType === "weighted") {
        const sw: Record<string, number> = {};
        Object.entries(weights).forEach(([k, v]) => {
          sw[String(k)] = v;
        });
        payload.weights = sw;
      }
      await api.post("/ensembles", payload);
      onSuccess();
    } catch (err: any) {
      const detail =
        err?.response?.data?.detail || err?.message || "Невідома помилка";
      toast.error(`Помилка: ${detail}`);
    } finally {
      setSubmitting(false);
    }
  };

  const sumWeights = Object.values(weights).reduce((s, w) => s + w, 0);


  const renderModelPicker = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      );
    }
    if (eligibleModels.length === 0) {
      return (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Немає натренованих моделей. Спочатку натренуйте кілька моделей у
            вкладці «Навчання моделі».
          </AlertDescription>
        </Alert>
      );
    }
    return (
      <div className="space-y-3">
        <p className="text-sm text-muted-foreground">
          Оберіть мінімум 2 моделі. Усі моделі мають бути натреновані на одному
          split (in-domain або cross-domain).
        </p>

        <div className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
          {eligibleModels.map((m) => {
            const selected = selectedIds.has(m.id);
            const disabled = !m.has_predictions;
            const Icon = MODEL_ICONS[m.model_type] || Brain;
            return (
              <Card
                key={m.id}
                className={cn(
                  "cursor-pointer transition-all",
                  selected && "ring-2 ring-primary",
                  disabled && "opacity-50 cursor-not-allowed"
                )}
                onClick={() => !disabled && toggleModel(m.id)}
              >
                <CardContent className="p-4 flex items-center gap-3">
                  <Checkbox
                    checked={selected}
                    disabled={disabled}
                    onCheckedChange={() => !disabled && toggleModel(m.id)}
                    onClick={(e) => e.stopPropagation()}
                  />
                  <div
                    className={cn(
                      "w-10 h-10 rounded-lg flex items-center justify-center shrink-0",
                      modelIconClasses(m.model_type)
                    )}
                  >
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm truncate">{m.name}</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      <Badge variant="secondary" className="text-[10px]">
                        {modelTypeLabel(m.model_type)}
                      </Badge>
                      {m.splits_used && (
                        <Badge variant="outline" className="text-[10px]">
                          {m.splits_used}
                        </Badge>
                      )}
                      {disabled && (
                        <Badge variant="destructive" className="text-[10px]">
                          без predictions
                        </Badge>
                      )}
                    </div>
                  </div>
                  <div className="text-right text-xs">
                    <div className="font-mono font-semibold text-green-600 dark:text-green-400">
                      {formatPercent(m.accuracy)}
                    </div>
                    <div className="text-muted-foreground">
                      F1: {formatPercent(m.f1_score)}
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {selectedIds.size > 0 && selectedIds.size < 2 && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>Потрібно мінімум 2 моделі.</AlertDescription>
          </Alert>
        )}
        {splitsValidation.mixedSplits && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Моделі мають різні splits: {splitsValidation.uniqueSplits.join(", ")}.
              Виберіть моделі з одного split.
            </AlertDescription>
          </Alert>
        )}
        {missingPredictions.length > 0 && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Моделі без predictions: {missingPredictions.map((m) => m.name).join(", ")}.
              Перетренуйте їх.
            </AlertDescription>
          </Alert>
        )}

        <div className="text-xs text-muted-foreground">
          Вибрано: <strong>{selectedIds.size}</strong> моделей
        </div>
      </div>
    );
  };

  const renderVotingPicker = () => (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        Оберіть стратегію об'єднання прогнозів
      </p>
      {VOTING_TYPES.map((v) => (
        <Card
          key={v.id}
          className={cn(
            "cursor-pointer transition-all",
            votingType === v.id && "ring-2 ring-primary"
          )}
          onClick={() => setVotingType(v.id)}
        >
          <CardContent className="p-4 flex items-center gap-3">
            <div
              className={cn(
                "w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0",
                votingType === v.id
                  ? "border-primary bg-primary"
                  : "border-muted-foreground"
              )}
            >
              {votingType === v.id && (
                <Check className="h-3 w-3 text-primary-foreground" />
              )}
            </div>
            <div>
              <p className="font-medium text-sm">{v.title}</p>
              <p className="text-xs text-muted-foreground mt-1">{v.desc}</p>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );

  const renderWeights = () => {
    if (votingType !== "weighted") {
      return (
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            Ви вибрали <strong>{votingType.toUpperCase()}</strong> voting — ваги
            не потрібні. Натисніть «Далі».
          </AlertDescription>
        </Alert>
      );
    }
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Ваги нормалізуються до суми 1.0 при evaluation.
          </p>
          <Button variant="outline" size="sm" onClick={normalizeWeights}>
            Нормалізувати зараз
          </Button>
        </div>

        <div className="space-y-3">
          {selectedModels.map((m) => {
            const w = weights[m.id] ?? 0;
            const norm = sumWeights > 0 ? w / sumWeights : 0;
            const Icon = MODEL_ICONS[m.model_type] || Brain;
            return (
              <Card key={m.id}>
                <CardContent className="p-4 space-y-2">
                  <div className="flex items-center gap-3">
                    <div
                      className={cn(
                        "w-8 h-8 rounded-lg flex items-center justify-center shrink-0",
                        modelIconClasses(m.model_type)
                      )}
                    >
                      <Icon className="h-4 w-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-sm truncate">{m.name}</p>
                      <p className="text-[11px] text-muted-foreground">
                        F1: {formatPercent(m.f1_score)} · Acc:{" "}
                        {formatPercent(m.accuracy)}
                      </p>
                    </div>
                    <div className="text-right text-sm">
                      <div className="font-mono font-semibold text-primary">
                        {norm.toFixed(2)}
                      </div>
                      <div className="text-[10px] text-muted-foreground">
                        raw: {w.toFixed(2)}
                      </div>
                    </div>
                  </div>
                  <input
                    type="range"
                    min={0.01}
                    max={1}
                    step={0.01}
                    value={w}
                    onChange={(e) => updateWeight(m.id, Number(e.target.value))}
                    className="w-full accent-primary"
                  />
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    );
  };

  const renderSummary = () => (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label className="text-sm">Назва ансамблю</Label>
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Наприклад: NB + DistilBERT + LLM Haiku"
        />
      </div>

      <Card>
        <CardContent className="p-4 space-y-3 text-sm">
          <p className="font-semibold">Підсумок конфігурації</p>
          <p>
            <span className="font-medium">Voting:</span>{" "}
            {votingType.toUpperCase()}
          </p>
          <p>
            <span className="font-medium">Моделей:</span> {selectedModels.length}
          </p>
          <div className="space-y-2 pt-1 border-t border-border">
            {selectedModels.map((m) => {
              const Icon = MODEL_ICONS[m.model_type] || Brain;
              const w =
                votingType === "weighted" && sumWeights > 0
                  ? (weights[m.id] ?? 0) / sumWeights
                  : null;
              return (
                <div key={m.id} className="flex items-center gap-2 text-xs">
                  <div
                    className={cn(
                      "w-7 h-7 rounded-lg flex items-center justify-center shrink-0",
                      modelIconClasses(m.model_type)
                    )}
                  >
                    <Icon className="h-3.5 w-3.5" />
                  </div>
                  <span className="truncate flex-1">{m.name}</span>
                  {w != null && (
                    <span className="font-mono text-primary shrink-0">
                      {w.toFixed(2)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription className="text-xs">
          Після створення ансамбль одразу оцінюється на test set
          (зазвичай {"<"} 5 секунд — predictions кешовані).
        </AlertDescription>
      </Alert>
    </div>
  );

  const renderStep = () => {
    if (step === 0) return renderModelPicker();
    if (step === 1) return renderVotingPicker();
    if (step === 2) return renderWeights();
    return renderSummary();
  };

  const isLastStep = step === 3;

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Layers className="h-5 w-5" />
            Створення ансамблю
          </DialogTitle>
        </DialogHeader>

        {/* Numbered step indicator — як у ClassificationWizard */}
        <div className="flex items-center mb-2">
          {STEP_LABELS.map((label, i) => {
            const state = i < step ? "done" : i === step ? "active" : "pending";
            return (
              <div key={label} className="flex items-center flex-1 last:flex-none">
                <div className="flex flex-col items-center">
                  <div
                    className={cn(
                      "w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0",
                      state === "done" && "bg-green-500 text-white",
                      state === "active" && "bg-primary text-primary-foreground",
                      state === "pending" && "bg-muted text-muted-foreground"
                    )}
                  >
                    {state === "done" ? <Check className="h-4 w-4" /> : i + 1}
                  </div>
                  <span
                    className={cn(
                      "text-[10px] mt-1 whitespace-nowrap",
                      state === "active" && "text-primary font-semibold",
                      state === "pending" && "text-muted-foreground opacity-40",
                      state === "done" && "text-muted-foreground"
                    )}
                  >
                    {label}
                  </span>
                </div>
                {i < STEP_LABELS.length - 1 && (
                  <div
                    className={cn(
                      "h-0.5 flex-1 -mt-4",
                      i < step ? "bg-green-500" : "bg-muted"
                    )}
                  />
                )}
              </div>
            );
          })}
        </div>

        <div className="py-2">{renderStep()}</div>

        <DialogFooter>
          <div className="flex justify-between w-full gap-2">
            {step > 0 ? (
              <Button variant="outline" onClick={handleBack} disabled={submitting}>
                <ChevronLeft className="h-4 w-4 mr-1" />
                Назад
              </Button>
            ) : (
              <Button variant="outline" onClick={onClose} disabled={submitting}>
                Скасувати
              </Button>
            )}

            {!isLastStep ? (
              <Button
                onClick={handleNext}
                disabled={
                  (step === 0 && !canProceedStep0) ||
                  (step === 2 && !canProceedStep2) ||
                  loading
                }
              >
                Далі
                <ChevronRight className="h-4 w-4 ml-1" />
              </Button>
            ) : (
              <Button onClick={handleSubmit} disabled={!canSubmit || submitting}>
                {submitting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Створення…
                  </>
                ) : (
                  "Створити ансамбль"
                )}
              </Button>
            )}
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
