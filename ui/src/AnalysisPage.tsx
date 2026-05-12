// ui/src/AnalysisPage.tsx
import React, { useEffect, useMemo, useState } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Textarea } from "./components/ui/textarea";
import { Badge } from "./components/ui/badge";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import {
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Loader2,
  Search,
  Brain,
  Sparkles,
  Cpu,
} from "lucide-react";
import { toast } from "sonner";
import type { ModelRecord } from "./types";

interface AnalyzeResult {
  label: "FAKE" | "REAL" | "UNCERTAIN";
  confidence: number;
  probability: number | null;
  reason?: string;
  base_model_used?: string;
  mode?: string;
}

interface AnalysisPageProps {
  onDeepCheckRequest?: (text: string) => void;
}

// Icon per model type
const TYPE_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  nb: Cpu,
  deberta: Brain,
  distilbert: Brain,
  llm: Sparkles,
  gin: Brain,
  sage: Brain,
};

const TYPE_LABELS: Record<string, string> = {
  nb: "Naive Bayes",
  deberta: "DeBERTa",
  distilbert: "DistilBERT",
  llm: "LLM (Gemini)",
  gin: "GIN",
  sage: "GraphSAGE",
};

export default function AnalysisPage({ onDeepCheckRequest }: AnalysisPageProps = {}) {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState<number | null>(null);
  const [text, setText] = useState<string>("");
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingModels, setLoadingModels] = useState<boolean>(true);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = () => {
    setLoadingModels(true);
    api
      .get<ModelRecord[]>("/models")
      .then(({ data }) => {
        setModels(data);
        // Auto-select first trained model (prefer non-LLM since it has accuracy),
        // else first LLM
        const firstTrained = data.find((m) => m.model_type !== "llm");
        const firstLLM = data.find((m) => m.model_type === "llm");
        const initial = firstTrained || firstLLM;
        if (initial) setModelId(initial.id);
      })
      .catch(() => { })
      .finally(() => setLoadingModels(false));
  };

  // Group models by type (trained ML vs LLM presets)
  const { trainedModels, llmPresets } = useMemo(() => {
    const trained = models.filter((m) => m.model_type !== "llm");
    const llm = models.filter((m) => m.model_type === "llm");
    return { trainedModels: trained, llmPresets: llm };
  }, [models]);

  const selectedModel = models.find((m) => m.id === modelId);
  const isLLMSelected = selectedModel?.model_type === "llm";

  const handleSubmit = async () => {
    if (!text.trim()) {
      toast.error("Введіть текст для аналізу");
      return;
    }
    if (!modelId) {
      toast.error("Оберіть модель");
      return;
    }

    setLoading(true);
    setResult(null);
    try {
      const { data } = await api.post<AnalyzeResult>("/analyze", {
        text,
        model_id: modelId,
      });
      setResult(data);

      if (data.label === "UNCERTAIN") {
        toast.warning("Модель не змогла дати впевнену оцінку");
      }
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      toast.error(typeof detail === "string" ? detail : "Помилка під час аналізу");
    } finally {
      setLoading(false);
    }
  };

  const isFake = result?.label === "FAKE";
  const isUncertain = result?.label === "UNCERTAIN";

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Аналіз тексту</h2>
        <p className="text-muted-foreground">
          Перевірте текст на ознаки дезінформації за допомогою обраної моделі
        </p>
      </div>

      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Модель</CardTitle>
          <CardDescription>
            Оберіть натреновану модель або LLM-пресет для класифікації
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loadingModels ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Завантаження моделей...
            </div>
          ) : models.length === 0 ? (
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>Немає доступних моделей.</p>
              <p className="text-xs">
                Натренуйте модель у розділі <strong>"Навчання моделі"</strong> або створіть{" "}
                <strong>LLM пресет</strong>.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <Select
                value={modelId?.toString() || ""}
                onValueChange={(v) => {
                  setModelId(Number(v));
                  setResult(null);
                }}
              >
                <SelectTrigger className="max-w-md">
                  <SelectValue placeholder="Оберіть модель" />
                </SelectTrigger>
                <SelectContent>
                  {trainedModels.length > 0 && (
                    <SelectGroup>
                      <SelectLabel className="flex items-center gap-1.5 text-xs">
                        <Brain className="h-3 w-3" />
                        Натреновані моделі
                      </SelectLabel>
                      {trainedModels.map((m) => {
                        const Icon = TYPE_ICONS[m.model_type] || Cpu;
                        return (
                          <SelectItem key={m.id} value={m.id.toString()}>
                            <span className="flex items-center gap-2">
                              <Icon className="h-3.5 w-3.5 shrink-0" />
                              <span className="truncate">{m.name || m.model_type}</span>
                              {m.accuracy != null && (
                                <span className="text-xs text-muted-foreground">
                                  {(m.accuracy * 100).toFixed(1)}%
                                </span>
                              )}
                            </span>
                          </SelectItem>
                        );
                      })}
                    </SelectGroup>
                  )}

                  {trainedModels.length > 0 && llmPresets.length > 0 && <SelectSeparator />}

                  {llmPresets.length > 0 && (
                    <SelectGroup>
                      <SelectLabel className="flex items-center gap-1.5 text-xs">
                        <Sparkles className="h-3 w-3" />
                        LLM пресети
                      </SelectLabel>
                      {llmPresets.map((m) => (
                        <SelectItem key={m.id} value={m.id.toString()}>
                          <span className="flex items-center gap-2">
                            <Sparkles className="h-3.5 w-3.5 shrink-0" />
                            <span className="truncate">{m.name}</span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  )}
                </SelectContent>
              </Select>

              {selectedModel && (
                <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <Badge variant="secondary" className="font-normal">
                    {TYPE_LABELS[selectedModel.model_type] || selectedModel.model_type}
                  </Badge>
                  {selectedModel.accuracy != null && (
                    <span>
                      Accuracy: <strong>{(selectedModel.accuracy * 100).toFixed(1)}%</strong>
                    </span>
                  )}
                  {isLLMSelected && (
                    <span className="italic">
                      LLM — результат без числових метрик (не тренувалась на цьому датасеті)
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Text Input */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Текст для аналізу</CardTitle>
          <CardDescription>
            Вставте текст новини, посту або заголовок — модель оцінить ймовірність дезінформації
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            rows={7}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Вставте текст новини для перевірки..."
          />
          <Button
            className="w-full"
            size="lg"
            onClick={handleSubmit}
            disabled={loading || models.length === 0}
          >
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {loading
              ? isLLMSelected
                ? "LLM думає..."
                : "Аналіз..."
              : "Аналізувати"}
          </Button>
        </CardContent>
      </Card>

      {/* Verdict */}
      {result && (
        <>
          <Card
            className={cn(
              "border-2",
              isFake
                ? "border-red-500 bg-red-50 dark:bg-red-950/20"
                : isUncertain
                  ? "border-amber-500 bg-amber-50 dark:bg-amber-950/20"
                  : "border-green-500 bg-green-50 dark:bg-green-950/20",
            )}
          >
            <CardContent className="py-8 text-center">
              {isFake ? (
                <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-3" />
              ) : isUncertain ? (
                <HelpCircle className="h-12 w-12 text-amber-500 mx-auto mb-3" />
              ) : (
                <CheckCircle2 className="h-12 w-12 text-green-500 mx-auto mb-3" />
              )}
              <h2
                className={cn(
                  "text-3xl font-bold",
                  isFake
                    ? "text-red-600 dark:text-red-400"
                    : isUncertain
                      ? "text-amber-600 dark:text-amber-400"
                      : "text-green-600 dark:text-green-400",
                )}
              >
                {isFake
                  ? "ДЕЗІНФОРМАЦІЯ"
                  : isUncertain
                    ? "НЕВИЗНАЧЕНО"
                    : "ДОСТОВІРНО"}
              </h2>
              {!isUncertain && (
                <p className="text-muted-foreground mt-2">
                  Впевненість: {(result.confidence * 100).toFixed(1)}%
                </p>
              )}
              {isUncertain && result.reason && (
                <p className="text-sm text-amber-700 dark:text-amber-400 mt-3 max-w-md mx-auto italic">
                  "{result.reason}"
                </p>
              )}
            </CardContent>
          </Card>

          {/* LLM reasoning (якщо модель дала reason) */}
          {!isUncertain && result.reason && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Sparkles className="h-5 w-5" />
                  Обґрунтування моделі
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground italic leading-relaxed">
                  "{result.reason}"
                </p>
              </CardContent>
            </Card>
          )}

          {/* Cross-link to deep verification */}
          <Card className="border-dashed">
            <CardContent className="py-4">
              <div className="flex items-start gap-3">
                <Search className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                <div className="flex-1 space-y-2">
                  <p className="text-sm font-medium">Потрібна детальніша перевірка?</p>
                  <p className="text-xs text-muted-foreground">
                    Запустити multi-hop верифікацію: витягування тверджень, пошук доказів у
                    новинах та соцмережах, аналіз консистентності.
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (onDeepCheckRequest) {
                        onDeepCheckRequest(text);
                      } else {
                        navigator.clipboard.writeText(text);
                        toast.info(
                          "Текст скопійовано. Вставте у вкладці 'Верифікація'.",
                        );
                      }
                    }}
                  >
                    <Search className="mr-2 h-4 w-4" />
                    Перевірити детальніше →
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Details */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Деталі</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-1.5 text-sm">
                <li className="flex justify-between py-1 border-b border-border">
                  <span className="font-medium">Модель</span>
                  <span className="text-muted-foreground text-right">
                    {selectedModel?.name || selectedModel?.model_type}
                  </span>
                </li>
                {result.probability != null && (
                  <li className="flex justify-between py-1 border-b border-border">
                    <span className="font-medium">Ймовірність FAKE</span>
                    <span className="text-muted-foreground">
                      {(result.probability * 100).toFixed(1)}%
                    </span>
                  </li>
                )}
                {result.base_model_used && (
                  <li className="flex justify-between py-1 border-b border-border">
                    <span className="font-medium">Базова модель</span>
                    <span className="text-muted-foreground">{result.base_model_used}</span>
                  </li>
                )}
                {result.mode && (
                  <li className="flex justify-between py-1">
                    <span className="font-medium">Режим</span>
                    <Badge variant="outline" className="text-xs">
                      {result.mode}
                    </Badge>
                  </li>
                )}
              </ul>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}