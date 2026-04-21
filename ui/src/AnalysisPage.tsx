import React, { useEffect, useState } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Textarea } from "./components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { AlertTriangle, CheckCircle2, Loader2, Search } from "lucide-react";
import { toast } from "sonner";
import type { ModelRecord } from "./types";

interface AnalyzeResult {
  label: "FAKE" | "REAL";
  confidence: number;
  probability: number;
}

interface AnalysisPageProps {
  onDeepCheckRequest?: (text: string) => void;
}

export default function AnalysisPage({ onDeepCheckRequest }: AnalysisPageProps = {}) {
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState<number | null>(null);
  const [text, setText] = useState<string>("");
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loadingModels, setLoadingModels] = useState<boolean>(true);

  useEffect(() => {
    api.get("/models")
      .then(({ data }) => {
        setModels(data);
        if (data.length > 0) setModelId(data[0].id);
      })
      .catch(() => {})
      .finally(() => setLoadingModels(false));
  }, []);

  const handleSubmit = async () => {
    if (!text.trim()) { toast.error("Введіть текст для аналізу"); return; }
    if (!modelId) { toast.error("Оберіть модель"); return; }

    setLoading(true);
    setResult(null);
    try {
      const { data } = await api.post("/analyze", { text, model_id: modelId });
      setResult(data);
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      toast.error(typeof detail === "string" ? detail : "Помилка під час аналізу");
    } finally {
      setLoading(false);
    }
  };

  const isFake = result?.label === "FAKE";
  const selectedModel = models.find((m) => m.id === modelId);

  return (
    <div className="space-y-6 max-w-2xl mx-auto">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Аналіз тексту</h2>
        <p className="text-muted-foreground">Перевірте текст на ознаки дезінформації</p>
      </div>

      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Модель</CardTitle>
        </CardHeader>
        <CardContent>
          {loadingModels ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Завантаження моделей...
            </div>
          ) : models.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Навчених моделей немає. Спочатку навчіть модель на вкладці "Навчання моделі".
            </p>
          ) : (
            <div className="space-y-2">
              <Select
                value={modelId?.toString() || ""}
                onValueChange={(v) => { setModelId(Number(v)); setResult(null); }}
              >
                <SelectTrigger className="max-w-sm">
                  <SelectValue placeholder="Оберіть модель" />
                </SelectTrigger>
                <SelectContent>
                  {models.map((m) => (
                    <SelectItem key={m.id} value={m.id.toString()}>
                      {m.name || m.model_type}
                      {m.accuracy != null ? ` — ${(m.accuracy * 100).toFixed(1)}%` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedModel && (
                <p className="text-xs text-muted-foreground">
                  Тип: <strong>{selectedModel.model_type}</strong>
                  {selectedModel.accuracy != null && <> · Accuracy: <strong>{(selectedModel.accuracy * 100).toFixed(1)}%</strong></>}
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Text Input */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Текст для аналізу</CardTitle>
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
            {loading ? "Аналіз..." : "Аналізувати"}
          </Button>
        </CardContent>
      </Card>

      {/* Verdict */}
      {result && (
        <>
          <Card className={cn(
            "border-2",
            isFake ? "border-red-500 bg-red-50 dark:bg-red-950/20" : "border-green-500 bg-green-50 dark:bg-green-950/20",
          )}>
            <CardContent className="py-8 text-center">
              {isFake ? (
                <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-3" />
              ) : (
                <CheckCircle2 className="h-12 w-12 text-green-500 mx-auto mb-3" />
              )}
              <h2 className={cn(
                "text-3xl font-bold",
                isFake ? "text-red-600 dark:text-red-400" : "text-green-600 dark:text-green-400"
              )}>
                {isFake ? "ДЕЗІНФОРМАЦІЯ" : "ДОСТОВІРНО"}
              </h2>
              <p className="text-muted-foreground mt-2">
                Впевненість: {(result.confidence * 100).toFixed(1)}%
              </p>
            </CardContent>
          </Card>

          <Card className="border-dashed">
            <CardContent className="py-4">
              <div className="flex items-start gap-3">
                <Search className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                <div className="flex-1 space-y-2">
                  <p className="text-sm font-medium">
                    Потрібна детальніша перевірка?
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Запустити multi-hop верифікацію: витягування тверджень, пошук доказів
                    у новинах та соцмережах, аналіз консистентності.
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      if (onDeepCheckRequest) {
                        onDeepCheckRequest(text);
                      } else {
                        navigator.clipboard.writeText(text);
                        toast.info("Текст скопійовано. Вставте у вкладці 'Верифікація'.");
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
        </>
      )}

      {/* Details */}
      {result && (
        <Card>
          <CardContent className="p-4">
            <ul className="space-y-2 text-sm">
              <li className="flex justify-between py-1 border-b border-border">
                <span className="font-medium">Модель</span>
                <span className="text-muted-foreground">{selectedModel?.name || selectedModel?.model_type}</span>
              </li>
              <li className="flex justify-between py-1">
                <span className="font-medium">Ймовірність FAKE</span>
                <span className="text-muted-foreground">{(result.probability * 100).toFixed(1)}%</span>
              </li>
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
