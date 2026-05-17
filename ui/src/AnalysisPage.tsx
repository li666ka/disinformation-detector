// ui/src/AnalysisPage.tsx
import React, { useEffect, useMemo, useState } from "react";
import api from "./api";
import { toast } from "sonner";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Label } from "./components/ui/label";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "./components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import { Checkbox } from "./components/ui/checkbox";
import { Switch } from "./components/ui/switch";
import { Alert, AlertDescription } from "./components/ui/alert";
import {
  FileText,
  Link as LinkIcon,
  Search,
  Loader2,
  Sparkles,
  Brain,
  Cpu,
  Globe,
  Settings2,
  Wand2,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";
import { AnalysisResultPanel } from "./components/AnalysisResultPanel";
import type {
  AnalyzeInputMode,
  AnalyzeV2Request,
  AnalyzeV2Response,
  ModelRecord,
} from "./types";

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
  llm: "LLM (Claude)",
  gin: "GIN",
  sage: "GraphSAGE",
};

const EXCLUDED_TYPES = new Set(["gin", "sage", "gnn"]);
const SOURCES = ["mastodon", "bluesky", "rss"] as const;

export default function AnalysisPage() {
  // Models + selection
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState<number | "auto">("auto");
  const [loadingModels, setLoadingModels] = useState<boolean>(true);

  // Request state
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AnalyzeV2Response | null>(null);

  // Per-mode input
  const [mode, setMode] = useState<AnalyzeInputMode>("text");
  const [textInput, setTextInput] = useState<string>("");
  const [urlInput, setUrlInput] = useState<string>("");
  const [claimInput, setClaimInput] = useState<string>("");

  // Options
  const [extractClaim, setExtractClaim] = useState<boolean>(true);
  const [factCheck, setFactCheck] = useState<boolean>(false);
  const [classifyExtracted, setClassifyExtracted] = useState<boolean>(false);
  const [searchSources, setSearchSources] = useState<Set<string>>(
    new Set(SOURCES),
  );
  const [searchLimit, setSearchLimit] = useState<number>(20);

  // ML server health (Colab ngrok tunnel часто падає — показуємо banner)
  type MLStatus = {
    ok: boolean;
    url_set: boolean;
    reachable: boolean;
    ready: boolean;
    detail: string;
    checked_url: string | null;
  };
  const [mlStatus, setMlStatus] = useState<MLStatus | null>(null);

  const refreshMlStatus = (force = false) => {
    api
      .get<MLStatus>(`/ml-server/status${force ? "?force=true" : ""}`)
      .then(({ data }) => setMlStatus(data))
      .catch(() => setMlStatus(null));
  };

  useEffect(() => {
    setLoadingModels(true);
    api
      .get<ModelRecord[]>("/models")
      .then(({ data }) => setModels(data))
      .catch(() => toast.error("Не вдалося завантажити моделі"))
      .finally(() => setLoadingModels(false));

    refreshMlStatus();
  }, []);

  // Чи обрана модель потребує Colab? Для LLM presets — ні, для всіх інших — так.
  const selectedModel =
    modelId === "auto" ? null : models.find((m) => m.id === modelId) || null;
  const needsColab =
    !selectedModel || (selectedModel.model_type !== "llm");

  const mlBlocking = needsColab && mlStatus !== null && !mlStatus.ok;

  const { realWorldModels, limitedModels } = useMemo(() => {
    const realWorld: ModelRecord[] = [];
    const limited: ModelRecord[] = [];
    for (const m of models) {
      (EXCLUDED_TYPES.has(m.model_type) ? limited : realWorld).push(m);
    }
    return { realWorldModels: realWorld, limitedModels: limited };
  }, [models]);

  const currentInput =
    mode === "text" ? textInput : mode === "url" ? urlInput : claimInput;
  const inputValid = currentInput.trim().length >= 3;
  const searchValid = mode !== "claim_search" || searchSources.size > 0;

  const handleAnalyze = async () => {
    if (!inputValid) {
      toast.error(
        mode === "text"
          ? "Введіть текст"
          : mode === "url"
          ? "Введіть URL"
          : "Введіть твердження",
      );
      return;
    }
    if (!searchValid) {
      toast.error("Оберіть хоча б одне джерело для пошуку");
      return;
    }

    setLoading(true);
    setResult(null);

    const req: AnalyzeV2Request = {
      input_mode: mode,
      input: currentInput.trim(),
      model_id: modelId === "auto" ? null : modelId,
      options: {
        extract_claim: extractClaim,
        classify: true,
        fact_check: factCheck,
        classify_extracted: classifyExtracted,
        search_sources: Array.from(searchSources),
        search_limit: searchLimit,
      },
    };

    try {
      const { data } = await api.post<AnalyzeV2Response>("/analyze/v2", req);
      setResult(data);
      if (data.classification?.label === "UNCERTAIN") {
        toast.warning("Модель не змогла дати впевнену оцінку");
      }
    } catch (err: any) {
      const body = err?.response?.data;
      if (body?.error === "ml_server_offline" || body?.error === "ml_server_not_ready") {
        // 503 з нашого ml_client → оновити banner і показати message як є
        refreshMlStatus(true);
        toast.error(body.message || "ML server недоступний");
      } else {
        const detail = body?.detail;
        toast.error(
          typeof detail === "string" ? detail : "Помилка під час аналізу",
        );
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSwitchMode = (v: string) => {
    setMode(v as AnalyzeInputMode);
    setResult(null);
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Аналіз тексту</h2>
        <p className="text-muted-foreground">
          Перевірте текст, пост за URL, або поширення твердження у соцмережах
        </p>
      </div>

      {/* ML server status banner — показуємо лише коли offline і обрана модель потребує Colab */}
      {mlBlocking && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription className="flex items-start justify-between gap-3 flex-wrap">
            <div>
              <strong>ML server offline.</strong>{" "}
              Запусти Colab notebook і онови <code>ML_SERVER_URL</code> (або
              <code> COLAB_NGROK_URL</code>) у <code>.env</code>, тоді перезапусти FastAPI.
              {mlStatus?.detail && (
                <div className="text-xs mt-1 opacity-80">
                  Деталі: {mlStatus.detail}
                  {mlStatus.checked_url ? ` (${mlStatus.checked_url})` : ""}
                </div>
              )}
              <div className="text-xs mt-1 opacity-80">
                LLM-пресети (Claude) продовжать працювати — для них Colab не потрібен.
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refreshMlStatus(true)}
              className="shrink-0"
            >
              <RefreshCw className="h-3.5 w-3.5 mr-1.5" />
              Перевірити
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Mode selector */}
      <Tabs value={mode} onValueChange={handleSwitchMode}>
        <TabsList className="grid grid-cols-3 max-w-2xl">
          <TabsTrigger value="text" className="gap-1.5">
            <FileText className="h-4 w-4" />
            Текст
          </TabsTrigger>
          <TabsTrigger value="url" className="gap-1.5">
            <LinkIcon className="h-4 w-4" />
            За URL
          </TabsTrigger>
          <TabsTrigger value="claim_search" className="gap-1.5">
            <Search className="h-4 w-4" />
            Поширення
          </TabsTrigger>
        </TabsList>

        <TabsContent value="text" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Звичайний текст</CardTitle>
              <CardDescription>
                Вставте текст / статтю / твіт — отримаєте verdict моделі +
                extraction
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="Вставте текст тут (мінімум 10 символів)…"
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                rows={6}
                className="resize-y"
              />
              <p className="text-xs text-muted-foreground mt-2">
                {textInput.length} символів
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="url" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Пост за посиланням</CardTitle>
              <CardDescription>
                URL поста з Mastodon, Bluesky або RSS-статті — fetch + аналіз
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Input
                placeholder="https://mastodon.social/@user/123… або https://bsky.app/profile/…/post/…"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !loading) handleAnalyze();
                }}
              />
              <div className="text-xs text-muted-foreground mt-2">
                Підтримуються Mastodon (@user/id), Bluesky
                (profile/handle/post/id), RSS articles.
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="claim_search" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Wand2 className="h-4 w-4" />
                Аналіз поширення твердження
              </CardTitle>
              <CardDescription>
                Введіть твердження → знайдемо схожі пости у соцмережах →
                класифікуємо кожен → агрегований verdict
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label>Твердження для перевірки</Label>
                <Textarea
                  placeholder='Наприклад: "COVID vaccine causes infertility"'
                  value={claimInput}
                  onChange={(e) => setClaimInput(e.target.value)}
                  rows={3}
                  className="mt-1.5 resize-y"
                />
              </div>

              <div className="space-y-3 pt-2 border-t">
                <Label className="text-xs">Параметри пошуку</Label>

                <div className="flex flex-wrap gap-3">
                  {SOURCES.map((src) => (
                    <label
                      key={src}
                      className="flex items-center gap-2 cursor-pointer"
                    >
                      <Checkbox
                        checked={searchSources.has(src)}
                        onCheckedChange={(checked) => {
                          setSearchSources((prev) => {
                            const next = new Set(prev);
                            if (checked) next.add(src);
                            else next.delete(src);
                            return next;
                          });
                        }}
                      />
                      <span className="text-sm capitalize">{src}</span>
                    </label>
                  ))}
                </div>

                <div className="flex items-center gap-3">
                  <Label htmlFor="search-limit" className="text-sm">
                    Максимум постів:
                  </Label>
                  <Input
                    id="search-limit"
                    type="number"
                    min={5}
                    max={50}
                    value={searchLimit}
                    onChange={(e) =>
                      setSearchLimit(
                        Math.max(
                          5,
                          Math.min(50, Number(e.target.value) || 20),
                        ),
                      )
                    }
                    className="w-20"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Settings2 className="h-4 w-4" />
            Налаштування
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Model */}
          <div className="space-y-1.5">
            <Label>Модель класифікації</Label>
            <Select
              value={modelId === "auto" ? "auto" : modelId.toString()}
              onValueChange={(v) =>
                setModelId(v === "auto" ? "auto" : Number(v))
              }
            >
              <SelectTrigger className="max-w-md">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">
                  <span className="flex items-center gap-2">
                    <Wand2 className="h-3.5 w-3.5" />
                    <span>Auto (найкраща для real-world)</span>
                  </span>
                </SelectItem>

                {realWorldModels.length > 0 && (
                  <SelectGroup>
                    <SelectLabel className="text-xs">Рекомендовані</SelectLabel>
                    {realWorldModels.map((m) => {
                      const Icon = TYPE_ICONS[m.model_type] || Cpu;
                      return (
                        <SelectItem key={m.id} value={m.id.toString()}>
                          <span className="flex items-center gap-2">
                            <Icon className="h-3.5 w-3.5 shrink-0" />
                            <span className="truncate">
                              {m.name || TYPE_LABELS[m.model_type]}
                            </span>
                            {m.f1_score != null && (
                              <span className="text-xs text-muted-foreground">
                                F1={m.f1_score.toFixed(3)}
                              </span>
                            )}
                          </span>
                        </SelectItem>
                      );
                    })}
                  </SelectGroup>
                )}

                {limitedModels.length > 0 && (
                  <SelectGroup>
                    <SelectLabel className="text-xs">
                      Потребують cascade
                    </SelectLabel>
                    {limitedModels.map((m) => {
                      const Icon = TYPE_ICONS[m.model_type] || Brain;
                      return (
                        <SelectItem key={m.id} value={m.id.toString()}>
                          <span className="flex items-center gap-2 opacity-75">
                            <Icon className="h-3.5 w-3.5 shrink-0" />
                            <span className="truncate">{m.name}</span>
                          </span>
                        </SelectItem>
                      );
                    })}
                  </SelectGroup>
                )}
              </SelectContent>
            </Select>
            {selectedModel && (
              <p className="text-xs text-muted-foreground">
                Тип:{" "}
                {TYPE_LABELS[selectedModel.model_type] || selectedModel.model_type}
                {selectedModel.f1_score != null &&
                  ` · F1=${selectedModel.f1_score.toFixed(3)}`}
              </p>
            )}
          </div>

          {/* Options */}
          <div className="space-y-2 pt-3 border-t">
            <Label className="text-xs">Опції</Label>
            <div className="space-y-2">
              <label className="flex items-center gap-2 cursor-pointer text-sm">
                <Switch
                  checked={extractClaim}
                  onCheckedChange={setExtractClaim}
                />
                <Brain className="h-3.5 w-3.5 text-blue-600" />
                <span>LLM extract claim + stance</span>
              </label>

              {extractClaim && (
                <label className="flex items-center gap-2 cursor-pointer text-sm ml-8">
                  <Switch
                    checked={classifyExtracted}
                    onCheckedChange={setClassifyExtracted}
                  />
                  <span className="text-xs">
                    Класифікувати extracted claim замість raw text
                  </span>
                </label>
              )}

              <label className="flex items-center gap-2 cursor-pointer text-sm">
                <Switch checked={factCheck} onCheckedChange={setFactCheck} />
                <Globe className="h-3.5 w-3.5 text-amber-600" />
                <span>Fact-check проти Google</span>
              </label>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Submit */}
      <div className="flex flex-wrap items-center gap-2">
        <Button
          onClick={handleAnalyze}
          disabled={loading || !inputValid || !searchValid || loadingModels || mlBlocking}
          size="lg"
          className="w-full sm:w-auto"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Аналізую…
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4 mr-2" />
              Аналізувати
            </>
          )}
        </Button>
      </div>

      {/* Result / empty state */}
      {result ? (
        <AnalysisResultPanel result={result} />
      ) : (
        !loading && (
          <Card className="border-dashed">
            <CardContent className="py-12 text-center">
              <FileText className="h-12 w-12 mx-auto text-muted-foreground mb-3" />
              <p className="text-sm text-muted-foreground">
                {mode === "text" && "Вставте текст і натисніть «Аналізувати»"}
                {mode === "url" && "Введіть URL поста і натисніть «Аналізувати»"}
                {mode === "claim_search" &&
                  "Введіть твердження для аналізу поширення"}
              </p>
            </CardContent>
          </Card>
        )
      )}
    </div>
  );
}
