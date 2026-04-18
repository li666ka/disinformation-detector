import React, { useEffect, useState, useCallback } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Label } from "./components/ui/label";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "./components/ui/table";
import {
  Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle,
} from "./components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "./components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import {
  Sparkles, Plus, Trash2, Loader2, CheckCircle2, X, Download, AlertTriangle,
} from "lucide-react";
import { toast } from "sonner";
import type {
  ModelRecord, LLMMode, FewShotExample,
  LLMPresetDefaults, LLMPresetTestResponse,
} from "./types";

const MODE_LABELS: Record<LLMMode, string> = {
  zero_shot: "Zero-shot",
  few_shot: "Few-shot",
  cot: "Chain-of-Thought",
  bagging: "Bagging",
};

function parseLlmConfig(json: string | null | undefined): Record<string, any> | null {
  if (!json) return null;
  try { return JSON.parse(json); } catch { return null; }
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("uk-UA");
}

export default function LLMPresetsPage() {
  const [presets, setPresets] = useState<ModelRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [defaults, setDefaults] = useState<LLMPresetDefaults | null>(null);

  // Create modal state
  const [createOpen, setCreateOpen] = useState(false);
  const [name, setName] = useState("");
  const [baseModel, setBaseModel] = useState("gemini-2.5-flash-lite");
  const [mode, setMode] = useState<LLMMode>("zero_shot");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [temperature, setTemperature] = useState(0.0);
  const [maxTokens, setMaxTokens] = useState(200);
  const [fewShotExamples, setFewShotExamples] = useState<FewShotExample[]>([]);
  const [cotInstruction, setCotInstruction] = useState("");
  const [baggingNCalls, setBaggingNCalls] = useState(3);
  const [saving, setSaving] = useState(false);

  // Inline test
  const [testText, setTestText] = useState("");
  const [testResult, setTestResult] = useState<LLMPresetTestResponse | null>(null);
  const [testing, setTesting] = useState(false);

  // Few-shot: add manual
  const [manualText, setManualText] = useState("");
  const [manualLabel, setManualLabel] = useState<"FAKE" | "REAL">("FAKE");

  // Few-shot: random samples
  const [nFake, setNFake] = useState(2);
  const [nReal, setNReal] = useState(2);
  const [loadingRandom, setLoadingRandom] = useState(false);

  const fetchPresets = useCallback(async () => {
    try {
      const { data } = await api.get<ModelRecord[]>("/models");
      setPresets(data.filter((m) => m.model_type === "llm"));
    } catch {
      toast.error("Не вдалося завантажити пресети");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchPresets(); }, [fetchPresets]);

  const fetchDefaults = async () => {
    try {
      const { data } = await api.get<LLMPresetDefaults>("/llm-presets/defaults");
      setDefaults(data);
      return data;
    } catch {
      toast.error("Не вдалося завантажити налаштування за замовчуванням");
      return null;
    }
  };

  const openCreateModal = async () => {
    const defs = defaults || await fetchDefaults();
    if (defs) {
      setSystemPrompt(defs.default_system_prompt);
      setCotInstruction(defs.default_cot_instruction);
      setBaseModel(defs.base_models[0] || "gemini-2.5-flash-lite");
      setTemperature(defs.default_temperature);
      setMaxTokens(defs.default_max_output_tokens);
      setBaggingNCalls(defs.default_bagging_n_calls);
    }
    setName("");
    setMode("zero_shot");
    setFewShotExamples([]);
    setTestText("");
    setTestResult(null);
    setManualText("");
    setCreateOpen(true);
  };

  const handleSave = async () => {
    if (!name.trim()) {
      toast.error("Вкажіть назву пресету");
      return;
    }
    setSaving(true);
    try {
      const payload: any = {
        name: name.trim(),
        base_model: baseModel,
        mode,
        system_prompt: systemPrompt || null,
        temperature,
        max_output_tokens: maxTokens,
      };
      if (mode === "few_shot" && fewShotExamples.length > 0) {
        payload.few_shot_examples = fewShotExamples;
      }
      if (mode === "cot" && cotInstruction) {
        payload.cot_instruction = cotInstruction;
      }
      if (mode === "bagging") {
        payload.bagging_n_calls = baggingNCalls;
      }
      await api.post("/llm-presets", payload);
      await fetchPresets();
      setCreateOpen(false);
      toast.success("Пресет збережено");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка збереження пресету");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm("Видалити пресет?")) return;
    try {
      await api.delete(`/llm-presets/${id}`);
      await fetchPresets();
      toast.success("Пресет видалено");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка видалення");
    }
  };

  const handleTest = async () => {
    if (!testText.trim()) {
      toast.error("Введіть тестовий текст");
      return;
    }
    setTesting(true);
    setTestResult(null);
    try {
      const payload: any = {
        base_model: baseModel,
        mode,
        system_prompt: systemPrompt || null,
        temperature,
        max_output_tokens: maxTokens,
        test_text: testText.trim(),
      };
      if (mode === "few_shot" && fewShotExamples.length > 0) {
        payload.few_shot_examples = fewShotExamples;
      }
      if (mode === "cot" && cotInstruction) {
        payload.cot_instruction = cotInstruction;
      }
      if (mode === "bagging") {
        payload.bagging_n_calls = baggingNCalls;
      }
      const { data } = await api.post<LLMPresetTestResponse>("/llm-presets/test", payload);
      setTestResult(data);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка тестування");
    } finally {
      setTesting(false);
    }
  };

  const handleLoadRandomSamples = async () => {
    setLoadingRandom(true);
    try {
      const { data } = await api.post<{ examples: FewShotExample[] }>("/llm-presets/random-samples", {
        n_fake: nFake,
        n_real: nReal,
      });
      setFewShotExamples((prev) => [...prev, ...data.examples]);
      toast.success(`Додано ${data.examples.length} прикладів`);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Не вдалося завантажити приклади. Перевірте підключення до ML-сервера.");
    } finally {
      setLoadingRandom(false);
    }
  };

  const addManualExample = () => {
    if (!manualText.trim() || manualText.trim().length < 5) {
      toast.error("Мін. 5 символів для прикладу");
      return;
    }
    setFewShotExamples((prev) => [...prev, { text: manualText.trim(), label: manualLabel }]);
    setManualText("");
  };

  const removeExample = (idx: number) => {
    setFewShotExamples((prev) => prev.filter((_, i) => i !== idx));
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
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">LLM Пресети</h2>
          <p className="text-muted-foreground">Конфігурації для класифікації через Gemini API</p>
        </div>
        <Button size="sm" onClick={openCreateModal}>
          <Plus className="mr-2 h-4 w-4" /> Новий пресет
        </Button>
      </div>

      {/* Content */}
      {presets.length === 0 ? (
        <div className="text-center py-16">
          <Sparkles className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">Пресетів ще немає</h3>
          <p className="text-muted-foreground mb-4">
            Створіть конфігурацію для класифікації тексту через LLM
          </p>
          <Button onClick={openCreateModal}>
            <Plus className="mr-2 h-4 w-4" /> Новий пресет
          </Button>
        </div>
      ) : (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Назва</TableHead>
                <TableHead>Модель</TableHead>
                <TableHead>Режим</TableHead>
                <TableHead className="text-center">Accuracy</TableHead>
                <TableHead className="text-center">Температура</TableHead>
                <TableHead>Дата</TableHead>
                <TableHead className="w-10" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {presets.map((p) => {
                const cfg = parseLlmConfig(p.llm_config);
                return (
                  <TableRow key={p.id} className={cn(p.is_active && "bg-primary/5")}>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-amber-100 dark:bg-amber-950 text-amber-600">
                          <Sparkles className="h-4 w-4" />
                        </div>
                        <span className="font-medium text-sm">{p.name}</span>
                      </div>
                    </TableCell>
                    <TableCell className="text-sm">
                      {cfg?.base_model || "?"}
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary" className="text-xs">
                        {cfg?.mode ? MODE_LABELS[cfg.mode as LLMMode] || cfg.mode : "?"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-center">
                      {p.accuracy != null ? (
                        <span className="font-medium text-green-600 dark:text-green-400">
                          {(p.accuracy * 100).toFixed(1)}%
                        </span>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </TableCell>
                    <TableCell className="text-center text-sm">
                      {cfg?.temperature ?? "?"}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDate(p.created_at)}
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 text-destructive hover:text-destructive"
                        onClick={() => handleDelete(p.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}

      {/* ── Create Preset Dialog ───────────────────────────────────────── */}
      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Створити LLM пресет</DialogTitle>
          </DialogHeader>

          <div className="space-y-4">
            {/* Common fields */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2 col-span-2">
                <Label>Назва *</Label>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Наприклад: Gemini Zero-shot v1"
                />
              </div>
              <div className="space-y-2">
                <Label>Базова модель</Label>
                <Select value={baseModel} onValueChange={setBaseModel}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {(defaults?.base_models || ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]).map((m) => (
                      <SelectItem key={m} value={m}>{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-2">
                  <Label>Температура</Label>
                  <Input
                    type="number"
                    min={0} max={2} step={0.1}
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value) || 0)}
                  />
                </div>
                <div className="space-y-2">
                  <Label>Max tokens</Label>
                  <Input
                    type="number"
                    min={50} max={2000} step={50}
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value) || 200)}
                  />
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>System prompt</Label>
              <Textarea
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                rows={4}
                className="text-xs font-mono"
              />
            </div>

            {/* Mode tabs */}
            <Tabs value={mode} onValueChange={(v) => setMode(v as LLMMode)}>
              <TabsList className="w-full">
                <TabsTrigger value="zero_shot" className="flex-1">Zero-shot</TabsTrigger>
                <TabsTrigger value="few_shot" className="flex-1">Few-shot</TabsTrigger>
                <TabsTrigger value="cot" className="flex-1">CoT</TabsTrigger>
                <TabsTrigger value="bagging" className="flex-1">Bagging</TabsTrigger>
              </TabsList>

              <TabsContent value="zero_shot">
                <p className="text-sm text-muted-foreground py-3">
                  Класифікація без прикладів — модель покладається лише на system prompt.
                </p>
              </TabsContent>

              <TabsContent value="few_shot">
                <div className="space-y-3 py-3">
                  <p className="text-sm text-muted-foreground">
                    Додайте приклади FAKE та REAL текстів для навчання в контексті.
                  </p>

                  {/* Existing examples */}
                  {fewShotExamples.length > 0 && (
                    <div className="space-y-2 max-h-40 overflow-y-auto">
                      {fewShotExamples.map((ex, i) => (
                        <div key={i} className="flex items-start gap-2 p-2 rounded bg-muted/50 text-sm">
                          <Badge
                            variant={ex.label === "FAKE" ? "destructive" : "default"}
                            className={cn(
                              "shrink-0 mt-0.5",
                              ex.label === "REAL" && "bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-400"
                            )}
                          >
                            {ex.label}
                          </Badge>
                          <span className="flex-1 text-xs line-clamp-2">{ex.text}</span>
                          <Button
                            variant="ghost" size="icon"
                            className="h-6 w-6 shrink-0"
                            onClick={() => removeExample(i)}
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Add from dataset */}
                  <div className="flex items-end gap-2 p-3 rounded-lg border">
                    <div className="space-y-1">
                      <Label className="text-xs">FAKE</Label>
                      <Input
                        type="number" min={0} max={10}
                        value={nFake}
                        onChange={(e) => setNFake(parseInt(e.target.value) || 0)}
                        className="w-16 h-8"
                      />
                    </div>
                    <div className="space-y-1">
                      <Label className="text-xs">REAL</Label>
                      <Input
                        type="number" min={0} max={10}
                        value={nReal}
                        onChange={(e) => setNReal(parseInt(e.target.value) || 0)}
                        className="w-16 h-8"
                      />
                    </div>
                    <Button
                      variant="outline" size="sm"
                      onClick={handleLoadRandomSamples}
                      disabled={loadingRandom || (nFake === 0 && nReal === 0)}
                    >
                      {loadingRandom ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : <Download className="mr-1 h-3 w-3" />}
                      Додати з датасету
                    </Button>
                  </div>

                  {/* Add manual */}
                  <div className="p-3 rounded-lg border space-y-2">
                    <div className="flex gap-2">
                      <Select value={manualLabel} onValueChange={(v) => setManualLabel(v as "FAKE" | "REAL")}>
                        <SelectTrigger className="w-24 h-8">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="FAKE">FAKE</SelectItem>
                          <SelectItem value="REAL">REAL</SelectItem>
                        </SelectContent>
                      </Select>
                      <Button variant="outline" size="sm" onClick={addManualExample} disabled={!manualText.trim()}>
                        Додати вручну
                      </Button>
                    </div>
                    <Textarea
                      value={manualText}
                      onChange={(e) => setManualText(e.target.value)}
                      placeholder="Текст прикладу..."
                      rows={2}
                      className="text-xs"
                    />
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="cot">
                <div className="space-y-2 py-3">
                  <Label>Інструкція Chain-of-Thought</Label>
                  <Textarea
                    value={cotInstruction}
                    onChange={(e) => setCotInstruction(e.target.value)}
                    rows={5}
                    className="text-xs font-mono"
                  />
                  <p className="text-xs text-muted-foreground">
                    Модель міркуватиме крок за кроком перед відповіддю. Max tokens буде збільшено до 500+.
                  </p>
                </div>
              </TabsContent>

              <TabsContent value="bagging">
                <div className="space-y-3 py-3">
                  <div className="space-y-2">
                    <Label>Кількість запитів (n_calls): {baggingNCalls}</Label>
                    <input
                      type="range" min={2} max={10} step={1}
                      value={baggingNCalls}
                      onChange={(e) => setBaggingNCalls(parseInt(e.target.value))}
                      className="w-full accent-primary"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>2</span><span>10</span>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Запуск N класифікацій з temperature &ge; 0.7, фінальна відповідь — majority voting.
                  </p>
                </div>
              </TabsContent>
            </Tabs>

            {/* Test section */}
            <div className="border-t pt-4 space-y-3">
              <p className="text-sm font-medium">Тест (опціонально)</p>
              <Textarea
                value={testText}
                onChange={(e) => setTestText(e.target.value)}
                placeholder="Вставте текст для тестування класифікації..."
                rows={3}
                className="text-xs"
              />
              <Button
                variant="outline" size="sm"
                onClick={handleTest}
                disabled={testing || !testText.trim()}
              >
                {testing && <Loader2 className="mr-2 h-3 w-3 animate-spin" />}
                Тестувати
              </Button>

              {testResult && (
                <Card className={cn(
                  "border",
                  testResult.label === "FAKE" && "border-red-300 bg-red-50 dark:bg-red-950/20",
                  testResult.label === "REAL" && "border-green-300 bg-green-50 dark:bg-green-950/20",
                )}>
                  <CardContent className="p-3 space-y-1 text-sm">
                    <div className="flex items-center gap-2">
                      <Badge variant={testResult.label === "FAKE" ? "destructive" : "default"}
                        className={cn(testResult.label === "REAL" && "bg-green-600 text-white")}>
                        {testResult.label}
                      </Badge>
                      <span className="font-medium">{(testResult.confidence * 100).toFixed(1)}%</span>
                      <span className="text-xs text-muted-foreground ml-auto">
                        {testResult.base_model_used} &middot; {testResult.elapsed_seconds.toFixed(1)}s
                      </span>
                    </div>
                    {testResult.reason && (
                      <p className="text-xs text-muted-foreground">{testResult.reason}</p>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateOpen(false)}>Скасувати</Button>
            <Button onClick={handleSave} disabled={saving || !name.trim()}>
              {saving && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Зберегти
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
