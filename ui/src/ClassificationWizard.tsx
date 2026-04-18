import React, { useState, useEffect } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Checkbox } from "./components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Label } from "./components/ui/label";
import { Check, ChevronLeft, ChevronRight, Users, User as UserIcon, Sparkles } from "lucide-react";
import type { FeatureGroupDef, ModelRecord } from "./types";

// ── Constants ────────────────────────────────────────────────────────────────

const FEATURE_GROUPS: Record<string, FeatureGroupDef> = {
  semantic: {
    label: "Семантичні",
    description: "TF-IDF векторизація тексту — основа класифікації",
    features: [{ key: "text", label: "Текст", type: "semantic" }],
  },
  emotional: {
    label: "Емоційні",
    description: "Емоційний тон, невербальна емоційність та базові емоції за NRC лексиконом",
    features: [
      { key: "sentiment_score", label: "Тональність тексту", type: "emotional" },
      { key: "emotion_intensity", label: "Інтенсивність емоцій", type: "emotional" },
      { key: "emoji_count", label: "Кількість емодзі", type: "emotional" },
      { key: "exclamation_count", label: "Знаки оклику", type: "emotional" },
      { key: "anger_score", label: "Гнів (NRC)", type: "emotional" },
      { key: "fear_score", label: "Страх (NRC)", type: "emotional" },
      { key: "anticipation_score", label: "Передчуття (NRC)", type: "emotional" },
      { key: "trust_score", label: "Довіра (NRC)", type: "emotional" },
      { key: "surprise_score", label: "Здивування (NRC)", type: "emotional" },
      { key: "sadness_score", label: "Смуток (NRC)", type: "emotional" },
      { key: "joy_score", label: "Радість (NRC)", type: "emotional" },
      { key: "disgust_score", label: "Відраза (NRC)", type: "emotional" },
      { key: "positive_score", label: "Позитивність (NRC)", type: "emotional" },
      { key: "negative_score", label: "Негативність (NRC)", type: "emotional" },
    ],
  },
  stylistic: {
    label: "Стилістичні",
    description: "Лінгвістичний стиль та читабельність тексту",
    features: [
      { key: "caps_ratio", label: "Частка ВЕЛИКИХ ЛІТЕР", type: "stylistic" },
      { key: "ttr", label: "Лексичне різноманіття", type: "stylistic" },
      { key: "repetition_score", label: "Повторюваність фраз", type: "stylistic" },
      { key: "avg_word_length", label: "Середня довжина слова", type: "stylistic" },
    ],
  },
  rhetorical: {
    label: "Риторичні",
    description: "Маніпулятивні риторичні прийоми та апеляції",
    features: [
      { key: "clickbait_score", label: "Клікбейт та маніпуляції", type: "rhetorical" },
      { key: "authority_refs", label: "Анонімні посилання", type: "rhetorical" },
      { key: "pronoun_ratio", label: "Займенники ми/вони", type: "rhetorical" },
      { key: "question_count", label: "Риторичні питання", type: "rhetorical" },
    ],
  },
  social: {
    label: "Соціальні",
    description: "Соціальні метадані: рейтинг, коментарі, домен",
    features: [
      { key: "upvote_ratio", label: "Upvote ratio", type: "social" },
      { key: "score_normalized", label: "Нормалізований score", type: "social" },
      { key: "num_comments_norm", label: "Кількість коментарів", type: "social" },
      { key: "domain_credibility", label: "Надійність домену", type: "social" },
      { key: "account_age_norm", label: "Вік акаунту", type: "social" },
      { key: "has_url", label: "Наявність URL", type: "social" },
    ],
  },
};

const ALL_FEATURE_KEYS = [
  "text",
  "sentiment_score","emotion_intensity","emoji_count","exclamation_count",
  "anger_score","fear_score","anticipation_score","trust_score","surprise_score",
  "sadness_score","joy_score","disgust_score","positive_score","negative_score",
  "caps_ratio","ttr","repetition_score","avg_word_length",
  "clickbait_score","authority_refs","pronoun_ratio","question_count",
  "upvote_ratio","score_normalized","num_comments_norm","domain_credibility","account_age_norm","has_url",
];

const MODEL_OPTIONS = [
  { id: "nb", name: "Naive Bayes", desc: "MultinomialNB / ComplementNB + TF-IDF", group: "classical" },
  { id: "deberta", name: "DistilBERT", desc: "Fine-tuned трансформер + додаткові ознаки", group: "neural" },
  { id: "llm", name: "Gemini (Zero-Shot)", desc: "Zero-shot класифікація через OpenAI API", group: "llm" },
];

const MODEL_LABELS: Record<string, string> = { nb: "Naive Bayes", deberta: "DistilBERT", llm: "Gemini (Zero-Shot)" };

const ENSEMBLE_STRATEGIES = [
  { id: "hard", name: "Hard Voting", desc: "Більшість голосів (majority label)" },
  { id: "soft", name: "Soft Voting", desc: "Середнє ймовірностей моделей" },
  { id: "weighted", name: "Weighted Voting", desc: "Зважена сума (ваги задаються вручну)" },
];

function parseLlmPresetConfig(llmConfigJson: string | null | undefined): string {
  if (!llmConfigJson) return "?";
  try {
    const cfg = JSON.parse(llmConfigJson);
    const modeLabels: Record<string, string> = {
      zero_shot: "Zero-shot", few_shot: "Few-shot", cot: "CoT", bagging: "Bagging",
    };
    return `${cfg.base_model || "?"} · ${modeLabels[cfg.mode] || cfg.mode}`;
  } catch {
    return "?";
  }
}

function buildDefaultMask(allTrue: boolean) {
  const mask: Record<string, boolean> = {};
  ALL_FEATURE_KEYS.forEach((k) => { mask[k] = allTrue; });
  return mask;
}

const DEFAULT_PARAMS: any = {
  nb: {
    variant: "multinomial", vectorizer: "tfidf", ngram: "1,1", alpha: "1.0",
    additional_groups: ["semantic"], feature_mask: buildDefaultMask(true),
    preprocessing: {
      removeUrls: true, removeMentions: true, cleaning: true, lowercase: true,
      removePunctuation: true, removeNumbers: false, removeStopwords: true,
      stemming: false, lemmatization: true,
    },
  },
  deberta: { integration_mode: "concat", additional_groups: ["semantic"], feature_mask: buildDefaultMask(false) },
  llm: { mode: "single", lang: "auto", additional_groups: ["semantic"], feature_mask: buildDefaultMask(true) },
};

function getDefaultParams(type: string) {
  const p = DEFAULT_PARAMS[type] || {};
  return { ...p, feature_mask: { ...(p.feature_mask || buildDefaultMask(true)) } };
}

type ClassificationWizardProps = {
  trainingMode?: boolean;
  onConfigChange?: (cfg: any) => void;
};

export default function ClassificationWizard({ trainingMode = false, onConfigChange }: ClassificationWizardProps) {
  const [mode, setMode] = useState<any>(null);
  const [step, setStep] = useState<number>(0);
  const [selectedModel, setSelectedModel] = useState<string>("nb");
  const [selectedModels, setSelectedModels] = useState<any[]>([]);
  const [modelParams, setModelParams] = useState<any>({});
  const [ensembleStrategy, setEnsembleStrategy] = useState<string>("hard");
  const [weights, setWeights] = useState<any>({});

  // LLM presets
  const [llmPresets, setLlmPresets] = useState<ModelRecord[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<number | null>(null);

  useEffect(() => {
    api.get<ModelRecord[]>("/models")
      .then((res) => setLlmPresets(res.data.filter((m) => m.model_type === "llm")))
      .catch(() => {});
  }, []);

  const STEPS: any = {
    single: ["Режим", "Модель", "Параметри", "Підсумок"],
    ensemble: ["Режим", "Стратегія", "Моделі", "Параметри", "Підсумок"],
  };
  const stepLabels = STEPS[mode] || STEPS.single;
  const lastStep = stepLabels.length - 1;
  const isLastStep = mode !== null && step === lastStep;

  const setParam = (type: string, key: string, val: any) =>
    setModelParams((prev: any) => ({ ...prev, [type]: { ...prev[type], [key]: val } }));

  const toggleGroup = (type: string, groupName: string) =>
    setModelParams((prev: any) => {
      const p = prev[type] || getDefaultParams(type);
      const groups = [...(p.additional_groups || [])];
      const mask = { ...(p.feature_mask || buildDefaultMask(true)) };
      const idx = groups.indexOf(groupName);
      if (idx >= 0) {
        groups.splice(idx, 1);
        FEATURE_GROUPS[groupName].features.forEach((f: any) => { mask[f.key] = false; });
      } else {
        groups.push(groupName);
        FEATURE_GROUPS[groupName].features.forEach((f: any) => { mask[f.key] = true; });
      }
      return { ...prev, [type]: { ...p, additional_groups: groups, feature_mask: mask } };
    });

  const toggleFeature = (type: string, key: string) =>
    setModelParams((prev: any) => {
      const p = prev[type] || getDefaultParams(type);
      const mask = { ...(p.feature_mask || buildDefaultMask(true)) };
      mask[key] = !mask[key];
      return { ...prev, [type]: { ...p, feature_mask: mask } };
    });

  const togglePreprocessing = (type: string, key: string) =>
    setModelParams((prev: any) => {
      const pp = prev[type]?.preprocessing || {};
      const updated = { ...pp, [key]: !pp[key] };
      if (key === "stemming" && updated.stemming) updated.lemmatization = false;
      if (key === "lemmatization" && updated.lemmatization) updated.stemming = false;
      return { ...prev, [type]: { ...prev[type], preprocessing: updated } };
    });

  const toggleModel = (id: string) =>
    setSelectedModels((prev: any[]) => prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]);

  const canNext = () => {
    if (step === 0) return mode !== null;
    if (mode === "single") { if (step === 1) return !!selectedModel; return true; }
    if (step === 1) return !!ensembleStrategy;
    if (step === 2) return selectedModels.length >= 2;
    return true;
  };

  const handleNext = () => {
    if (!canNext()) return;
    const paramsStep = mode === "single" ? 1 : 2;
    if (step === paramsStep) {
      const models = mode === "single" ? [selectedModel] : selectedModels;
      setModelParams((prev: any) => {
        const updates: any = {};
        models.forEach((t: string) => { if (!prev[t]) updates[t] = getDefaultParams(t); });
        return Object.keys(updates).length ? { ...prev, ...updates } : prev;
      });
      if (mode === "ensemble") {
        setWeights((prev: any) => {
          const next: any = {};
          selectedModels.forEach((m: string) => { next[m] = prev[m] !== undefined ? prev[m] : 1; });
          return next;
        });
      }
    }
    setStep((s) => s + 1);
  };

  const handleBack = () => setStep((s) => s - 1);

  const buildRequest = () => {
    const modelTypes = mode === "single" ? [selectedModel] : selectedModels;
    const models = modelTypes.map((type: string) => {
      const p = modelParams[type] || getDefaultParams(type);
      const groups = p.additional_groups || [];
      const mask = p.feature_mask || buildDefaultMask(false);
      const hasFeatures = Object.values(mask).some(Boolean);
      const model: any = { model: type };
      if (hasFeatures) { model.additional_features = { groups, mask }; }
      else { model.additional_features = null; }
      if (type === "nb") {
        model.variant = p.variant || "multinomial";
        model.vectorizer = p.vectorizer || "tfidf";
        model.ngram_range = p.ngram || "1,1";
        model.alpha = p.alpha || "1.0";
      }
      if (type === "deberta") { model.integration_mode = p.integration_mode || "concat"; }
      if (type === "llm") {
        if (selectedPresetId) { model.preset_id = selectedPresetId; }
        model.mode = p.mode || "zero_shot";
      }
      return model;
    });
    const nbParams = modelParams.nb || getDefaultParams("nb");
    const firstModel = modelTypes[0];
    const preprocessing = firstModel === "nb" ? nbParams.preprocessing : { removeUrls: true, removeMentions: true, cleaning: true };
    const req: any = { mode, models, preprocessing };
    if (mode === "ensemble") {
      req.ensemble = { strategy: ensembleStrategy };
      if (ensembleStrategy === "weighted") {
        const total = (Object.values(weights).reduce((s: number, v: any) => s + Number(v), 0) as number) || 1;
        const normWeights: any = {};
        selectedModels.forEach((m: string) => { normWeights[m] = (weights[m] || 0) / total; });
        req.ensemble.weights = normWeights;
      }
    }
    return req;
  };

  React.useEffect(() => {
    if (!trainingMode || !onConfigChange || !mode) return;
    const config = buildRequest();
    onConfigChange(config);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingMode, mode, selectedModel, selectedModels, modelParams, ensembleStrategy, weights, selectedPresetId]);

  // ── Step renderers ─────────────────────────────────────────────────────

  const renderModeSelect = () => (
    <div className="space-y-4">
      <p className="font-medium">Оберіть режим класифікації</p>
      <div className="grid grid-cols-2 gap-4">
        {[
          { id: "single", icon: UserIcon, title: "Одна модель", desc: "Обрати один алгоритм та налаштувати його параметри" },
          { id: "ensemble", icon: Users, title: "Ансамбль", desc: "Два або більше алгоритмів — голосування або зважена сума" },
        ].map((m) => {
          const Icon = m.icon;
          return (
            <Card
              key={m.id}
              className={cn("cursor-pointer transition-all hover:shadow-md", mode === m.id && "ring-2 ring-primary")}
              onClick={() => setMode(m.id)}
            >
              <CardContent className="p-6 text-center">
                <Icon className={cn("h-8 w-8 mx-auto mb-3", mode === m.id ? "text-primary" : "text-muted-foreground")} />
                <p className="font-semibold text-sm">{m.title}</p>
                <p className="text-xs text-muted-foreground mt-1">{m.desc}</p>
              </CardContent>
            </Card>
          );
        })}
      </div>
      {!mode && <p className="text-sm text-muted-foreground">Клікніть на картку щоб обрати режим</p>}
    </div>
  );

  const renderSingleModelSelect = () => (
    <div className="space-y-3">
      <p className="font-medium">Оберіть алгоритм</p>
      <div className="space-y-2">
        {MODEL_OPTIONS.map((m) => (
          <Card
            key={m.id}
            className={cn("cursor-pointer transition-all", selectedModel === m.id && "ring-2 ring-primary")}
            onClick={() => setSelectedModel(m.id)}
          >
            <CardContent className="p-4 flex items-center gap-3">
              <div className={cn(
                "w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0",
                selectedModel === m.id ? "border-primary bg-primary" : "border-muted-foreground"
              )}>
                {selectedModel === m.id && <Check className="h-3 w-3 text-primary-foreground" />}
              </div>
              <div>
                <p className="font-medium text-sm">{m.name}</p>
                <p className="text-xs text-muted-foreground">{m.desc}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  const renderEnsembleModelSelect = () => (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">Оберіть мінімум 2 моделі для ансамблю</p>
      <div className="space-y-2">
        {MODEL_OPTIONS.map((m) => {
          const checked = selectedModels.includes(m.id);
          return (
            <Card
              key={m.id}
              className={cn("cursor-pointer transition-all", checked && "ring-2 ring-primary")}
              onClick={() => toggleModel(m.id)}
            >
              <CardContent className="p-4 flex items-center gap-3">
                <Checkbox checked={checked} onCheckedChange={() => toggleModel(m.id)} />
                <div>
                  <p className="font-medium text-sm">{m.name}</p>
                  <p className="text-xs text-muted-foreground">{m.desc}</p>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
      {selectedModels.length === 1 && (
        <p className="text-sm text-destructive">Оберіть ще хоча б одну модель</p>
      )}
    </div>
  );

  const renderModelParams = (type: string) => {
    const p = modelParams[type] || getDefaultParams(type);
    const upd = (key: string, val: any) => setParam(type, key, val);
    const groups = p.additional_groups || [];

    switch (type) {
      case "nb": {
        const pp = p.preprocessing || {};
        return (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-xs">Варіант</Label>
                <Select value={p.variant || "multinomial"} onValueChange={(v) => upd("variant", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="multinomial">MultinomialNB</SelectItem>
                    <SelectItem value="complement">ComplementNB</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Векторизація</Label>
                <Select value={p.vectorizer || "tfidf"} onValueChange={(v) => upd("vectorizer", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tfidf">TF-IDF</SelectItem>
                    <SelectItem value="count">CountVectorizer</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Н-грами</Label>
                <Select value={p.ngram || "1,1"} onValueChange={(v) => upd("ngram", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1,1">(1,1)</SelectItem>
                    <SelectItem value="1,2">(1,2)</SelectItem>
                    <SelectItem value="1,3">(1,3)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Alpha (Laplace)</Label>
                <Select value={p.alpha || "1.0"} onValueChange={(v) => upd("alpha", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["0.01","0.1","0.5","1.0","2.0"].map((v) => (
                      <SelectItem key={v} value={v}>{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Preprocessing */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Checkbox
                  checked={["removeUrls","removeMentions","cleaning","lowercase","removeStopwords","removePunctuation","removeNumbers"].every(k => pp[k])}
                  onCheckedChange={(checked) => {
                    setModelParams((prev: any) => {
                      const oldPp = prev[type]?.preprocessing || {};
                      const updated: any = { ...oldPp };
                      ["removeUrls","removeMentions","cleaning","lowercase","removeStopwords","removePunctuation","removeNumbers"].forEach(k => { updated[k] = !!checked; });
                      return { ...prev, [type]: { ...prev[type], preprocessing: updated } };
                    });
                  }}
                />
                <Label className="text-sm font-medium">Попередня обробка тексту</Label>
              </div>
              <div className="grid grid-cols-2 gap-2 pl-6">
                {[
                  { key: "removeUrls", label: "Видалення URL-посилань" },
                  { key: "removeMentions", label: "Видалення @згадок" },
                  { key: "cleaning", label: "Очищення пробілів" },
                  { key: "lowercase", label: "Нижній регістр" },
                  { key: "removeStopwords", label: "Видалення стоп-слів" },
                  { key: "removePunctuation", label: "Видалення пунктуації" },
                  { key: "removeNumbers", label: "Видалення чисел" },
                ].map((opt) => (
                  <label key={opt.key} className="flex items-center gap-2 text-xs cursor-pointer">
                    <Checkbox
                      checked={pp[opt.key] || false}
                      onCheckedChange={() => togglePreprocessing(type, opt.key)}
                    />
                    {opt.label}
                  </label>
                ))}
              </div>
              <div className="pl-6 space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Нормалізація слів:</span>
                <div className="flex gap-4">
                  {[
                    { value: "none", label: "Немає" },
                    { value: "stemming", label: "Стемінг" },
                    { value: "lemmatization", label: "Лематизація" },
                  ].map((opt) => {
                    const current = pp.stemming ? "stemming" : pp.lemmatization ? "lemmatization" : "none";
                    return (
                      <label key={opt.value} className="flex items-center gap-2 text-xs cursor-pointer">
                        <input
                          type="radio"
                          name={`normalization-${type}`}
                          checked={current === opt.value}
                          onChange={() => {
                            setModelParams((prev: any) => ({
                              ...prev,
                              [type]: {
                                ...prev[type],
                                preprocessing: {
                                  ...prev[type]?.preprocessing,
                                  stemming: opt.value === "stemming",
                                  lemmatization: opt.value === "lemmatization",
                                },
                              },
                            }));
                          }}
                          className="accent-primary"
                        />
                        {opt.label}
                      </label>
                    );
                  })}
                </div>
              </div>
            </div>

            <AdditionalFeaturesSection type={type} groups={groups} mask={p.feature_mask} toggleGroup={toggleGroup} toggleFeature={toggleFeature} />
          </div>
        );
      }
      case "deberta":
        return (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">Fine-tuned DistilBERT — класифікація тексту через трансформер (concat mode)</p>
            <AdditionalFeaturesSection type={type} groups={groups} mask={p.feature_mask} toggleGroup={toggleGroup} toggleFeature={toggleFeature} />
          </div>
        );
      case "llm":
        return (
          <div className="space-y-4">
            <div>
              <Label className="text-xs">LLM пресет</Label>
              <Select
                value={selectedPresetId?.toString() || ""}
                onValueChange={(v) => setSelectedPresetId(parseInt(v))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Оберіть пресет..." />
                </SelectTrigger>
                <SelectContent>
                  {llmPresets.length === 0 ? (
                    <div className="p-3 text-sm text-muted-foreground text-center">
                      Пресетів ще немає. Створіть у розділі "LLM пресети".
                    </div>
                  ) : (
                    llmPresets.map((pr) => {
                      const cfg = parseLlmPresetConfig(pr.llm_config);
                      return (
                        <SelectItem key={pr.id} value={pr.id.toString()}>
                          <span className="flex items-center gap-2">
                            <Sparkles className="h-3 w-3" />
                            {pr.name} &middot; {cfg}
                          </span>
                        </SelectItem>
                      );
                    })
                  )}
                </SelectContent>
              </Select>
            </div>
            {selectedPresetId && (() => {
              const preset = llmPresets.find((pr) => pr.id === selectedPresetId);
              if (!preset) return null;
              const cfg = parseLlmPresetConfig(preset.llm_config);
              return (
                <div className="text-xs text-muted-foreground p-2 rounded bg-muted/50">
                  <span className="font-medium">{preset.name}</span> &middot; {cfg}
                </div>
              );
            })()}
          </div>
        );
      default: return null;
    }
  };

  const renderParamsStep = () => {
    const models = mode === "single" ? [selectedModel] : selectedModels;
    if (models.length === 1) {
      return (
        <div className="space-y-3">
          <p className="font-medium">Параметри {MODEL_LABELS[models[0]] || models[0]}</p>
          {renderModelParams(models[0])}
        </div>
      );
    }
    return (
      <div className="space-y-4">
        <p className="text-sm text-muted-foreground">Налаштуйте параметри кожної моделі</p>
        {models.map((type: string) => (
          <details key={type} open className="group">
            <summary className="flex items-center gap-2 p-3 bg-muted rounded-lg cursor-pointer font-medium text-sm">
              {MODEL_LABELS[type] || type}
            </summary>
            <div className="pt-4 pl-2">{renderModelParams(type)}</div>
          </details>
        ))}
      </div>
    );
  };

  const renderEnsembleStrategy = () => {
    const total = Object.values(weights).reduce((s: number, v: any) => s + Number(v), 0) as number;
    return (
      <div className="space-y-4">
        <p className="text-sm text-muted-foreground">Оберіть стратегію об'єднання прогнозів</p>
        <div className="grid grid-cols-2 gap-3">
          {ENSEMBLE_STRATEGIES.map((s: any) => (
            <Card
              key={s.id}
              className={cn("cursor-pointer transition-all", ensembleStrategy === s.id && "ring-2 ring-primary")}
              onClick={() => setEnsembleStrategy(s.id)}
            >
              <CardContent className="p-4">
                <p className="font-medium text-sm">{s.name}</p>
                <p className="text-xs text-muted-foreground mt-1">{s.desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>
        {ensembleStrategy === "weighted" && (
          <div className="space-y-4 mt-4">
            <p className="font-medium text-sm">Ваги моделей</p>
            <p className="text-xs text-muted-foreground">Автоматично нормалізуються до суми 1.0</p>
            {selectedModels.map((mid: string) => {
              const norm = total > 0 ? Number(weights[mid] || 0) / total : 0;
              return (
                <div key={mid} className="space-y-1">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{MODEL_LABELS[mid]}</span>
                    <span className="text-primary font-semibold">{norm.toFixed(2)}</span>
                  </div>
                  <input
                    type="range" min="0" max="10" step="0.5"
                    value={weights[mid] || 0}
                    onChange={(e) => setWeights((p: any) => ({ ...p, [mid]: parseFloat(e.target.value) }))}
                    className="w-full accent-primary"
                  />
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  const renderConfigSummary = () => {
    const modelIds = mode === "single" ? [selectedModel] : selectedModels;
    const strategyObj = ENSEMBLE_STRATEGIES.find((s: any) => s.id === ensembleStrategy);
    const total = (Object.values(weights).reduce((s: number, v: any) => s + Number(v), 0) as number) || 1;

    return (
      <Card>
        <CardContent className="p-4 space-y-3 text-sm">
          <p className="font-semibold">Підсумок конфігурації</p>
          <p><span className="font-medium">Режим:</span> {mode === "single" ? "Одна модель" : "Ансамбль"}</p>
          {mode === "ensemble" && <p><span className="font-medium">Стратегія:</span> {strategyObj?.name}</p>}
          {mode === "ensemble" && ensembleStrategy === "weighted" && (
            <p><span className="font-medium">Ваги:</span> {modelIds.map((m: string) => `${MODEL_LABELS[m]} ${(Number(weights[m] || 0) / total).toFixed(2)}`).join(", ")}</p>
          )}
          {modelIds.map((mid: string) => {
            const p = modelParams[mid] || getDefaultParams(mid);
            const groups = p.additional_groups || [];
            const mask = p.feature_mask || {};
            const activeCount = groups.length > 0
              ? Object.entries(mask).filter(([k, v]: any) => v && ALL_FEATURE_KEYS.includes(k)).length
              : 0;
            return (
              <div key={mid} className="pt-2 border-t border-border">
                <p className="font-semibold text-xs">{MODEL_LABELS[mid] || mid}</p>
                {mid === "nb" && (() => {
                  const pp = p.preprocessing || {};
                  const ppLabels: any = {
                    removeUrls:"URL", removeMentions:"@згадки", cleaning:"пробіли", lowercase:"lowercase",
                    removeStopwords:"стоп-слова", stemming:"стемінг", lemmatization:"лематизація",
                    removePunctuation:"пунктуація", removeNumbers:"числа",
                  };
                  const activePreprocessing = Object.entries(pp).filter(([,v]) => v).map(([k]) => ppLabels[k] || k);
                  return (
                    <>
                      <p><span className="font-medium">Варіант:</span> {(p.variant || "multinomial").charAt(0).toUpperCase() + (p.variant || "multinomial").slice(1)}NB</p>
                      <p><span className="font-medium">Векторизація:</span> {p.vectorizer === "count" ? "CountVectorizer" : "TF-IDF"}</p>
                      <p><span className="font-medium">Н-грами:</span> ({p.ngram || "1,1"})</p>
                      <p><span className="font-medium">Обробка:</span> {activePreprocessing.length > 0 ? activePreprocessing.join(", ") : "немає"}</p>
                    </>
                  );
                })()}
                {mid === "deberta" && <p><span className="font-medium">Інтеграція:</span> {({ concat: "Concat", multiview: "Multi-view" } as any)[p.integration_mode || "concat"]}</p>}
                {mid === "llm" && (() => {
                  const preset = selectedPresetId ? llmPresets.find((pr) => pr.id === selectedPresetId) : null;
                  return preset ? (
                    <p><span className="font-medium">Пресет:</span> {preset.name} ({parseLlmPresetConfig(preset.llm_config)})</p>
                  ) : (
                    <p><span className="font-medium">Пресет:</span> не обрано</p>
                  );
                })()}
                <p><span className="font-medium">Ознаки:</span> {groups.length > 0 ? groups.map((g: string) => FEATURE_GROUPS[g]?.label).join(", ") : "вимкнено"}{activeCount > 0 ? ` (${activeCount} додаткових)` : ""}</p>
              </div>
            );
          })}
          <div className="pt-2 text-right">
            <Button variant="outline" size="sm" onClick={() => setStep(mode === "single" ? 2 : 3)}>Змінити</Button>
          </div>
        </CardContent>
      </Card>
    );
  };

  const renderCurrentStep = () => {
    if (step === 0) return renderModeSelect();
    if (mode === "single") {
      if (step === 1) return renderSingleModelSelect();
      if (step === 2) return renderParamsStep();
      if (step === 3) return renderConfigSummary();
      return null;
    }
    if (step === 1) return renderEnsembleStrategy();
    if (step === 2) return renderEnsembleModelSelect();
    if (step === 3) return renderParamsStep();
    if (step === 4) return renderConfigSummary();
    return null;
  };

  return (
    <div className="max-w-3xl mx-auto">
      {/* Step indicator */}
      {mode !== null && (
        <div className="flex items-center mb-6">
          {stepLabels.map((label: string, i: number) => {
            const state = i < step ? "done" : i === step ? "active" : "pending";
            return (
              <React.Fragment key={i}>
                <div className="flex flex-col items-center flex-1">
                  <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0",
                    state === "done" && "bg-green-500 text-white",
                    state === "active" && "bg-primary text-primary-foreground",
                    state === "pending" && "bg-muted text-muted-foreground",
                  )}>
                    {state === "done" ? <Check className="h-4 w-4" /> : i + 1}
                  </div>
                  <span className={cn(
                    "text-[10px] mt-1",
                    state === "active" ? "text-primary font-semibold" : "text-muted-foreground",
                    state === "pending" && "opacity-40",
                  )}>
                    {label}
                  </span>
                </div>
                {i < stepLabels.length - 1 && (
                  <div className={cn("h-0.5 flex-1 -mt-4", i < step ? "bg-green-500" : "bg-muted")} />
                )}
              </React.Fragment>
            );
          })}
        </div>
      )}

      {/* Current step content */}
      <div className="mb-4">{renderCurrentStep()}</div>

      {/* Navigation */}
      <div className="flex justify-between">
        <div>
          {step > 0 && (
            <Button variant="outline" onClick={handleBack}>
              <ChevronLeft className="h-4 w-4 mr-1" /> Назад
            </Button>
          )}
        </div>
        <div>
          {!isLastStep && (
            <Button onClick={handleNext} disabled={!canNext()}>
              Далі <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Additional Features Section ──────────────────────────────────────────────

function AdditionalFeaturesSection({ type, groups, mask, toggleGroup, toggleFeature }: any) {
  const featureMask = mask || {};
  return (
    <div className="space-y-3">
      <p className="font-medium text-sm">Ознаки:</p>
      <p className="text-xs text-muted-foreground">Оберіть групи ознак для класифікації:</p>
      {Object.entries(FEATURE_GROUPS).map(([groupKey, groupDef]: any) => {
        const isActive = groups.includes(groupKey);
        const hasSubFeatures = groupDef.features.length > 0;
        const activeInGroup = isActive && hasSubFeatures
          ? groupDef.features.filter((f: any) => featureMask[f.key]).length
          : 0;
        return (
          <div key={groupKey}>
            <div
              className={cn(
                "flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all border",
                isActive ? "border-primary bg-primary/5" : "border-border bg-muted/30 hover:bg-muted/50"
              )}
              onClick={() => toggleGroup(type, groupKey)}
            >
              <Checkbox checked={isActive} onCheckedChange={() => toggleGroup(type, groupKey)} />
              <div className="flex-1">
                <span className="font-medium text-sm">{groupDef.label}</span>
                <span className="text-xs text-muted-foreground ml-2">{groupDef.description}</span>
              </div>
              {isActive && hasSubFeatures && (
                <Badge variant="secondary" className="text-[10px]">{activeInGroup}/{groupDef.features.length}</Badge>
              )}
            </div>
            {isActive && hasSubFeatures && (
              <div className="pl-9 py-2 space-y-1">
                {groupDef.features.map((f: any) => (
                  <label key={f.key} className="flex items-center gap-2 text-xs cursor-pointer py-0.5">
                    <Checkbox
                      checked={featureMask[f.key] || false}
                      onCheckedChange={() => toggleFeature(type, f.key)}
                    />
                    {f.label}
                  </label>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
