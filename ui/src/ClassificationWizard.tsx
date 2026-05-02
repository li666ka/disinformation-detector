import React, { useEffect, useState } from "react";
import { cn } from "./lib/utils";
import api from "./api";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Checkbox } from "./components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Label } from "./components/ui/label";
import { Check, ChevronLeft, ChevronRight, Users, User as UserIcon } from "lucide-react";
import type { Dataset, FeatureGroupDef } from "./types";

// ── Constants ────────────────────────────────────────────────────────────────

const FEATURE_GROUPS: Record<string, FeatureGroupDef> = {
  semantic: {
    label: "Семантичні",
    description: "Векторизація тексту — основа класифікації",
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
    description: "Форма тексту + риторичні маніпуляції (clickbait, authority refs)",
    features: [
      // Original stylistic (4)
      { key: "caps_ratio", label: "Частка ВЕЛИКИХ ЛІТЕР", type: "stylistic" },
      { key: "ttr", label: "Лексичне різноманіття", type: "stylistic" },
      { key: "repetition_score", label: "Повторюваність фраз", type: "stylistic" },
      { key: "avg_word_length", label: "Середня довжина слова", type: "stylistic" },
      // Merged from rhetorical (4)
      { key: "clickbait_score", label: "Клікбейт та маніпуляції", type: "stylistic" },
      { key: "authority_refs", label: "Анонімні посилання", type: "stylistic" },
      { key: "pronoun_ratio", label: "Займенники ми/вони", type: "stylistic" },
      { key: "question_count", label: "Риторичні питання", type: "stylistic" },
    ],
  },
  social: {
    label: "Соціальні + Graph",
    description: "Профілі користувачів-поширювачів + структура поширення",
    features: [
      // Profile counts (6)
      { key: "followers_count_norm", label: "Кількість підписників", type: "social" },
      { key: "friends_count_norm", label: "Кількість підписок", type: "social" },
      { key: "ff_ratio", label: "Співвідношення followers/friends", type: "social" },
      { key: "statuses_count_norm", label: "Кількість твітів", type: "social" },
      { key: "account_age_norm", label: "Вік акаунту", type: "social" },
      { key: "statuses_per_day", label: "Постів на день (bot signal)", type: "social" },
      // Profile flags + strings (6)
      { key: "verified", label: "Верифікований акаунт", type: "social" },
      { key: "has_description", label: "Наявність опису", type: "social" },
      { key: "has_location", label: "Наявність локації", type: "social" },
      { key: "description_length_norm", label: "Довжина опису", type: "social" },
      { key: "screen_name_length_norm", label: "Довжина username", type: "social" },
      { key: "screen_name_digits_ratio", label: "Частка цифр у username", type: "social" },
      // Engagement (5)
      { key: "like_count_norm", label: "Кількість лайків поста", type: "social" },
      { key: "retweet_count_norm", label: "Кількість ретвітів", type: "social" },
      { key: "reply_count_norm", label: "Кількість коментарів", type: "social" },
      { key: "like_to_retweet_ratio", label: "Likes/Retweets ratio (Shu 2020)", type: "social" },
      { key: "engagement_rate", label: "Engagement rate (Cha 2010)", type: "social" },
      // Graph features (6 НОВИХ — обчислюються per-article з cascade)
      { key: "cascade_depth_norm", label: "Глибина каскаду reply tree", type: "social", isGraph: true },
      { key: "cascade_breadth_norm", label: "Ширина каскаду", type: "social", isGraph: true },
      { key: "lifetime_hours_norm", label: "Тривалість поширення (год)", type: "social", isGraph: true },
      { key: "retweets_per_tweet", label: "Ретвіти на твіт", type: "social", isGraph: true },
      { key: "replies_per_tweet", label: "Коментарі на твіт", type: "social", isGraph: true },
      { key: "unique_users_norm", label: "Унікальні користувачі", type: "social", isGraph: true },
    ],
  },
};

const GROUP_LABEL_OVERRIDES: Record<string, Record<string, string>> = {
  nb: { semantic: "Лексичні" },
};

const getGroupLabel = (groupKey: string, modelType?: string): string =>
  (modelType && GROUP_LABEL_OVERRIDES[modelType]?.[groupKey]) ||
  FEATURE_GROUPS[groupKey]?.label ||
  groupKey;

const ALL_FEATURE_KEYS = [
  "text",
  // Emotional — 14 keys
  "sentiment_score", "emotion_intensity", "emoji_count", "exclamation_count",
  "anger_score", "fear_score", "anticipation_score", "trust_score", "surprise_score",
  "sadness_score", "joy_score", "disgust_score", "positive_score", "negative_score",
  // Stylistic — 8 keys (4 form + 4 rhetorical, об'єднано)
  "caps_ratio", "ttr", "repetition_score", "avg_word_length",
  "clickbait_score", "authority_refs", "pronoun_ratio", "question_count",
  // Social — 23 keys (17 profile/engagement + 6 graph)
  "followers_count_norm", "friends_count_norm", "ff_ratio",
  "statuses_count_norm", "account_age_norm", "statuses_per_day",
  "verified", "has_description", "has_location",
  "description_length_norm", "screen_name_length_norm", "screen_name_digits_ratio",
  "like_count_norm", "retweet_count_norm", "reply_count_norm",
  "like_to_retweet_ratio", "engagement_rate",
  // Graph cascade (6 нових)
  "cascade_depth_norm", "cascade_breadth_norm", "lifetime_hours_norm",
  "retweets_per_tweet", "replies_per_tweet", "unique_users_norm",
];

const MODEL_OPTIONS = [
  { id: "nb", name: "Naive Bayes", desc: "MultinomialNB / ComplementNB + TF-IDF", group: "classical" },
  { id: "distilbert", name: "DistilBERT", desc: "Fine-tuning DistilBERT на повний article_text (article-level)", group: "neural" },
  { id: "gin", name: "GIN", desc: "Graph Isomorphism Network на графі стаття→твіти→ретвіти→коментарі", group: "graph" },
  { id: "sage", name: "GraphSAGE", desc: "Inductive GNN на графі стаття→твіти→ретвіти→коментарі", group: "graph" },
];

const MODEL_LABELS: Record<string, string> = {
  nb: "Naive Bayes",
  distilbert: "DistilBERT",
  gin: "GIN",
  sage: "GraphSAGE",
};

const ENSEMBLE_STRATEGIES = [
  { id: "hard", name: "Hard Voting", desc: "Більшість голосів (majority label)" },
  { id: "soft", name: "Soft Voting", desc: "Середнє ймовірностей моделей" },
  { id: "weighted", name: "Weighted Voting", desc: "Зважена сума (ваги задаються вручну)" },
];

function buildDefaultMask(allTrue: boolean, forceGroups: string[] = []) {
  const mask: Record<string, boolean> = {};
  ALL_FEATURE_KEYS.forEach((k) => { mask[k] = allTrue; });
  forceGroups.forEach((g) => {
    FEATURE_GROUPS[g]?.features.forEach((f: any) => { mask[f.key] = true; });
  });
  return mask;
}

const DEFAULT_PARAMS: any = {
  nb: {
    variant: "complement", vectorizer: "tfidf", ngram: "1,1", alpha: "1.0",
    use_text: true,
    additional_groups: ["semantic"], feature_mask: buildDefaultMask(false, ["semantic"]),
    preprocessing: {
      removeUrls: true, removeMentions: true, cleaning: true, lowercase: true,
      removePunctuation: true, removeNumbers: false, removeStopwords: true,
      stemming: false, lemmatization: true,
    },
  },
  distilbert: { integration_mode: "concat", additional_groups: ["semantic"], feature_mask: buildDefaultMask(false, ["semantic"]) },
  gin: {
    hidden_dim: "128",
    num_layers: "3",
    dropout: "0.5",
    learning_rate: "0.001",
    epochs: "50",
    pooling: "mean",
    additional_groups: ["semantic"],
    feature_mask: buildDefaultMask(false, ["semantic"]),
  },
  sage: {
    hidden_dim: "128",
    num_layers: "2",
    dropout: "0.5",
    learning_rate: "0.001",
    epochs: "50",
    aggregator: "mean",
    additional_groups: ["semantic"],
    feature_mask: buildDefaultMask(false, ["semantic"]),
  },
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
  const [activeDataset, setActiveDataset] = useState<Dataset | null>(null);

  useEffect(() => {
    if (!trainingMode) return;
    api
      .get<Dataset | null>("/datasets/active/info")
      .then((resp) => setActiveDataset(resp.data ?? null))
      .catch(() => setActiveDataset(null));
  }, [trainingMode]);

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
      const hasAnyFeature = Object.values(mask).some(Boolean);
      const hasGroups = groups.length > 0;
      const model: any = { model: type };
      if (hasGroups || hasAnyFeature) { model.additional_features = { groups, mask }; }
      else { model.additional_features = null; }
      if (type === "nb") {
        model.variant = p.variant || "complement";
        model.vectorizer = p.vectorizer || "tfidf";
        model.ngram_range = p.ngram || "1,1";
        model.alpha = p.alpha || "1.0";
        model.use_text = p.use_text ?? true;
      }
      if (type === "distilbert") { model.integration_mode = p.integration_mode || "concat"; }
      if (type === "gin" || type === "sage") {
        model.hidden_dim = p.hidden_dim || "128";
        model.num_layers = p.num_layers || (type === "gin" ? "3" : "2");
        model.dropout = p.dropout || "0.5";
        model.learning_rate = p.learning_rate || "0.001";
        model.epochs = p.epochs || "50";
        if (type === "gin") model.pooling = p.pooling || "mean";
        if (type === "sage") model.aggregator = p.aggregator || "mean";
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
  }, [trainingMode, mode, selectedModel, selectedModels, modelParams, ensembleStrategy, weights]);

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
                <Select value={p.variant || "complement"} onValueChange={(v) => upd("variant", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="complement">ComplementNB</SelectItem>
                    <SelectItem value="multinomial">MultinomialNB</SelectItem>
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
                    {["0.01", "0.1", "0.5", "1.0", "2.0"].map((v) => (
                      <SelectItem key={v} value={v}>{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Use TF-IDF tokens (для ablation) */}
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Checkbox
                  id={`nb-use-text-${type}`}
                  checked={p.use_text ?? true}
                  onCheckedChange={(checked) => upd("use_text", checked === true)}
                />
                <label
                  htmlFor={`nb-use-text-${type}`}
                  className="text-sm cursor-pointer"
                >
                  Використовувати TF-IDF tokens (тексти статей)
                </label>
              </div>
              <p className="text-xs text-muted-foreground pl-6">
                💡 Вимкніть для ablation: тренування тільки на feature
                engineering без bag-of-words.
              </p>
            </div>

            {/* Preprocessing */}
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <Checkbox
                  checked={["removeUrls", "removeMentions", "cleaning", "lowercase", "removeStopwords", "removePunctuation", "removeNumbers"].every(k => pp[k])}
                  onCheckedChange={(checked) => {
                    setModelParams((prev: any) => {
                      const oldPp = prev[type]?.preprocessing || {};
                      const updated: any = { ...oldPp };
                      ["removeUrls", "removeMentions", "cleaning", "lowercase", "removeStopwords", "removePunctuation", "removeNumbers"].forEach(k => { updated[k] = !!checked; });
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
      case "distilbert":
        return (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">Fine-tuning DistilBERT на article_title + article_text (article-level)</p>
            <AdditionalFeaturesSection type={type} groups={groups} mask={p.feature_mask} toggleGroup={toggleGroup} toggleFeature={toggleFeature} />
          </div>
        );
      case "gin":
      case "sage": {
        const isGin = type === "gin";
        return (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              {isGin
                ? "Graph Isomorphism Network — навчання на графі новина→твіти→retweet/reply з MiniLM ембеддингами вузлів"
                : "GraphSAGE — inductive GNN з sampling-агрегацією сусідів у графі поширення новини"}
            </p>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-xs">Hidden dim</Label>
                <Select value={p.hidden_dim || "128"} onValueChange={(v) => upd("hidden_dim", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["64", "128", "256", "512"].map((v) => (
                      <SelectItem key={v} value={v}>{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Кількість шарів</Label>
                <Select value={p.num_layers || (isGin ? "3" : "2")} onValueChange={(v) => upd("num_layers", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["2", "3", "4", "5"].map((v) => (
                      <SelectItem key={v} value={v}>{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Dropout</Label>
                <Select value={p.dropout || "0.5"} onValueChange={(v) => upd("dropout", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["0.0", "0.2", "0.3", "0.5", "0.7"].map((v) => (
                      <SelectItem key={v} value={v}>{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Learning rate</Label>
                <Select value={p.learning_rate || "0.001"} onValueChange={(v) => upd("learning_rate", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["0.0001", "0.0005", "0.001", "0.005", "0.01"].map((v) => (
                      <SelectItem key={v} value={v}>{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Epochs</Label>
                <Select value={p.epochs || "50"} onValueChange={(v) => upd("epochs", v)}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["20", "50", "100", "200"].map((v) => (
                      <SelectItem key={v} value={v}>{v}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {isGin ? (
                <div className="space-y-2">
                  <Label className="text-xs">Pooling</Label>
                  <Select value={p.pooling || "mean"} onValueChange={(v) => upd("pooling", v)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="mean">Mean</SelectItem>
                      <SelectItem value="sum">Sum</SelectItem>
                      <SelectItem value="max">Max</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              ) : (
                <div className="space-y-2">
                  <Label className="text-xs">Aggregator</Label>
                  <Select value={p.aggregator || "mean"} onValueChange={(v) => upd("aggregator", v)}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="mean">Mean</SelectItem>
                      <SelectItem value="max">Max</SelectItem>
                      <SelectItem value="lstm">LSTM</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              )}
            </div>
            <AdditionalFeaturesSection type={type} groups={groups} mask={p.feature_mask} toggleGroup={toggleGroup} toggleFeature={toggleFeature} />
          </div>
        );
      }
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

          {activeDataset && (
            <div className="rounded-lg bg-muted/50 p-3 text-sm space-y-1">
              <p className="font-medium">Дані для тренування:</p>
              <p className="text-muted-foreground">
                Датасет: {activeDataset.name}
              </p>
              <p className="text-muted-foreground">
                Split:{" "}
                {activeDataset.active_split
                  ? `${activeDataset.active_split} (фіксований)`
                  : "auto-split 70/15/15 (генерується автоматично)"}
              </p>
              {activeDataset.active_split && (
                <p className="text-xs text-muted-foreground">
                  💡 Щоб змінити split — поверніться на DatasetsPage
                </p>
              )}
            </div>
          )}

          <p><span className="font-medium">Режим:</span> {mode === "single" ? "Одна модель" : "Ансамбль"}</p>
          {mode === "ensemble" && <p><span className="font-medium">Стратегія:</span> {strategyObj?.name}</p>}
          {mode === "ensemble" && ensembleStrategy === "weighted" && (
            <p><span className="font-medium">Ваги:</span> {modelIds.map((m: string) => `${MODEL_LABELS[m]} ${(Number(weights[m] || 0) / total).toFixed(2)}`).join(", ")}</p>
          )}
          {modelIds.map((mid: string) => {
            const p = modelParams[mid] || getDefaultParams(mid);
            const groups = p.additional_groups || [];
            const mask = p.feature_mask || {};
            const activeCount = groups.reduce((sum: number, g: string) => {
              const feats = FEATURE_GROUPS[g]?.features || [];
              return sum + feats.filter((f: any) => mask[f.key] && f.key !== "text").length;
            }, 0);
            return (
              <div key={mid} className="pt-2 border-t border-border">
                <p className="font-semibold text-xs">{MODEL_LABELS[mid] || mid}</p>
                {mid === "nb" && (() => {
                  const pp = p.preprocessing || {};
                  const ppLabels: any = {
                    removeUrls: "URL", removeMentions: "@згадки", cleaning: "пробіли", lowercase: "lowercase",
                    removeStopwords: "стоп-слова", stemming: "стемінг", lemmatization: "лематизація",
                    removePunctuation: "пунктуація", removeNumbers: "числа",
                  };
                  const activePreprocessing = Object.entries(pp).filter(([, v]) => v).map(([k]) => ppLabels[k] || k);
                  return (
                    <>
                      <p><span className="font-medium">Варіант:</span> {(p.variant || "complement").charAt(0).toUpperCase() + (p.variant || "complement").slice(1)}NB</p>
                      {(p.use_text ?? true) ? (
                        <>
                          <p><span className="font-medium">Векторизація:</span> {p.vectorizer === "count" ? "CountVectorizer" : "TF-IDF"}</p>
                          <p><span className="font-medium">Н-грами:</span> ({p.ngram || "1,1"})</p>
                        </>
                      ) : (
                        <p><span className="font-medium">Режим:</span> features-only (без TF-IDF, ablation)</p>
                      )}
                      <p><span className="font-medium">Обробка:</span> {activePreprocessing.length > 0 ? activePreprocessing.join(", ") : "немає"}</p>
                    </>
                  );
                })()}
                {mid === "distilbert" && <p><span className="font-medium">Інтеграція:</span> {({ concat: "Concat", multiview: "Multi-view" } as any)[p.integration_mode || "concat"]}</p>}
                {(mid === "gin" || mid === "sage") && (
                  <>
                    <p><span className="font-medium">Hidden dim:</span> {p.hidden_dim || "128"}</p>
                    <p><span className="font-medium">Шарів:</span> {p.num_layers || (mid === "gin" ? "3" : "2")}</p>
                    <p><span className="font-medium">Dropout:</span> {p.dropout || "0.5"}</p>
                    <p><span className="font-medium">LR:</span> {p.learning_rate || "0.001"}</p>
                    <p><span className="font-medium">Epochs:</span> {p.epochs || "50"}</p>
                    {mid === "gin" && <p><span className="font-medium">Pooling:</span> {p.pooling || "mean"}</p>}
                    {mid === "sage" && <p><span className="font-medium">Aggregator:</span> {p.aggregator || "mean"}</p>}
                  </>
                )}
                <p><span className="font-medium">Ознаки:</span> {groups.length > 0 ? groups.map((g: string) => getGroupLabel(g, mid)).join(", ") : "вимкнено"}{activeCount > 0 ? ` (${activeCount} додаткових)` : ""}</p>
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
                <span className="font-medium text-sm">{getGroupLabel(groupKey, type)}</span>
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
                    <span>{f.label}</span>
                    {f.isGraph && (
                      <Badge variant="secondary" className="ml-1 text-[9px] px-1 py-0 h-4">
                        graph
                      </Badge>
                    )}
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
