import React, { useState } from "react";
import type { FeatureGroupDef } from "./types";

// ── Константи ─────────────────────────────────────────────────────────────────

const FEATURE_GROUPS: Record<string, FeatureGroupDef> = {
  semantic: {
    label: "Семантичні",
    description: "TF-IDF векторизація тексту — основа класифікації",
    features: [
      { key: "text", label: "Текст", type: "semantic" },
    ],
  },
  emotional: {
    label: "Емоційні",
    description: "Емоційний тон, невербальна емоційність та базові емоції за NRC лексиконом",
    features: [
      { key: "sentiment_score", label: "Тональність тексту", type: "emotional" },
      { key: "emotion_intensity", label: "Інтенсивність емоцій", type: "emotional" },
      { key: "emoji_count", label: "Кількість емодзі", type: "emotional" },
      { key: "exclamation_count", label: "Знаки оклику", type: "emotional" },
      // NRC Base Emotions (10)
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
  // Semantic
  "text",
  // Emotional
  "sentiment_score",
  "emotion_intensity",
  "emoji_count",
  "exclamation_count",
  "anger_score",
  "fear_score",
  "anticipation_score",
  "trust_score",
  "surprise_score",
  "sadness_score",
  "joy_score",
  "disgust_score",
  "positive_score",
  "negative_score",
  // Stylistic
  "caps_ratio",
  "ttr",
  "repetition_score",
  "avg_word_length",
  // Rhetorical
  "clickbait_score",
  "authority_refs",
  "pronoun_ratio",
  "question_count",
  // Social
  "upvote_ratio",
  "score_normalized",
  "num_comments_norm",
  "domain_credibility",
  "account_age_norm",
  "has_url",
];

const MODEL_OPTIONS = [
  {
    id: "nb",
    name: "Naive Bayes",
    desc: "MultinomialNB / ComplementNB + TF-IDF",
    group: "classical",
  },
  {
    id: "deberta",
    name: "DistilBERT",
    desc: "Fine-tuned трансформер + додаткові ознаки",
    group: "neural",
  },
  {
    id: "llm",
    name: "Gemini (Zero-Shot)",
    desc: "Zero-shot класифікація через OpenAI API",
    group: "llm",
  },
];

const MODEL_LABELS: Record<string, string> = {
  nb: "Naive Bayes",
  deberta: "DistilBERT",
  llm: "Gemini (Zero-Shot)",
};

const ENSEMBLE_STRATEGIES = [
  {
    id: "hard",
    name: "Hard Voting",
    desc: "Більшість голосів (majority label)",
  },
  { id: "soft", name: "Soft Voting", desc: "Середнє ймовірностей моделей" },
  {
    id: "weighted",
    name: "Weighted Voting",
    desc: "Зважена сума (ваги задаються вручну)",
  },
];

function buildDefaultMask(allTrue: boolean) {
  const mask: Record<string, boolean> = {};
  ALL_FEATURE_KEYS.forEach((k) => {
    mask[k] = allTrue;
  });
  return mask;
}

const DEFAULT_PARAMS: any = {
  nb: {
    variant: "multinomial",
    vectorizer: "tfidf",
    ngram: "1,1",
    alpha: "1.0",
    additional_groups: ["semantic"],
    feature_mask: buildDefaultMask(true),
    preprocessing: {
      removeUrls: true,
      removeMentions: true,
      cleaning: true,
      lowercase: true,
      removePunctuation: true,
      removeNumbers: false,
      removeStopwords: true,
      stemming: false,
      lemmatization: true,
    },
  },
  deberta: {
    integration_mode: "concat",
    additional_groups: ["semantic"],
    feature_mask: buildDefaultMask(false),
  },
  llm: {
    mode: "single",
    lang: "auto",
    additional_groups: ["semantic"],
    feature_mask: buildDefaultMask(true),
  },
};

function getDefaultParams(type: string) {
  const p = DEFAULT_PARAMS[type] || {};
  return {
    ...p,
    feature_mask: {
      ...(p.feature_mask || buildDefaultMask(true)),
    },
  };
}

// ── Стилі ─────────────────────────────────────────────────────────────────────

const S: any = {
  modeCard: (active: boolean) => ({
    flex: 1,
    padding: "28px 22px",
    borderRadius: 14,
    cursor: "pointer",
    border: `2px solid ${active ? "#4a90d9" : "#e0e0e0"}`,
    background: active ? "#f0f7ff" : "white",
    transition: "all 0.15s",
    userSelect: "none",
    textAlign: "center",
    minHeight: 120,
  }),
  modeIcon: { fontSize: 32, marginBottom: 10 },
  modeTitle: { fontWeight: 700, fontSize: 15, marginBottom: 6 },
  modeDesc: { color: "#888", fontSize: 12, lineHeight: 1.5 },
  stepBar: {
    display: "flex",
    alignItems: "center",
    marginBottom: 28,
    padding: "0 10px",
  },
  stepDot: (state: string) => ({
    width: 30,
    height: 30,
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background:
      state === "done" ? "#28a745" : state === "active" ? "#4a90d9" : "#e0e0e0",
    color: state !== "pending" ? "white" : "#999",
    fontWeight: 700,
    fontSize: 12,
    flexShrink: 0,
  }),
  stepLabel: (active: boolean) => ({
    fontSize: 10,
    marginTop: 4,
    color: active ? "#4a90d9" : "#666",
    fontWeight: active ? 600 : 400,
  }),
  connector: (done: boolean) => ({
    height: 2,
    flex: 1,
    background: done ? "#28a745" : "#e0e0e0",
    marginBottom: 20,
  }),
  primaryBtn: (disabled: boolean) => ({
    padding: "10px 28px",
    borderRadius: 8,
    border: "none",
    fontSize: 13,
    fontWeight: 600,
    background: disabled ? "#ccc" : "#4a90d9",
    color: disabled ? "#888" : "white",
    cursor: disabled ? "not-allowed" : "pointer",
  }),
  secondaryBtn: (disabled: boolean) => ({
    padding: "10px 28px",
    borderRadius: 8,
    border: `1px solid ${disabled ? "#ddd" : "#ccc"}`,
    fontSize: 13,
    fontWeight: 600,
    background: "white",
    color: disabled ? "#999" : "#333",
    cursor: disabled ? "not-allowed" : "pointer",
  }),
  accordion: { marginBottom: 14, borderRadius: 10, overflow: "hidden" },
  accordionHead: {
    padding: "13px 18px",
    fontWeight: 700,
    cursor: "pointer",
    background: "#eef1f5",
    fontSize: 13,
    display: "flex",
    alignItems: "center",
    gap: 8,
    borderRadius: 10,
  },
  accordionBody: { padding: "16px 18px 8px" },
  groupTitle: {
    fontWeight: 600,
    color: "#555",
    fontSize: 10,
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    marginBottom: 8,
  },
  groupToggle: (active: boolean) => ({
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "10px 14px",
    borderRadius: 10,
    marginBottom: 8,
    cursor: "pointer",
    userSelect: "none",
    border: `2px solid ${active ? "#4a90d9" : "#e0e0e0"}`,
    background: active ? "#f0f7ff" : "#fafafa",
    transition: "all 0.15s",
  }),
};

type ClassificationWizardProps = {
  trainingMode?: boolean;
  onConfigChange?: (cfg: any) => void;
};

export default function ClassificationWizard({
  trainingMode = false,
  onConfigChange,
}: ClassificationWizardProps) {
  const [mode, setMode] = useState<any>(null); // 'single' | 'ensemble'
  const [step, setStep] = useState<number>(0);
  const [selectedModel, setSelectedModel] = useState<string>("nb");
  const [selectedModels, setSelectedModels] = useState<any[]>([]);
  const [modelParams, setModelParams] = useState<any>({});
  const [ensembleStrategy, setEnsembleStrategy] = useState<string>("hard");
  const [weights, setWeights] = useState<any>({});

  // ── Кроки ───────────────────────────────────────────────────────────────────
  const STEPS: any = {
    single: ["Режим", "Модель", "Параметри", "Підсумок"],
    ensemble: ["Режим", "Стратегія", "Моделі", "Параметри", "Підсумок"],
  };
  const stepLabels = STEPS[mode] || STEPS.single;
  const lastStep = stepLabels.length - 1;
  const isLastStep = mode !== null && step === lastStep;

  // ── Хелпери параметрів ──────────────────────────────────────────────────────
  const setParam = (type: string, key: string, val: any) =>
    setModelParams((prev: any) => ({
      ...prev,
      [type]: { ...prev[type], [key]: val },
    }));

  const toggleGroup = (type: string, groupName: string) =>
    setModelParams((prev: any) => {
      const p = prev[type] || getDefaultParams(type);
      const groups = [...(p.additional_groups || [])];
      const mask = { ...(p.feature_mask || buildDefaultMask(true)) };
      const idx = groups.indexOf(groupName);
      if (idx >= 0) {
        groups.splice(idx, 1);
        FEATURE_GROUPS[groupName].features.forEach((f: any) => {
          mask[f.key] = false;
        });
      } else {
        groups.push(groupName);
        FEATURE_GROUPS[groupName].features.forEach((f: any) => {
          mask[f.key] = true;
        });
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
      // stemming and lemmatization are mutually exclusive
      if (key === "stemming" && updated.stemming) updated.lemmatization = false;
      if (key === "lemmatization" && updated.lemmatization) updated.stemming = false;
      return { ...prev, [type]: { ...prev[type], preprocessing: updated } };
    });

  const toggleModel = (id: string) =>
    setSelectedModels((prev: any[]) =>
      prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id],
    );

  // ── Навігація ───────────────────────────────────────────────────────────────
  const canNext = () => {
    if (step === 0) return mode !== null;
    if (mode === "single") {
      if (step === 1) return !!selectedModel;
      return true;
    }
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
        models.forEach((t: string) => {
          if (!prev[t]) updates[t] = getDefaultParams(t);
        });
        return Object.keys(updates).length ? { ...prev, ...updates } : prev;
      });

      if (mode === "ensemble") {
        setWeights((prev: any) => {
          const next: any = {};
          selectedModels.forEach((m: string) => {
            next[m] = prev[m] !== undefined ? prev[m] : 1;
          });
          return next;
        });
      }
    }
    setStep((s) => s + 1);
  };

  const handleBack = () => {
    setStep((s) => s - 1);
  };

  // ── Формування запиту ───────────────────────────────────────────────────────
  const buildRequest = () => {
    const modelTypes = mode === "single" ? [selectedModel] : selectedModels;
    const models = modelTypes.map((type: string) => {
      const p = modelParams[type] || getDefaultParams(type);
      const groups = p.additional_groups || [];
      const mask = p.feature_mask || buildDefaultMask(false);
      const hasFeatures = Object.values(mask).some(Boolean);

      const model: any = { model: type };

      if (hasFeatures) {
        model.additional_features = { groups, mask };
      } else {
        model.additional_features = null;
      }

      if (type === "nb") {
        model.variant = p.variant || "multinomial";
        model.vectorizer = p.vectorizer || "tfidf";
        model.ngram_range = p.ngram || "1,1";
        model.alpha = p.alpha || "1.0";
      }
      if (type === "deberta") {
        model.integration_mode = p.integration_mode || "concat";
      }
      if (type === "llm") {
        model.mode = p.mode || "single";
      }
      return model;
    });

    const nbParams = modelParams.nb || getDefaultParams("nb");
    const firstModel = modelTypes[0];
    const preprocessing = firstModel === "nb"
      ? nbParams.preprocessing
      : { removeUrls: true, removeMentions: true, cleaning: true };
    const req: any = { mode, models, preprocessing };
    if (mode === "ensemble") {
      req.ensemble = { strategy: ensembleStrategy };
      if (ensembleStrategy === "weighted") {
        const total =
          (Object.values(weights).reduce(
            (s: number, v: any) => s + Number(v),
            0,
          ) as number) || 1;
        const normWeights: any = {};
        selectedModels.forEach((m: string) => {
          normWeights[m] = (weights[m] || 0) / total;
        });
        req.ensemble.weights = normWeights;
      }
    }
    return req;
  };

  // trainingMode — передаємо конфігурацію батьківському компоненту
  React.useEffect(() => {
    if (!trainingMode || !onConfigChange || !mode) return;
    const config = buildRequest();
    onConfigChange(config);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    trainingMode,
    mode,
    selectedModel,
    selectedModels,
    modelParams,
    ensembleStrategy,
    weights,
  ]);

  // ── Крок 0: Вибір режиму ──────────────────────────────────────────────────
  const renderModeSelect = () => (
    <div>
      <p style={{ fontWeight: 600, marginBottom: 20, fontSize: 15 }}>
        Оберіть режим класифікації
      </p>
      <div style={{ display: "flex", gap: 16 }}>
        <div
          onClick={() => setMode("single")}
          style={S.modeCard(mode === "single")}
        >
          <div style={S.modeIcon}>👤</div>
          <div style={S.modeTitle}>Одна модель</div>
          <div style={S.modeDesc}>
            Обрати один алгоритм та налаштувати його параметри
          </div>
        </div>
        <div
          onClick={() => setMode("ensemble")}
          style={S.modeCard(mode === "ensemble")}
        >
          <div style={S.modeIcon}>&#x1F465;</div>
          <div style={S.modeTitle}>Ансамбль</div>
          <div style={S.modeDesc}>
            Два або більше алгоритмів — голосування або зважена сума
          </div>
        </div>
      </div>
      {!mode && (
        <p style={{ marginTop: 16, color: "#888", fontSize: 13 }}>
          Клікніть на картку щоб обрати режим
        </p>
      )}
    </div>
  );

  // ── Крок 1A: Вибір однієї моделі ─────────────────────────────────────────
  const renderSingleModelSelect = () => (
    <div>
      <p style={{ fontWeight: 600, marginBottom: 14, fontSize: 15 }}>
        Оберіть алгоритм
      </p>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {MODEL_OPTIONS.map((m) => (
          <label
            key={m.id}
            className={`model-option ${selectedModel === m.id ? "selected" : ""}`}
            style={{ cursor: "pointer" }}
          >
            <input
              type="radio"
              name="singleModel"
              value={m.id}
              checked={selectedModel === m.id}
              onChange={() => setSelectedModel(m.id)}
            />
            <div className="model-info">
              <span className="model-name">{m.name}</span>
              <span className="model-description">{m.desc}</span>
            </div>
          </label>
        ))}
      </div>
    </div>
  );

  // ── Крок 1B: Вибір моделей для ансамблю ──────────────────────────────────
  const renderEnsembleModelSelect = () => {
    const groups = [
      { label: "Класичні", ids: ["nb"] },
      { label: "Нейромережеві", ids: ["deberta"] },
      { label: "LLM", ids: ["llm"] },
    ];
    return (
      <div>
        <p style={{ marginBottom: 16, color: "#666" }}>
          Оберіть мінімум 2 моделі для ансамблю
        </p>
        {groups.map((g) => (
          <div key={g.label} style={{ marginBottom: 18 }}>
            <p style={S.groupTitle}>{g.label}</p>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {MODEL_OPTIONS.filter((m) => g.ids.includes(m.id)).map((m) => {
                const checked = selectedModels.includes(m.id);
                return (
                  <label
                    key={m.id}
                    className={`model-option ${checked ? "selected" : ""}`}
                    style={{ cursor: "pointer" }}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleModel(m.id)}
                    />
                    <div className="model-info">
                      <span className="model-name">{m.name}</span>
                      <span className="model-description">{m.desc}</span>
                    </div>
                  </label>
                );
              })}
            </div>
          </div>
        ))}
        {selectedModels.length === 1 && (
          <p style={{ color: "#dc3545", fontSize: 13 }}>
            Оберіть ще хоча б одну модель
          </p>
        )}
      </div>
    );
  };

  // ── Крок 2: Параметри обраних моделей ─────────────────────────────────────
  const renderModelParams = (type: string) => {
    const p = modelParams[type] || getDefaultParams(type);
    const upd = (key: string, val: any) => setParam(type, key, val);
    const groups = p.additional_groups || [];

    switch (type) {
      case "nb": {
        const pp = p.preprocessing || {};
        return (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div>
                <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 4, color: "#333" }}>
                  Варіант
                </label>
                <select
                  className="translator-select"
                  value={p.variant || "multinomial"}
                  onChange={(e) => upd("variant", e.target.value)}
                  style={{ width: "100%", maxWidth: 240 }}
                >
                  <option value="multinomial">MultinomialNB</option>
                  <option value="complement">ComplementNB</option>
                </select>
              </div>
              <div>
                <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 4, color: "#333" }}>
                  Векторизація
                </label>
                <select
                  className="translator-select"
                  value={p.vectorizer || "tfidf"}
                  onChange={(e) => upd("vectorizer", e.target.value)}
                  style={{ width: "100%", maxWidth: 240 }}
                >
                  <option value="tfidf">TF-IDF</option>
                  <option value="count">CountVectorizer</option>
                </select>
              </div>
              <div>
                <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 4, color: "#333" }}>
                  Н-грами
                </label>
                <select
                  className="translator-select"
                  value={p.ngram || "1,1"}
                  onChange={(e) => upd("ngram", e.target.value)}
                  style={{ width: "100%", maxWidth: 240 }}
                >
                  <option value="1,1">(1,1)</option>
                  <option value="1,2">(1,2)</option>
                  <option value="1,3">(1,3)</option>
                </select>
              </div>
              <div>
                <label style={{ display: "block", fontSize: 12, fontWeight: 600, marginBottom: 4, color: "#333" }}>
                  Alpha (Laplace smoothing)
                </label>
                <select
                  className="translator-select"
                  value={p.alpha || "1.0"}
                  onChange={(e) => upd("alpha", e.target.value)}
                  style={{ width: "100%", maxWidth: 240 }}
                >
                  <option value="0.01">0.01</option>
                  <option value="0.1">0.1</option>
                  <option value="0.5">0.5</option>
                  <option value="1.0">1.0</option>
                  <option value="2.0">2.0</option>
                </select>
              </div>
            </div>
            <div>
              <label
                style={{
                  fontWeight: 600,
                  marginBottom: 10,
                  fontSize: 13,
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  cursor: "pointer",
                }}
              >
                <input
                  type="checkbox"
                  checked={(() => {
                    const keys = ["removeUrls", "removeMentions", "cleaning", "lowercase", "removeStopwords", "removePunctuation", "removeNumbers"];
                    return keys.every(k => pp[k]);
                  })()}
                  onChange={(e) => {
                    const val = e.target.checked;
                    setModelParams((prev: any) => {
                      const oldPp = prev[type]?.preprocessing || {};
                      const updated: any = { ...oldPp };
                      ["removeUrls", "removeMentions", "cleaning", "lowercase", "removeStopwords", "removePunctuation", "removeNumbers"].forEach(k => { updated[k] = val; });
                      return { ...prev, [type]: { ...prev[type], preprocessing: updated } };
                    });
                  }}
                />
                Попередня обробка тексту
              </label>
              <div
                className="checkbox-grid"
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "6px 16px",
                  fontSize: 12,
                }}
              >
                {[
                  { key: "removeUrls", label: "Видалення URL-посилань" },
                  { key: "removeMentions", label: "Видалення @згадок" },
                  { key: "cleaning", label: "Очищення пробілів" },
                  { key: "lowercase", label: "Нижній регістр" },
                  { key: "removeStopwords", label: "Видалення стоп-слів" },
                  { key: "removePunctuation", label: "Видалення пунктуації" },
                  { key: "removeNumbers", label: "Видалення чисел" },
                ].map((opt) => (
                  <label key={opt.key} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={pp[opt.key] || false}
                      onChange={() => togglePreprocessing(type, opt.key)}
                    />
                    <span>{opt.label}</span>
                  </label>
                ))}
                <div style={{ gridColumn: "1 / -1", marginTop: 6 }}>
                  <span style={{ fontSize: 12, fontWeight: 500, color: "#555" }}>Нормалізація слів:</span>
                  <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
                    {[
                      { value: "none", label: "Немає" },
                      { value: "stemming", label: "Стемінг" },
                      { value: "lemmatization", label: "Лематизація" },
                    ].map((opt) => {
                      const current = pp.stemming ? "stemming" : pp.lemmatization ? "lemmatization" : "none";
                      return (
                        <label key={opt.value} className="checkbox-label">
                          <input
                            type="radio"
                            name={`normalization-${type}`}
                            checked={current === opt.value}
                            onChange={() => {
                              setModelParams((prev: any) => {
                                const oldPp = prev[type]?.preprocessing || {};
                                return {
                                  ...prev,
                                  [type]: {
                                    ...prev[type],
                                    preprocessing: {
                                      ...oldPp,
                                      stemming: opt.value === "stemming",
                                      lemmatization: opt.value === "lemmatization",
                                    },
                                  },
                                };
                              });
                            }}
                          />
                          <span>{opt.label}</span>
                        </label>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
            <AdditionalFeaturesSection
              type={type}
              groups={groups}
              mask={p.feature_mask}
              toggleGroup={toggleGroup}
              toggleFeature={toggleFeature}

            />
          </div>
        );
      }

      case "deberta":
        return (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <p style={{ fontSize: 13, color: "#888" }}>
              Fine-tuned DistilBERT — класифікація тексту через трансформер (concat mode)
            </p>
            <AdditionalFeaturesSection
              type={type}
              groups={groups}
              mask={p.feature_mask}
              toggleGroup={toggleGroup}
              toggleFeature={toggleFeature}
            />
          </div>
        );

      case "llm":
        return (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              {[
                { id: "single", name: "Single", desc: "1 запит, temp=0, детерміновано" },
                { id: "bagging", name: "Bagging", desc: "N=3 запити, temp=0.7, majority voting" },
              ].map((m: any) => (
                <label
                  key={m.id}
                  className={`model-option ${(p.mode || "single") === m.id ? "selected" : ""
                    }`}
                  style={{ fontSize: 13, flex: 1 }}
                >
                  <input
                    type="radio"
                    name={`llm_mode_${type}`}
                    value={m.id}
                    checked={(p.mode || "single") === m.id}
                    onChange={() => upd("mode", m.id)}
                  />
                  <div className="model-info">
                    <span className="model-name">{m.name}</span>
                    <span className="model-description">{m.desc}</span>
                  </div>
                </label>
              ))}
            </div>
            <label className="param-inline">
              <span>Мова промпту:</span>
              <select
                className="translator-select"
                value={p.lang || "auto"}
                onChange={(e) => upd("lang", e.target.value)}
              >
                <option value="uk">Українська</option>
                <option value="ru">Російська</option>
                <option value="auto">Авто</option>
              </select>
            </label>
            <AdditionalFeaturesSection
              type={type}
              groups={groups}
              mask={p.feature_mask}
              toggleGroup={toggleGroup}
              toggleFeature={toggleFeature}
            />
          </div>
        );

      default:
        return null;
    }
  };

  const renderConfigSummary = () => {
    const modelIds = mode === "single" ? [selectedModel] : selectedModels;
    const strategyObj = ENSEMBLE_STRATEGIES.find((s: any) => s.id === ensembleStrategy);
    const total =
      (Object.values(weights).reduce((s: number, v: any) => s + Number(v), 0) as number) ||
      1;

    return (
      <div style={{ padding: 14, background: "#f0f4f8", borderRadius: 10, fontSize: 12, lineHeight: 1.8 }}>
        <p style={{ fontWeight: 700, marginBottom: 6, fontSize: 13 }}>Підсумок конфігурації</p>
        <p>
          <strong>Режим:</strong> {mode === "single" ? "Одна модель" : "Ансамбль"}
        </p>
        {mode === "ensemble" && (
          <p>
            <strong>Стратегія:</strong> {strategyObj?.name}
          </p>
        )}
        {mode === "ensemble" && ensembleStrategy === "weighted" && (
          <p>
            <strong>Ваги:</strong>{" "}
            {modelIds
              .map(
                (m: string) =>
                  `${MODEL_LABELS[m]} ${(Number(weights[m] || 0) / total).toFixed(2)}`,
              )
              .join(", ")}
          </p>
        )}

        {modelIds.map((mid: string) => {
          const p = modelParams[mid] || getDefaultParams(mid);
          const groups = p.additional_groups || [];
          const mask = p.feature_mask || {};
          const activeCount =
            groups.length > 0
              ? Object.entries(mask).filter(([k, v]: any) => v && ALL_FEATURE_KEYS.includes(k)).length
              : 0;
          return (
            <div key={mid} style={{ marginTop: 8, paddingTop: 8, borderTop: "1px solid #d0d8e0" }}>
              <p style={{ fontWeight: 700, fontSize: 12, marginBottom: 2 }}>
                {MODEL_LABELS[mid] || mid}
              </p>
              {mid === "nb" && (() => {
                const pp = p.preprocessing || {};
                const ppLabels: any = {
                  removeUrls: "URL",
                  removeMentions: "@згадки",
                  cleaning: "пробіли",
                  lowercase: "lowercase",
                  removeStopwords: "стоп-слова",
                  stemming: "стемінг",
                  lemmatization: "лематизація",
                  removePunctuation: "пунктуація",
                  removeNumbers: "числа",
                };
                const activePreprocessing = Object.entries(pp)
                  .filter(([, v]) => v)
                  .map(([k]) => ppLabels[k] || k);
                return (
                  <>
                    <p>
                      <strong>Варіант:</strong>{" "}
                      {(p.variant || "multinomial").charAt(0).toUpperCase() +
                        (p.variant || "multinomial").slice(1)}NB
                    </p>
                    <p>
                      <strong>Векторизація:</strong>{" "}
                      {p.vectorizer === "count" ? "CountVectorizer" : "TF-IDF"}
                    </p>
                    <p>
                      <strong>Н-грами:</strong> ({p.ngram || "1,1"})
                    </p>
                    <p>
                      <strong>Обробка:</strong>{" "}
                      {activePreprocessing.length > 0 ? activePreprocessing.join(", ") : "немає"}
                    </p>
                  </>
                );
              })()}
              {mid === "deberta" && (
                <p>
                  <strong>Інтеграція:</strong>{" "}
                  {({ concat: "Concat", multiview: "Multi-view" } as any)[
                    p.integration_mode || "concat"
                  ]}
                </p>
              )}
              {mid === "llm" && (
                <>
                  <p>
                    <strong>Режим:</strong> {(p.mode || "single") === "single" ? "Single" : "Bagging"}
                  </p>
                  <p>
                    <strong>Мова:</strong>{" "}
                    {({ uk: "Українська", ru: "Російська", auto: "Авто" } as any)[p.lang || "auto"]}
                  </p>
                </>
              )}
              <p>
                <strong>Ознаки:</strong>{" "}
                {groups.length > 0 ? groups.map((g: string) => FEATURE_GROUPS[g]?.label).join(", ") : "вимкнено"}
                {activeCount > 0 ? ` (${activeCount} додаткових)` : ""}
              </p>
            </div>
          );
        })}

        <div style={{ marginTop: 16, textAlign: "right" }}>
          <button onClick={() => setStep(mode === "single" ? 2 : 3)} style={S.secondaryBtn(false)}>
            Змінити
          </button>
        </div>
      </div>
    );
  };

  const renderParamsStep = () => {
    const models = mode === "single" ? [selectedModel] : selectedModels;
    if (models.length === 1) {
      return (
        <div>
          <p style={{ fontWeight: 600, marginBottom: 14, fontSize: 15 }}>
            Параметри {MODEL_LABELS[models[0]] || models[0]}
          </p>
          {renderModelParams(models[0])}
        </div>
      );
    }
    return (
      <div>
        <p style={{ marginBottom: 16, color: "#888", fontSize: 12 }}>
          Налаштуйте параметри кожної моделі
        </p>
        {models.map((type: string) => (
          <details key={type} open style={S.accordion}>
            <summary style={S.accordionHead}>
              <span>{MODEL_LABELS[type] || type}</span>
            </summary>
            <div style={S.accordionBody}>{renderModelParams(type)}</div>
          </details>
        ))}
      </div>
    );
  };

  const renderSummaryStep = () => <div>{renderConfigSummary()}</div>;

  const renderEnsembleStrategy = () => {
    const total = Object.values(weights).reduce(
      (s: number, v: any) => s + Number(v),
      0,
    ) as number;
    return (
      <div>
        <p style={{ marginBottom: 16, color: "#888", fontSize: 12 }}>Оберіть стратегію об'єднання прогнозів</p>
        <div className="model-grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {ENSEMBLE_STRATEGIES.map((s: any) => (
            <label key={s.id} className={`model-option ${ensembleStrategy === s.id ? "selected" : ""}`}>
              <input
                type="radio"
                name="ensStrategy"
                value={s.id}
                checked={ensembleStrategy === s.id}
                onChange={() => setEnsembleStrategy(s.id)}
              />
              <div className="model-info">
                <span className="model-name">{s.name}</span>
                <span className="model-description">{s.desc}</span>
              </div>
            </label>
          ))}
        </div>
        {ensembleStrategy === "weighted" && (
          <div style={{ marginTop: 20 }}>
            <p style={{ fontWeight: 600, fontSize: 13, marginBottom: 6 }}>Ваги моделей</p>
            <p style={{ color: "#888", fontSize: 11, marginBottom: 14 }}>
              Автоматично нормалізуються до суми 1.0
            </p>
            {selectedModels.map((mid: string) => {
              const norm = total > 0 ? Number(weights[mid] || 0) / total : 0;
              return (
                <div key={mid} style={{ marginBottom: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontWeight: 600, fontSize: 13 }}>{MODEL_LABELS[mid]}</span>
                    <span style={{ color: "#4a90d9", fontWeight: 600, fontSize: 12 }}>
                      {norm.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="10"
                    step="0.5"
                    value={weights[mid] || 0}
                    onChange={(e) => setWeights((p: any) => ({ ...p, [mid]: parseFloat(e.target.value) }))}
                    style={{ width: "100%" }}
                  />
                </div>
              );
            })}
          </div>
        )}
      </div>
    );
  };

  const renderCurrentStep = () => {
    if (step === 0) return renderModeSelect();
    if (mode === "single") {
      if (step === 1) return renderSingleModelSelect();
      if (step === 2) return renderParamsStep();
      if (step === 3) return renderSummaryStep();
      return null;
    }
    if (step === 1) return renderEnsembleStrategy();
    if (step === 2) return renderEnsembleModelSelect();
    if (step === 3) return renderParamsStep();
    if (step === 4) return renderSummaryStep();
    return null;
  };

  return (
    <div style={{ maxWidth: 780, margin: "0 auto" }}>
      {mode !== null && (
        <div style={S.stepBar}>
          {stepLabels.map((label: string, i: number) => {
            const state = i < step ? "done" : i === step ? "active" : "pending";
            return (
              <React.Fragment key={i}>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    flex: 1,
                    opacity: state === "pending" ? 0.4 : 1,
                  }}
                >
                  <div style={S.stepDot(state)}>{state === "done" ? "\u2713" : i + 1}</div>
                  <span style={S.stepLabel(state === "active")}>{label}</span>
                </div>
                {i < stepLabels.length - 1 && <div style={S.connector(i < step)} />}
              </React.Fragment>
            );
          })}
        </div>
      )}

      <div className="section-block">{renderCurrentStep()}</div>

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 16 }}>
        <div>
          {step > 0 && (
            <button onClick={handleBack} style={S.secondaryBtn(false)}>
              &larr; Назад
            </button>
          )}
        </div>
        <div>
          {!isLastStep && (
            <button
              onClick={handleNext}
              disabled={!canNext()}
              style={S.primaryBtn(!canNext())}
            >
              Далі &rarr;
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Секція додаткових ознак (3 групи з toggle + чекбокси) ───────────────────

function AdditionalFeaturesSection({ type, groups, mask, toggleGroup, toggleFeature }: any) {
  const featureMask = mask || {};

  return (
    <div>
      <p style={{ fontWeight: 600, fontSize: 13, marginBottom: 6 }}>Ознаки:</p>
      <p style={{ color: "#888", fontSize: 11, marginBottom: 10 }}>
        Оберіть групи ознак для класифікації:
      </p>
      {Object.entries(FEATURE_GROUPS).map(([groupKey, groupDef]: any) => {
        const isActive = groups.includes(groupKey);
        const hasSubFeatures = groupDef.features.length > 0;
        const activeInGroup = isActive && hasSubFeatures
          ? groupDef.features.filter((f: any) => featureMask[f.key]).length
          : 0;

        return (
          <div key={groupKey} style={{ marginBottom: 10 }}>
            <div style={S.groupToggle(isActive)} onClick={() => toggleGroup(type, groupKey)}>
              <input type="checkbox" checked={isActive} readOnly style={{ accentColor: "#4a90d9" }} />
              <div style={{ flex: 1 }}>
                <span style={{ fontWeight: 600, fontSize: 13 }}>{groupDef.label}</span>
                <span style={{ color: "#888", fontSize: 11, marginLeft: 8 }}>
                  {groupDef.description}
                </span>
              </div>
              {isActive && hasSubFeatures && (
                <span style={{ color: "#4a90d9", fontSize: 11, fontWeight: 600 }}>
                  {activeInGroup}/{groupDef.features.length}
                </span>
              )}
            </div>

            {isActive && hasSubFeatures && (
              <div style={{ paddingLeft: 36, paddingBottom: 4 }}>
                {groupDef.features.map((f: any) => (
                  <label
                    key={f.key}
                    className="checkbox-label"
                    style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, fontSize: 12 }}
                  >
                    <input
                      type="checkbox"
                      checked={featureMask[f.key] || false}
                      onChange={() => toggleFeature(type, f.key)}
                    />
                    <span>{f.label}</span>
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

