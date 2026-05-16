import { useState } from "react";
import api from "./api";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Alert, AlertDescription } from "./components/ui/alert";
import {
  Globe,
  AlertCircle,
  Loader2,
  Sparkles,
  Check,
  X,
  HelpCircle,
} from "lucide-react";
import { cn } from "./lib/utils";

interface AnalyzeResponse {
  platform: string;
  post_url: string;
  post_id: string;
  author_handle: string;
  author_display_name: string | null;
  content: string;
  created_at: string;
  language: string | null;
  engagement: {
    replies: number;
    reposts: number;
    likes: number;
  };
  media_count: number;
  media_attachments: { type: string; url: string; description: string | null }[] | null;
  extraction: {
    is_news_claim: boolean;
    reasoning_for_is_news: string;
    claim_text: string | null;
    claim_subject: string | null;
    claim_type: string | null;
    author_stance: string;
    stance_reasoning: string;
    emotion: string;
    emotion_intensity: string;
    confidence_in_claim: string;
    confidence_reasoning: string;
    cited_sources: string[];
    has_url: boolean;
    language_detected: string | null;
    is_translation: boolean;
  };
  classification: {
    verdict?: string;
    confidence?: number;
    reasoning?: string;
    would_need_to_verify?: string[];
    error?: string;
    model?: string;
  } | null;
  summary: string;
}

type VerdictInfo = {
  label: string;
  badgeClass: string;
  borderClass: string;
  icon: typeof Check;
};

const VERDICT_LABELS: Record<string, VerdictInfo> = {
  TRUE: {
    label: "Ймовірно ПРАВДА",
    badgeClass:
      "bg-green-100 text-green-800 border-green-300 dark:bg-green-950/40 dark:text-green-200 dark:border-green-900",
    borderClass: "border-green-300 dark:border-green-900",
    icon: Check,
  },
  FALSE: {
    label: "Ймовірно ФЕЙК",
    badgeClass:
      "bg-red-100 text-red-800 border-red-300 dark:bg-red-950/40 dark:text-red-200 dark:border-red-900",
    borderClass: "border-red-300 dark:border-red-900",
    icon: X,
  },
  UNCERTAIN: {
    label: "НЕВИЗНАЧЕНО",
    badgeClass:
      "bg-yellow-100 text-yellow-800 border-yellow-300 dark:bg-yellow-950/40 dark:text-yellow-200 dark:border-yellow-900",
    borderClass: "border-yellow-300 dark:border-yellow-900",
    icon: HelpCircle,
  },
  UNKNOWN: {
    label: "Помилка",
    badgeClass:
      "bg-gray-100 text-gray-800 border-gray-300 dark:bg-gray-900 dark:text-gray-200 dark:border-gray-700",
    borderClass: "border-gray-300 dark:border-gray-700",
    icon: AlertCircle,
  },
};

const STANCE_LABELS: Record<string, string> = {
  supporting: "підтримує",
  skeptical: "скептично",
  neutral: "нейтрально",
  ironic: "іронічно",
  outraged: "обурено",
  amused: "розважально",
  sympathetic: "співчутливо",
  unclear: "незрозуміло",
};

const MODEL_OPTIONS = [
  { value: "claude-haiku", label: "Haiku (швидко)" },
  { value: "claude-sonnet", label: "Sonnet (баланс)" },
  { value: "claude-opus", label: "Opus (найрозумніший)" },
];

export default function RealWorldPage() {
  const [url, setUrl] = useState("");
  const [extractionModel, setExtractionModel] = useState("claude-haiku");
  const [classificationModel, setClassificationModel] = useState("claude-haiku");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);

  const handleAnalyze = async () => {
    if (!url.trim()) {
      setError("Введіть URL поста");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const { data } = await api.post<AnalyzeResponse>("/real_world/analyze", {
        url: url.trim(),
        extraction_model: extractionModel,
        classification_model: classificationModel,
      });
      setResult(data);
    } catch (err: any) {
      const detail = err?.response?.data?.detail || err?.message || "невідома";
      setError(`Помилка: ${detail}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-2xl font-semibold flex items-center gap-2">
          <Globe className="w-6 h-6" />
          Аналіз поста
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Введіть URL поста з Mastodon або Bluesky — LLM розпакує claim та класифікує його
        </p>
      </div>

      <Card>
        <CardContent className="pt-6 space-y-4">
          <div className="space-y-2">
            <Label>URL поста</Label>
            <Input
              placeholder="https://mastodon.social/@user/123... або https://bsky.app/profile/user.bsky.social/post/abc..."
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !loading) handleAnalyze();
              }}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-xs">Модель для extraction</Label>
              <select
                value={extractionModel}
                onChange={(e) => setExtractionModel(e.target.value)}
                className="w-full h-9 rounded-md border bg-background px-3 text-sm"
              >
                {MODEL_OPTIONS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <Label className="text-xs">Модель для classification</Label>
              <select
                value={classificationModel}
                onChange={(e) => setClassificationModel(e.target.value)}
                className="w-full h-9 rounded-md border bg-background px-3 text-sm"
              >
                {MODEL_OPTIONS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>
          </div>

          <Button onClick={handleAnalyze} disabled={loading || !url.trim()}>
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Аналізую…
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Проаналізувати
              </>
            )}
          </Button>

          <div className="text-xs text-muted-foreground">
            Підтримувані формати:
            <ul className="mt-1 ml-4 list-disc">
              <li>
                Mastodon: <code>https://&lt;instance&gt;/@user/123…</code>
              </li>
              <li>
                Bluesky: <code>https://bsky.app/profile/user/post/abc…</code>
              </li>
            </ul>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="w-4 h-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {result && <ResultCard result={result} />}
    </div>
  );
}

function ResultCard({ result }: { result: AnalyzeResponse }) {
  const { extraction, classification } = result;
  const verdict = classification?.verdict || "UNKNOWN";
  const verdictInfo = VERDICT_LABELS[verdict] || VERDICT_LABELS.UNKNOWN;
  const VerdictIcon = verdictInfo.icon;

  return (
    <div className="space-y-4">
      {/* Original Post */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between text-base">
            <span className="flex items-center gap-2 flex-wrap">
              <Badge variant="outline">{result.platform}</Badge>
              <span>{result.author_handle}</span>
              {result.author_display_name && (
                <span className="text-xs text-muted-foreground">
                  ({result.author_display_name})
                </span>
              )}
            </span>
            <a
              href={result.post_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-blue-600 hover:underline shrink-0"
            >
              Відкрити ↗
            </a>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm whitespace-pre-wrap">{result.content}</p>

          <div className="mt-4 flex flex-wrap gap-4 text-xs text-muted-foreground">
            <span>💬 {result.engagement.replies}</span>
            <span>🔁 {result.engagement.reposts}</span>
            <span>♥ {result.engagement.likes}</span>
            {result.language && <span>🌐 {result.language}</span>}
            {result.media_count > 0 && <span>📎 {result.media_count} медіа</span>}
            {result.created_at && (
              <span>
                {new Date(result.created_at).toLocaleString("uk-UA")}
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Extraction */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">LLM Extraction</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {!extraction.is_news_claim ? (
            <Alert>
              <AlertCircle className="w-4 h-4" />
              <AlertDescription>
                <strong>Пост НЕ містить перевіряємого news claim.</strong>
                <p className="text-xs mt-1">{extraction.reasoning_for_is_news}</p>
              </AlertDescription>
            </Alert>
          ) : (
            <>
              <div>
                <div className="text-xs text-muted-foreground mb-1">CLAIM (новина)</div>
                <p className="text-sm font-medium">{extraction.claim_text}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                {extraction.claim_type && (
                  <div>
                    <span className="text-xs text-muted-foreground">Тема: </span>
                    <Badge variant="secondary">{extraction.claim_type}</Badge>
                  </div>
                )}
                {extraction.claim_subject && (
                  <div>
                    <span className="text-xs text-muted-foreground">Суб'єкт: </span>
                    {extraction.claim_subject}
                  </div>
                )}
                <div>
                  <span className="text-xs text-muted-foreground">Stance автора: </span>
                  <Badge variant="outline">
                    {STANCE_LABELS[extraction.author_stance] || extraction.author_stance}
                  </Badge>
                </div>
                <div>
                  <span className="text-xs text-muted-foreground">Емоція: </span>
                  {extraction.emotion} ({extraction.emotion_intensity})
                </div>
                <div>
                  <span className="text-xs text-muted-foreground">Впевненість автора: </span>
                  {extraction.confidence_in_claim}
                </div>
                {extraction.language_detected && (
                  <div>
                    <span className="text-xs text-muted-foreground">Мова: </span>
                    {extraction.language_detected}
                    {extraction.is_translation && " (переклад)"}
                  </div>
                )}
              </div>

              {extraction.cited_sources.length > 0 && (
                <div className="flex flex-wrap items-center gap-1">
                  <span className="text-xs text-muted-foreground mr-1">
                    Згадані джерела:
                  </span>
                  {extraction.cited_sources.map((s, i) => (
                    <Badge key={i} variant="outline">
                      {s}
                    </Badge>
                  ))}
                </div>
              )}

              {extraction.stance_reasoning && (
                <div className="text-xs text-muted-foreground border-t pt-2">
                  <strong>Reasoning:</strong> {extraction.stance_reasoning}
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Classification */}
      {extraction.is_news_claim && classification && (
        <Card className={cn("border-2", verdictInfo.borderClass)}>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <VerdictIcon className="w-5 h-5" />
              Verdict
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <Badge className={cn(verdictInfo.badgeClass, "text-sm px-3 py-1")}>
                {verdictInfo.label}
              </Badge>
              {classification.confidence != null && (
                <span className="text-sm text-muted-foreground">
                  Впевненість:{" "}
                  <strong>{(classification.confidence * 100).toFixed(0)}%</strong>
                </span>
              )}
              {classification.model && (
                <span className="text-xs text-muted-foreground">
                  ({classification.model})
                </span>
              )}
            </div>

            {classification.reasoning && (
              <div>
                <div className="text-xs text-muted-foreground mb-1">Reasoning</div>
                <p className="text-sm">{classification.reasoning}</p>
              </div>
            )}

            {classification.would_need_to_verify &&
              classification.would_need_to_verify.length > 0 && (
                <div>
                  <div className="text-xs text-muted-foreground mb-1">
                    Що варто перевірити
                  </div>
                  <ul className="text-sm list-disc ml-5 space-y-1">
                    {classification.would_need_to_verify.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}

            {classification.error && (
              <Alert variant="destructive">
                <AlertDescription>{classification.error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}

      {/* Summary */}
      <Card>
        <CardContent className="pt-6">
          <div className="text-xs text-muted-foreground mb-1">Summary</div>
          <p className="text-sm">{result.summary}</p>
        </CardContent>
      </Card>
    </div>
  );
}
