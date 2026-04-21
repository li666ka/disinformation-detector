// ui/src/VerificationPage.tsx
import React, { useState, useEffect } from "react";
import api from "./api";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./components/ui/card";
import { Textarea } from "./components/ui/textarea";
import { Label } from "./components/ui/label";
import { Checkbox } from "./components/ui/checkbox";
import { Progress } from "./components/ui/progress";
import { Badge } from "./components/ui/badge";
import {
    ShieldCheck, Loader2, AlertTriangle, CheckCircle2, HelpCircle,
    Newspaper, MessageCircle, Sparkles, Clock, ExternalLink, Target,
    ChevronDown, ChevronRight,
} from "lucide-react";
import { cn } from "./lib/utils";
import { toast } from "sonner";
import type {
    VerificationResult, Verdict, Evidence, EvidenceStance, Claim,
    EvidenceBundle,
} from "./types";

const EXAMPLE_TEXTS = [
    "Pfizer vaccine causes autism in kids. Big pharma is hiding this!",
    "OMG so excited for the new Taylor Swift album tomorrow!!",
    "Breaking: Ukraine received Patriot missile systems from US allies.",
];

interface VerificationPageProps {
    initialText?: string;
}

export default function VerificationPage({ initialText = "" }: VerificationPageProps = {}) {
    const [text, setText] = useState(initialText);

    useEffect(() => {
        if (initialText) {
            setText(initialText);
        }
    }, [initialText]);
    const [loading, setLoading] = useState(false);
    const [useLlm, setUseLlm] = useState(true);
    const [result, setResult] = useState<VerificationResult | null>(null);
    const [step, setStep] = useState<string>("");

    const runVerification = async () => {
        if (!text.trim()) {
            toast.error("Введіть текст для перевірки");
            return;
        }

        setLoading(true);
        setResult(null);
        setStep("Витягування тверджень...");

        // Симуляція кроків для UX (реальний pipeline одним запитом)
        const stepTimeouts: NodeJS.Timeout[] = [];
        stepTimeouts.push(setTimeout(() => setStep("Пошук evidence у джерелах..."), 4000));
        stepTimeouts.push(setTimeout(() => setStep("Аналіз stance кожного evidence..."), 12000));
        stepTimeouts.push(setTimeout(() => setStep("Формування вердикту..."), 25000));

        try {
            const { data } = await api.post<VerificationResult>("/verification/verify", {
                text: text.trim(),
                max_claims: 2,
                limit_rss: 5,
                limit_social: 10,
                use_llm_verdict: useLlm,
            });
            setResult(data);
            toast.success(`Перевірка завершена за ${data.total_time_seconds.toFixed(1)}с`);
        } catch (err: any) {
            toast.error(err.response?.data?.detail || "Помилка перевірки");
        } finally {
            stepTimeouts.forEach(clearTimeout);
            setLoading(false);
            setStep("");
        }
    };

    return (
        <div className="space-y-6">
            {/* Header */}
            <div>
                <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
                    <ShieldCheck className="h-6 w-6 text-primary" />
                    Верифікація твердження
                </h2>
                <p className="text-muted-foreground mt-1">
                    Multi-hop pipeline: витягування твердження → пошук evidence → stance → вердикт
                </p>
            </div>

            {/* Input */}
            <Card>
                <CardHeader>
                    <CardTitle className="text-lg">Текст для перевірки</CardTitle>
                    <CardDescription>
                        Вставте пост, твіт, заголовок новини або будь-яке твердження
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="space-y-2">
                        <Label htmlFor="verify-text">Текст</Label>
                        <Textarea
                            id="verify-text"
                            value={text}
                            onChange={(e) => setText(e.target.value)}
                            placeholder="Наприклад: Pfizer vaccine causes autism in kids..."
                            rows={5}
                            disabled={loading}
                            className="resize-none"
                        />
                        <p className="text-xs text-muted-foreground">
                            {text.length} символів
                        </p>
                    </div>

                    {/* Examples */}
                    <div>
                        <p className="text-xs text-muted-foreground mb-2">Приклади для тесту:</p>
                        <div className="flex flex-wrap gap-2">
                            {EXAMPLE_TEXTS.map((ex, i) => (
                                <Button
                                    key={i}
                                    variant="outline"
                                    size="sm"
                                    onClick={() => setText(ex)}
                                    disabled={loading}
                                    className="text-xs"
                                >
                                    {ex.slice(0, 40)}…
                                </Button>
                            ))}
                        </div>
                    </div>

                    {/* Options */}
                    <div className="flex items-center gap-2">
                        <Checkbox
                            id="use-llm"
                            checked={useLlm}
                            onCheckedChange={(v) => setUseLlm(v === true)}
                            disabled={loading}
                        />
                        <Label htmlFor="use-llm" className="text-sm font-normal cursor-pointer">
                            LLM aggregation для вердикту (повільніше, але краще обґрунтування)
                        </Label>
                    </div>

                    <Button
                        onClick={runVerification}
                        disabled={loading || !text.trim()}
                        size="lg"
                        className="w-full"
                    >
                        {loading ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Перевіряю...
                            </>
                        ) : (
                            <>
                                <ShieldCheck className="mr-2 h-4 w-4" />
                                Запустити перевірку
                            </>
                        )}
                    </Button>

                    {loading && step && (
                        <div className="space-y-2">
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                <Loader2 className="h-3 w-3 animate-spin" />
                                {step}
                            </div>
                            <Progress value={undefined} className="h-1" />
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* Result */}
            {result && <VerificationResultDisplay result={result} />}
        </div>
    );
}

// ──────────────────────────────────────────────────────────────────────────

function VerificationResultDisplay({ result }: { result: VerificationResult }) {
    const hasClaims = result.claims.length > 0;

    return (
        <div className="space-y-6">
            {/* Overall verdict */}
            {result.overall_verdict && (
                <VerdictCard verdict={result.overall_verdict} isOverall={true} />
            )}

            {/* No claims case */}
            {!hasClaims && (
                <Card className="border-amber-200 dark:border-amber-900/50">
                    <CardContent className="py-6">
                        <div className="flex items-start gap-3">
                            <HelpCircle className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
                            <div>
                                <h3 className="text-sm font-semibold">Немає перевірюваних тверджень</h3>
                                <p className="text-xs text-muted-foreground mt-1">
                                    У тексті не знайдено фактичних тверджень, які можна верифікувати.
                                    Можливо, це емоційна реакція, питання або особистий досвід.
                                </p>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Per-claim details */}
            {result.claims.map((claim, i) => {
                const bundle = result.evidence_bundles[i];
                const verdict = result.verdicts[i];
                return (
                    <ClaimSection
                        key={i}
                        claim={claim}
                        bundle={bundle}
                        verdict={verdict}
                        index={i}
                        total={result.claims.length}
                    />
                );
            })}

            {/* Timing footer */}
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                Загальний час: {result.total_time_seconds.toFixed(1)}с
            </div>
        </div>
    );
}

// ──────────────────────────────────────────────────────────────────────────

function ClaimSection({
    claim, bundle, verdict, index, total,
}: {
    claim: Claim;
    bundle: EvidenceBundle;
    verdict: Verdict;
    index: number;
    total: number;
}) {
    const [expanded, setExpanded] = useState(total === 1);  // auto-expand якщо тільки один claim

    const allEvidence = [...bundle.rss_evidence, ...bundle.social_evidence];
    const supports = allEvidence.filter(e => e.stance === "supports");
    const refutes = allEvidence.filter(e => e.stance === "refutes");
    const unrelated = allEvidence.filter(e => e.stance === "unrelated");
    const unknown = allEvidence.filter(e => e.stance === "unknown" || !e.stance);

    return (
        <Card>
            <CardHeader className="cursor-pointer" onClick={() => setExpanded(!expanded)}>
                <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 text-xs text-muted-foreground mb-1">
                            {total > 1 && <span>Claim {index + 1} з {total}</span>}
                            <Badge variant="outline" className="text-[10px]">
                                {claim.claim_type}
                            </Badge>
                            {claim.entities.length > 0 && (
                                <span>· {claim.entities.slice(0, 3).join(", ")}</span>
                            )}
                        </div>
                        <CardTitle className="text-base leading-snug">
                            "{claim.text}"
                        </CardTitle>
                    </div>
                    <Button variant="ghost" size="sm" className="shrink-0 h-7 w-7 p-0">
                        {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                    </Button>
                </div>
            </CardHeader>

            {expanded && (
                <CardContent className="space-y-4 pt-0">
                    {/* Per-claim verdict (якщо не overall) */}
                    {total > 1 && <VerdictCard verdict={verdict} isOverall={false} compact />}

                    {/* Evidence summary */}
                    <div className="grid grid-cols-4 gap-2">
                        <SummaryCell count={supports.length} label="Підтримують" color="green" />
                        <SummaryCell count={refutes.length} label="Спростовують" color="red" />
                        <SummaryCell count={unrelated.length} label="Нерелевантні" color="gray" />
                        <SummaryCell count={unknown.length} label="Невизначено" color="gray" />
                    </div>

                    {/* Query used */}
                    {bundle.query_used && (
                        <div className="text-xs text-muted-foreground flex items-center gap-1">
                            <Target className="h-3 w-3" />
                            Пошук: <code className="bg-muted px-1.5 py-0.5 rounded font-mono text-[11px]">
                                {bundle.query_used}
                            </code>
                            <span className="ml-auto">
                                {bundle.total_found} evidence знайдено за {bundle.retrieval_time_seconds.toFixed(1)}с
                            </span>
                        </div>
                    )}

                    {/* Evidence lists */}
                    {refutes.length > 0 && (
                        <EvidenceList
                            items={refutes}
                            title="Спростовують"
                            icon={AlertTriangle}
                            tone="red"
                        />
                    )}

                    {supports.length > 0 && (
                        <EvidenceList
                            items={supports}
                            title="Підтримують"
                            icon={CheckCircle2}
                            tone="green"
                        />
                    )}

                    {unrelated.length > 0 && (
                        <EvidenceList
                            items={unrelated.slice(0, 3)}
                            title="Нерелевантні"
                            icon={HelpCircle}
                            tone="gray"
                            collapsible
                        />
                    )}
                </CardContent>
            )}
        </Card>
    );
}

function SummaryCell({
    count, label, color,
}: {
    count: number;
    label: string;
    color: "green" | "red" | "gray";
}) {
    const bg = {
        green: "bg-green-50 border-green-200 text-green-800 dark:bg-green-950/30 dark:border-green-900 dark:text-green-300",
        red: "bg-red-50 border-red-200 text-red-800 dark:bg-red-950/30 dark:border-red-900 dark:text-red-300",
        gray: "bg-muted border-border text-muted-foreground",
    }[color];
    return (
        <div className={cn("rounded-md border p-2 text-center", bg)}>
            <p className="text-xl font-semibold leading-none">{count}</p>
            <p className="text-[10px] uppercase tracking-wide mt-1">{label}</p>
        </div>
    );
}

// ──────────────────────────────────────────────────────────────────────────

function VerdictCard({
    verdict, isOverall, compact = false,
}: {
    verdict: Verdict;
    isOverall: boolean;
    compact?: boolean;
}) {
    const config = getVerdictStyle(verdict.label);

    return (
        <Card className={cn("border-2", config.border)}>
            <CardContent className={cn("p-6", compact && "p-4")}>
                <div className="flex items-start gap-4">
                    <div className={cn("rounded-full p-3 flex-shrink-0", config.bgIcon)}>
                        <config.Icon className={cn("h-6 w-6", config.iconColor)} />
                    </div>
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap mb-2">
                            <h3 className={cn("text-xl font-bold", config.text, compact && "text-lg")}>
                                {config.label}
                            </h3>
                            {isOverall && (
                                <Badge variant="outline" className="text-[10px]">Overall</Badge>
                            )}
                            <span className="text-sm text-muted-foreground">
                                Впевненість: <strong>{(verdict.confidence * 100).toFixed(0)}%</strong>
                            </span>
                            {verdict.model_used && (
                                <Badge variant="outline" className="text-[10px] ml-auto">
                                    {verdict.model_used.includes("llm") ? (
                                        <><Sparkles className="h-2.5 w-2.5 mr-1" /> LLM</>
                                    ) : "Rule-based"}
                                </Badge>
                            )}
                        </div>

                        <p className="text-sm leading-relaxed">{verdict.reasoning}</p>

                        {/* Breakdown */}
                        {(verdict.breakdown.supports + verdict.breakdown.refutes) > 0 && (
                            <div className="mt-3 pt-3 border-t flex items-center gap-4 text-xs flex-wrap">
                                <span className="flex items-center gap-1">
                                    <CheckCircle2 className="h-3 w-3 text-green-600" />
                                    {verdict.breakdown.supports} підтримують
                                    <span className="text-muted-foreground">
                                        (вага: {verdict.breakdown.weighted_supports.toFixed(2)})
                                    </span>
                                </span>
                                <span className="flex items-center gap-1">
                                    <AlertTriangle className="h-3 w-3 text-red-600" />
                                    {verdict.breakdown.refutes} спростовують
                                    <span className="text-muted-foreground">
                                        (вага: {verdict.breakdown.weighted_refutes.toFixed(2)})
                                    </span>
                                </span>
                                <span className="text-muted-foreground ml-auto">
                                    {verdict.evidence_count} всього evidence
                                </span>
                            </div>
                        )}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}

function getVerdictStyle(label: "FAKE" | "REAL" | "UNCERTAIN") {
    if (label === "FAKE") {
        return {
            Icon: AlertTriangle,
            label: "ДЕЗІНФОРМАЦІЯ",
            border: "border-red-300 dark:border-red-900",
            bgIcon: "bg-red-100 dark:bg-red-950/50",
            iconColor: "text-red-600 dark:text-red-400",
            text: "text-red-700 dark:text-red-300",
        };
    }
    if (label === "REAL") {
        return {
            Icon: CheckCircle2,
            label: "ДОСТОВІРНО",
            border: "border-green-300 dark:border-green-900",
            bgIcon: "bg-green-100 dark:bg-green-950/50",
            iconColor: "text-green-600 dark:text-green-400",
            text: "text-green-700 dark:text-green-300",
        };
    }
    return {
        Icon: HelpCircle,
        label: "НЕВИЗНАЧЕНО",
        border: "border-amber-300 dark:border-amber-900",
        bgIcon: "bg-amber-100 dark:bg-amber-950/50",
        iconColor: "text-amber-600 dark:text-amber-400",
        text: "text-amber-700 dark:text-amber-300",
    };
}

// ──────────────────────────────────────────────────────────────────────────

function EvidenceList({
    items, title, icon: Icon, tone, collapsible = false,
}: {
    items: Evidence[];
    title: string;
    icon: any;
    tone: "green" | "red" | "gray";
    collapsible?: boolean;
}) {
    const [expanded, setExpanded] = useState(!collapsible);

    const iconColor = {
        green: "text-green-600 dark:text-green-400",
        red: "text-red-600 dark:text-red-400",
        gray: "text-muted-foreground",
    }[tone];

    return (
        <div className="space-y-2">
            <button
                onClick={() => collapsible && setExpanded(!expanded)}
                className={cn(
                    "flex items-center gap-2 text-sm font-medium",
                    collapsible && "hover:text-primary cursor-pointer"
                )}
            >
                <Icon className={cn("h-4 w-4", iconColor)} />
                {title} ({items.length})
                {collapsible && (expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />)}
            </button>

            {expanded && (
                <div className="space-y-1.5 pl-2">
                    {items.map((ev, i) => <EvidenceItem key={i} evidence={ev} />)}
                </div>
            )}
        </div>
    );
}

function EvidenceItem({ evidence }: { evidence: Evidence }) {
    const sourceIcon = evidence.source_type === "rss" ? Newspaper : MessageCircle;
    const SourceIcon = sourceIcon;

    const authorityColor =
        (evidence.authority_weight ?? 0.5) >= 0.8 ? "text-blue-700 dark:text-blue-400" :
            (evidence.authority_weight ?? 0.5) >= 0.6 ? "text-muted-foreground" :
                "text-muted-foreground opacity-70";

    const titleOrText = evidence.title || evidence.text;
    const displayText = titleOrText.length > 150
        ? titleOrText.slice(0, 150) + "…"
        : titleOrText;

    return (
        <div className="rounded-md border border-border bg-card p-3 text-xs space-y-1.5 hover:bg-muted/30 transition-colors">
            <div className="flex items-center gap-2 flex-wrap">
                <SourceIcon className={cn("h-3.5 w-3.5 flex-shrink-0", authorityColor)} />
                <span className={cn("font-medium truncate", authorityColor)}>
                    {evidence.source_name}
                </span>
                {evidence.author_verified && (
                    <Badge className="bg-blue-100 text-blue-700 border-blue-200 text-[9px] px-1 py-0">verified</Badge>
                )}
                {evidence.authority_weight !== null && evidence.authority_weight !== undefined && evidence.authority_weight >= 0.8 && (
                    <Badge variant="outline" className="text-[9px] px-1 py-0">authoritative</Badge>
                )}
                <span className="text-muted-foreground text-[10px] ml-auto">
                    confidence: {((evidence.stance_confidence ?? 0) * 100).toFixed(0)}%
                </span>
                {evidence.url && (
                    <a
                        href={evidence.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary hover:underline flex items-center gap-0.5"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <ExternalLink className="h-3 w-3" />
                    </a>
                )}
            </div>
            <p className="text-foreground leading-snug">{displayText}</p>
            {evidence.stance_reasoning && (
                <p className="text-muted-foreground italic leading-snug text-[11px]">
                    → {evidence.stance_reasoning}
                </p>
            )}
        </div>
    );
}