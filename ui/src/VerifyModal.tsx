// ui/src/VerifyModal.tsx
import React, { useState } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "./components/ui/dialog";
import { Badge } from "./components/ui/badge";
import { Card, CardContent } from "./components/ui/card";
import {
    Loader2, ShieldCheck, AlertTriangle, Info, CheckCircle2,
    Heart, Repeat2, MessageCircle, Users, Clock, Globe,
    TrendingUp, Link2, Search
} from "lucide-react";
import { toast } from "sonner";
import type { VerifyNewsResponse, VerificationSignal, NewsItem } from "./types";

interface VerifyModalProps {
    open: boolean;
    onClose: () => void;
    /** Одне з трьох: url, title або text — або їх комбінація */
    url?: string;
    title?: string;
    text?: string;
    /** Опційно: передати попередньо завантажений результат */
    initialResult?: VerifyNewsResponse | null;
}

const SIGNAL_ICONS: Record<string, any> = {
    info: Info,
    warn: AlertTriangle,
    alert: AlertTriangle,
};

const SIGNAL_STYLES: Record<string, string> = {
    info: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/40 dark:text-blue-300 dark:border-blue-900",
    warn: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-900",
    alert: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950/40 dark:text-red-300 dark:border-red-900",
};

export default function VerifyModal({ open, onClose, url, title, text, initialResult }: VerifyModalProps) {
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<VerifyNewsResponse | null>(initialResult || null);

    const runVerification = async () => {
        if (!url && !title && !text) {
            toast.error("Потрібно вказати URL, заголовок або текст");
            return;
        }
        setLoading(true);
        setResult(null);
        try {
            const { data } = await api.post<VerifyNewsResponse>("/sources/verify-news", {
                url,
                title,
                text,
                social_sources: ["bluesky", "mastodon"],
                limit_per_source: 10,
            });
            setResult(data);
        } catch (err: any) {
            toast.error(err.response?.data?.detail || "Помилка перевірки");
        } finally {
            setLoading(false);
        }
    };

    // Reset and re-run when modal opens or input changes
    React.useEffect(() => {
        if (open && !initialResult) {
            setResult(null);
            runVerification();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [open, url, title, text]);

    return (
        <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
            <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <ShieldCheck className="h-5 w-5 text-primary" />
                        Cross-Platform Verification
                    </DialogTitle>
                    <DialogDescription>
                        Пошук згадок у Bluesky та Mastodon + аналіз сигналів достовірності
                    </DialogDescription>
                </DialogHeader>

                {loading && (
                    <div className="flex items-center justify-center py-16">
                        <div className="text-center space-y-3">
                            <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto" />
                            <p className="text-sm text-muted-foreground">
                                Шукаю у соцмережах...
                            </p>
                        </div>
                    </div>
                )}

                {result && !loading && <VerifyResults result={result} />}

                {!loading && !result && (
                    <div className="text-center py-8">
                        <Button onClick={runVerification}>
                            <Search className="mr-2 h-4 w-4" />
                            Розпочати перевірку
                        </Button>
                    </div>
                )}
            </DialogContent>
        </Dialog>
    );
}

// ── Results component ─────────────────────────────────────────────────────

function VerifyResults({ result }: { result: VerifyNewsResponse }) {
    return (
        <div className="space-y-5">
            {/* Query used */}
            <div className="flex items-center gap-2 text-sm">
                <Search className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-muted-foreground">Пошуковий запит:</span>
                <code className="text-xs bg-muted px-2 py-0.5 rounded font-mono">
                    {result.query_used || "—"}
                </code>
            </div>

            {/* Signals block — найважливіше */}
            {result.signals.length > 0 && (
                <div className="space-y-2">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-primary" />
                        Сигнали
                    </h3>
                    {result.signals.map((sig, i) => (
                        <SignalCard key={i} signal={sig} />
                    ))}
                </div>
            )}

            {/* Stats grid */}
            <div>
                <h3 className="text-sm font-semibold mb-2">Агрегована статистика</h3>
                <StatsGrid result={result} />
            </div>

            {/* Related posts */}
            <div>
                <h3 className="text-sm font-semibold mb-2">
                    Обговорення у соцмережах ({result.related_posts.length})
                </h3>
                {result.related_posts.length === 0 ? (
                    <div className="text-center py-8 text-sm text-muted-foreground border border-dashed rounded-md">
                        Жодного посту не знайдено. Можливо, ця новина ще не обговорюється.
                    </div>
                ) : (
                    <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
                        {result.related_posts.slice(0, 20).map((post) => (
                            <RelatedPostCard key={post.id} post={post} />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

// ── Signal card ───────────────────────────────────────────────────────────

function SignalCard({ signal }: { signal: VerificationSignal }) {
    const Icon = SIGNAL_ICONS[signal.severity] || Info;
    if (signal.severity === "info") {
        // use CheckCircle for positive signals
        const PositiveIcon = CheckCircle2;
        return (
            <div className={cn(
                "flex items-start gap-3 p-3 rounded-md border",
                SIGNAL_STYLES[signal.severity]
            )}>
                <PositiveIcon className="h-4 w-4 flex-shrink-0 mt-0.5" />
                <div className="flex-1 text-sm">
                    <p>{signal.note}</p>
                </div>
            </div>
        );
    }
    return (
        <div className={cn(
            "flex items-start gap-3 p-3 rounded-md border",
            SIGNAL_STYLES[signal.severity]
        )}>
            <Icon className="h-4 w-4 flex-shrink-0 mt-0.5" />
            <div className="flex-1 text-sm">
                <p className="font-medium">{signal.note}</p>
                {signal.value !== undefined && signal.value !== null && (
                    <p className="text-xs opacity-80 mt-0.5">
                        Значення: {signal.value}
                    </p>
                )}
            </div>
        </div>
    );
}

// ── Stats grid ────────────────────────────────────────────────────────────

function StatsGrid({ result }: { result: VerifyNewsResponse }) {
    const stats = result.stats;
    const items: Array<{ icon: any; label: string; value: string | number; hint?: string }> = [];

    items.push({
        icon: TrendingUp,
        label: "Знайдено постів",
        value: stats.total_related,
        hint: Object.entries(stats.by_source)
            .filter(([, n]) => n > 0)
            .map(([s, n]) => `${s}: ${n}`)
            .join(" · "),
    });

    if (stats.total_engagement) {
        items.push({
            icon: Heart,
            label: "Лайки",
            value: stats.total_engagement.likes.toLocaleString(),
        });
        items.push({
            icon: Repeat2,
            label: "Репости",
            value: stats.total_engagement.reposts.toLocaleString(),
        });
        items.push({
            icon: MessageCircle,
            label: "Коментарі",
            value: stats.total_engagement.replies.toLocaleString(),
        });
    }

    if (stats.verified_authors_pct !== null && stats.verified_authors_pct !== undefined) {
        items.push({
            icon: ShieldCheck,
            label: "Верифіковані автори",
            value: `${stats.verified_authors_pct}%`,
        });
    }

    if (stats.custom_domain_authors_pct !== null && stats.custom_domain_authors_pct !== undefined) {
        items.push({
            icon: Globe,
            label: "Custom-домен автори",
            value: `${stats.custom_domain_authors_pct}%`,
            hint: "Bluesky handle як nytimes.com",
        });
    }

    if (stats.avg_account_age_days !== null && stats.avg_account_age_days !== undefined) {
        items.push({
            icon: Clock,
            label: "Середній вік акаунту",
            value: `${Math.round(stats.avg_account_age_days / 30)} міс.`,
            hint: `${stats.avg_account_age_days} днів`,
        });
    }

    if (stats.avg_followers_count !== null && stats.avg_followers_count !== undefined) {
        items.push({
            icon: Users,
            label: "Середня к-сть фоловерів",
            value: stats.avg_followers_count.toLocaleString(),
        });
    }

    if (stats.domain_mentioned_count > 0 && stats.original_domain) {
        items.push({
            icon: Link2,
            label: "Згадки джерела",
            value: `${stats.domain_mentioned_count}/${stats.total_related}`,
            hint: stats.original_domain,
        });
    }

    return (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
            {items.map((item, i) => (
                <div key={i} className="rounded-md border border-border bg-card p-3">
                    <div className="flex items-center gap-2 mb-1">
                        <item.icon className="h-3.5 w-3.5 text-muted-foreground" />
                        <p className="text-xs font-medium text-muted-foreground truncate">
                            {item.label}
                        </p>
                    </div>
                    <p className="text-lg font-semibold leading-none tracking-tight">
                        {item.value}
                    </p>
                    {item.hint && (
                        <p className="text-[10px] text-muted-foreground mt-1 truncate">
                            {item.hint}
                        </p>
                    )}
                </div>
            ))}
        </div>
    );
}

// ── Related post card ─────────────────────────────────────────────────────

function RelatedPostCard({ post }: { post: NewsItem }) {
    const text = post.text.length > 200 ? post.text.slice(0, 200) + "…" : post.text;

    return (
        <Card className="hover:bg-muted/30 transition-colors">
            <CardContent className="p-3">
                <div className="flex items-center gap-2 mb-2 flex-wrap">
                    <Badge variant="outline" className="text-[10px] uppercase">
                        {post.source}
                    </Badge>
                    {post.author_handle && (
                        <span className="text-xs font-medium truncate">
                            {post.author_handle}
                        </span>
                    )}
                    {post.author_is_verified && (
                        <Badge className="bg-blue-100 text-blue-700 border-blue-200 text-[10px] dark:bg-blue-950/40 dark:text-blue-300">
                            <CheckCircle2 className="h-2.5 w-2.5 mr-0.5" />
                            verified
                        </Badge>
                    )}
                    {post.author_has_custom_domain && (
                        <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200 text-[10px] dark:bg-emerald-950/40 dark:text-emerald-300">
                            <Globe className="h-2.5 w-2.5 mr-0.5" />
                            domain
                        </Badge>
                    )}
                    {post.url && (
                        <a
                            href={post.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="ml-auto text-xs text-primary hover:underline"
                        >
                            Відкрити →
                        </a>
                    )}
                </div>

                <p className="text-sm leading-snug mb-2">{text}</p>

                <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
                    {post.likes_count !== null && post.likes_count !== undefined && (
                        <span className="flex items-center gap-1">
                            <Heart className="h-3 w-3" />
                            {post.likes_count.toLocaleString()}
                        </span>
                    )}
                    {post.reposts_count !== null && post.reposts_count !== undefined && (
                        <span className="flex items-center gap-1">
                            <Repeat2 className="h-3 w-3" />
                            {post.reposts_count.toLocaleString()}
                        </span>
                    )}
                    {post.replies_count !== null && post.replies_count !== undefined && (
                        <span className="flex items-center gap-1">
                            <MessageCircle className="h-3 w-3" />
                            {post.replies_count.toLocaleString()}
                        </span>
                    )}
                    {post.author_followers_count !== null && post.author_followers_count !== undefined && (
                        <span className="flex items-center gap-1 ml-auto">
                            <Users className="h-3 w-3" />
                            {post.author_followers_count.toLocaleString()}
                        </span>
                    )}
                    {post.author_account_age_days !== null && post.author_account_age_days !== undefined && (
                        <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {Math.round(post.author_account_age_days / 30)} міс.
                        </span>
                    )}
                </div>
            </CardContent>
        </Card>
    );
}