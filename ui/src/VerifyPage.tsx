// ui/src/VerifyPage.tsx
import React, { useState } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Label } from "./components/ui/label";
import { Badge } from "./components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import {
    ShieldCheck, Loader2, Search, AlertTriangle, Info, CheckCircle2,
    Heart, Repeat2, MessageCircle, Users, Clock, Globe, TrendingUp, Link2,
} from "lucide-react";
import { toast } from "sonner";
import type { VerifyNewsResponse, VerificationSignal, NewsItem } from "./types";

// Reuse the results renderer by importing internal components from VerifyModal
// (or duplicate — we'll duplicate a tiny bit for clarity)

export default function VerifyPage() {
    const [mode, setMode] = useState("url");
    const [url, setUrl] = useState("");
    const [title, setTitle] = useState("");
    const [text, setText] = useState("");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<VerifyNewsResponse | null>(null);

    const runVerification = async () => {
        if (mode === "url" && !url.trim()) {
            toast.error("Введіть URL");
            return;
        }
        if (mode === "title" && !title.trim()) {
            toast.error("Введіть заголовок");
            return;
        }
        if (mode === "text" && !text.trim()) {
            toast.error("Введіть текст");
            return;
        }

        setLoading(true);
        setResult(null);
        try {
            const payload: any = {
                social_sources: ["bluesky", "mastodon"],
                limit_per_source: 15,
            };
            if (mode === "url") payload.url = url.trim();
            if (mode === "title") payload.title = title.trim();
            if (mode === "text") payload.text = text.trim();

            const { data } = await api.post<VerifyNewsResponse>("/sources/verify-news", payload);
            setResult(data);
        } catch (err: any) {
            toast.error(err.response?.data?.detail || "Помилка перевірки");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-2xl font-bold tracking-tight flex items-center gap-2">
                    <ShieldCheck className="h-6 w-6 text-primary" />
                    Cross-Platform Verify
                </h2>
                <p className="text-muted-foreground mt-1">
                    Перевірка новини через пошук згадок у Bluesky та Mastodon
                </p>
            </div>

            {/* Input card */}
            <Card>
                <CardHeader>
                    <CardTitle className="text-lg">Вхідні дані</CardTitle>
                    <CardDescription>
                        Вкажіть новину одним із способів нижче
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <Tabs value={mode} onValueChange={setMode}>
                        <TabsList className="grid w-full grid-cols-3 mb-4">
                            <TabsTrigger value="url">URL</TabsTrigger>
                            <TabsTrigger value="title">Заголовок</TabsTrigger>
                            <TabsTrigger value="text">Текст</TabsTrigger>
                        </TabsList>

                        <TabsContent value="url" className="space-y-3">
                            <div className="space-y-2">
                                <Label htmlFor="url-input">URL статті</Label>
                                <Input
                                    id="url-input"
                                    type="url"
                                    placeholder="https://www.reuters.com/world/..."
                                    value={url}
                                    onChange={(e) => setUrl(e.target.value)}
                                />
                                <p className="text-xs text-muted-foreground">
                                    Вставте посилання на статтю з новинного сайту. Система витягне
                                    заголовок і шукатиме згадки у соцмережах.
                                </p>
                            </div>
                        </TabsContent>

                        <TabsContent value="title" className="space-y-3">
                            <div className="space-y-2">
                                <Label htmlFor="title-input">Заголовок новини</Label>
                                <Input
                                    id="title-input"
                                    placeholder="Ukraine announces new defense package from allies"
                                    value={title}
                                    onChange={(e) => setTitle(e.target.value)}
                                />
                                <p className="text-xs text-muted-foreground">
                                    З заголовку витягнуться ключові слова для пошуку.
                                </p>
                            </div>
                        </TabsContent>

                        <TabsContent value="text" className="space-y-3">
                            <div className="space-y-2">
                                <Label htmlFor="text-input">Текст повідомлення</Label>
                                <Textarea
                                    id="text-input"
                                    placeholder="Фрагмент тексту, який потрібно перевірити..."
                                    value={text}
                                    onChange={(e) => setText(e.target.value)}
                                    rows={4}
                                />
                            </div>
                        </TabsContent>
                    </Tabs>

                    <Button
                        className="w-full mt-4"
                        onClick={runVerification}
                        disabled={loading}
                    >
                        {loading ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Перевіряю...
                            </>
                        ) : (
                            <>
                                <Search className="mr-2 h-4 w-4" />
                                Перевірити
                            </>
                        )}
                    </Button>
                </CardContent>
            </Card>

            {/* Result */}
            {result && (
                <Card>
                    <CardHeader>
                        <CardTitle className="text-lg">Результат перевірки</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <VerifyResultsInline result={result} />
                    </CardContent>
                </Card>
            )}
        </div>
    );
}

// ── Inline version of VerifyResults ──────────────────────────────────────

const SIGNAL_STYLES: Record<string, string> = {
    info: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950/40 dark:text-blue-300 dark:border-blue-900",
    warn: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/40 dark:text-amber-300 dark:border-amber-900",
    alert: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950/40 dark:text-red-300 dark:border-red-900",
};

function VerifyResultsInline({ result }: { result: VerifyNewsResponse }) {
    return (
        <div className="space-y-5">
            <div className="flex items-center gap-2 text-sm">
                <Search className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-muted-foreground">Запит:</span>
                <code className="text-xs bg-muted px-2 py-0.5 rounded font-mono">
                    {result.query_used || "—"}
                </code>
            </div>

            {result.signals.length > 0 && (
                <div className="space-y-2">
                    <h3 className="text-sm font-semibold flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-primary" />
                        Сигнали
                    </h3>
                    {result.signals.map((sig, i) => (
                        <SignalCardInline key={i} signal={sig} />
                    ))}
                </div>
            )}

            <div>
                <h3 className="text-sm font-semibold mb-2">Агрегована статистика</h3>
                <StatsGridInline result={result} />
            </div>

            <div>
                <h3 className="text-sm font-semibold mb-2">
                    Обговорення ({result.related_posts.length})
                </h3>
                {result.related_posts.length === 0 ? (
                    <div className="text-center py-8 text-sm text-muted-foreground border border-dashed rounded-md">
                        Жодного посту не знайдено.
                    </div>
                ) : (
                    <div className="space-y-2">
                        {result.related_posts.slice(0, 30).map((post) => (
                            <RelatedPostInline key={post.id} post={post} />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

function SignalCardInline({ signal }: { signal: VerificationSignal }) {
    const Icon = signal.severity === "info" ? CheckCircle2 : (signal.severity === "alert" ? AlertTriangle : Info);
    return (
        <div className={cn("flex items-start gap-3 p-3 rounded-md border", SIGNAL_STYLES[signal.severity])}>
            <Icon className="h-4 w-4 flex-shrink-0 mt-0.5" />
            <div className="flex-1 text-sm">
                <p className="font-medium">{signal.note}</p>
                {signal.value !== undefined && signal.value !== null && (
                    <p className="text-xs opacity-80 mt-0.5">Значення: {signal.value}</p>
                )}
            </div>
        </div>
    );
}

function StatsGridInline({ result }: { result: VerifyNewsResponse }) {
    const stats = result.stats;
    const items: Array<{ icon: any; label: string; value: string | number; hint?: string }> = [];

    items.push({
        icon: TrendingUp, label: "Постів", value: stats.total_related,
        hint: Object.entries(stats.by_source).filter(([, n]) => n > 0).map(([s, n]) => `${s}: ${n}`).join(" · "),
    });
    items.push({ icon: Heart, label: "Лайки", value: stats.total_engagement.likes.toLocaleString() });
    items.push({ icon: Repeat2, label: "Репости", value: stats.total_engagement.reposts.toLocaleString() });
    items.push({ icon: MessageCircle, label: "Коментарі", value: stats.total_engagement.replies.toLocaleString() });

    if (stats.verified_authors_pct != null) {
        items.push({ icon: ShieldCheck, label: "Верифікованих", value: `${stats.verified_authors_pct}%` });
    }
    if (stats.custom_domain_authors_pct != null) {
        items.push({ icon: Globe, label: "Custom-домен", value: `${stats.custom_domain_authors_pct}%` });
    }
    if (stats.avg_account_age_days != null) {
        items.push({ icon: Clock, label: "Вік акаунту", value: `${Math.round(stats.avg_account_age_days / 30)} міс.` });
    }
    if (stats.avg_followers_count != null) {
        items.push({ icon: Users, label: "Фоловери", value: stats.avg_followers_count.toLocaleString() });
    }
    if (stats.domain_mentioned_count > 0 && stats.original_domain) {
        items.push({ icon: Link2, label: "Згадки джерела", value: `${stats.domain_mentioned_count}/${stats.total_related}`, hint: stats.original_domain });
    }

    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {items.map((item, i) => (
                <div key={i} className="rounded-md border border-border bg-card p-3">
                    <div className="flex items-center gap-2 mb-1">
                        <item.icon className="h-3.5 w-3.5 text-muted-foreground" />
                        <p className="text-xs font-medium text-muted-foreground truncate">{item.label}</p>
                    </div>
                    <p className="text-lg font-semibold leading-none tracking-tight">{item.value}</p>
                    {item.hint && (
                        <p className="text-[10px] text-muted-foreground mt-1 truncate">{item.hint}</p>
                    )}
                </div>
            ))}
        </div>
    );
}

function RelatedPostInline({ post }: { post: NewsItem }) {
    const text = post.text.length > 250 ? post.text.slice(0, 250) + "…" : post.text;
    return (
        <div className="rounded-md border border-border p-3 hover:bg-muted/30 transition-colors">
            <div className="flex items-center gap-2 mb-2 flex-wrap">
                <Badge variant="outline" className="text-[10px] uppercase">{post.source}</Badge>
                {post.author_handle && (
                    <span className="text-xs font-medium truncate">{post.author_handle}</span>
                )}
                {post.author_is_verified && (
                    <Badge className="bg-blue-100 text-blue-700 border-blue-200 text-[10px]">
                        <CheckCircle2 className="h-2.5 w-2.5 mr-0.5" /> verified
                    </Badge>
                )}
                {post.author_has_custom_domain && (
                    <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200 text-[10px]">
                        <Globe className="h-2.5 w-2.5 mr-0.5" /> domain
                    </Badge>
                )}
                {post.url && (
                    <a href={post.url} target="_blank" rel="noopener noreferrer" className="ml-auto text-xs text-primary hover:underline">
                        Відкрити →
                    </a>
                )}
            </div>
            <p className="text-sm leading-snug mb-2">{text}</p>
            <div className="flex items-center gap-3 text-[11px] text-muted-foreground flex-wrap">
                {post.likes_count != null && <span className="flex items-center gap-1"><Heart className="h-3 w-3" />{post.likes_count.toLocaleString()}</span>}
                {post.reposts_count != null && <span className="flex items-center gap-1"><Repeat2 className="h-3 w-3" />{post.reposts_count.toLocaleString()}</span>}
                {post.replies_count != null && <span className="flex items-center gap-1"><MessageCircle className="h-3 w-3" />{post.replies_count.toLocaleString()}</span>}
                {post.author_followers_count != null && (
                    <span className="flex items-center gap-1 ml-auto">
                        <Users className="h-3 w-3" />{post.author_followers_count.toLocaleString()} фол.
                    </span>
                )}
                {post.author_account_age_days != null && (
                    <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />{Math.round(post.author_account_age_days / 30)} міс.
                    </span>
                )}
            </div>
        </div>
    );
}