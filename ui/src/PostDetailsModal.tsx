// ui/src/PostDetailsModal.tsx
import React, { useState, useEffect } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "./components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Badge } from "./components/ui/badge";
import { Card, CardContent } from "./components/ui/card";
import {
    Loader2, Heart, Repeat2, MessageCircle, Quote, Users,
    CheckCircle2, Globe, Clock, AlertTriangle, Bot, ShieldAlert,
    TrendingUp, Info
} from "lucide-react";
import { toast } from "sonner";
import type { PostDetailsResponse, UserProfile, Reply, ProfileGroupStats } from "./types";

interface PostDetailsModalProps {
    open: boolean;
    onClose: () => void;
    postId: string;
}

export default function PostDetailsModal({ open, onClose, postId }: PostDetailsModalProps) {
    const [loading, setLoading] = useState(false);
    const [data, setData] = useState<PostDetailsResponse | null>(null);
    const [activeTab, setActiveTab] = useState("replies");

    useEffect(() => {
        if (!open || !postId) return;

        const fetchDetails = async () => {
            setLoading(true);
            setData(null);
            try {
                const { data } = await api.get<PostDetailsResponse>("/sources/post-details", {
                    params: {
                        post_id: postId,
                        max_replies: 50,
                        max_likers: 100,
                        max_reposters: 100,
                    },
                });
                setData(data);
            } catch (err: any) {
                toast.error(err.response?.data?.detail || "Не вдалося отримати деталі поста");
            } finally {
                setLoading(false);
            }
        };

        fetchDetails();
    }, [open, postId]);

    return (
        <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
            <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Users className="h-5 w-5 text-primary" />
                        Деталі поста
                    </DialogTitle>
                    <DialogDescription>
                        Повна інформація про взаємодії — репости, лайки, коментарі та профілі учасників
                    </DialogDescription>
                </DialogHeader>

                {loading && (
                    <div className="flex items-center justify-center py-16">
                        <div className="text-center space-y-3">
                            <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto" />
                            <p className="text-sm text-muted-foreground">
                                Завантажую взаємодії...
                            </p>
                        </div>
                    </div>
                )}

                {data && !loading && (
                    <div className="space-y-4">
                        {/* Fetched limits info */}
                        {data.fetched_limits && <FetchedLimitsInfo limits={data.fetched_limits} />}

                        {/* Tabs for different participant types */}
                        <Tabs value={activeTab} onValueChange={setActiveTab}>
                            <TabsList className="grid w-full grid-cols-4">
                                <TabsTrigger value="replies" className="gap-1.5">
                                    <MessageCircle className="h-3.5 w-3.5" />
                                    Коментарі ({data.replies.length})
                                </TabsTrigger>
                                <TabsTrigger value="reposts" className="gap-1.5">
                                    <Repeat2 className="h-3.5 w-3.5" />
                                    Репости ({data.reposted_by.length})
                                </TabsTrigger>
                                <TabsTrigger value="likes" className="gap-1.5">
                                    <Heart className="h-3.5 w-3.5" />
                                    Лайки ({data.liked_by.length})
                                </TabsTrigger>
                                <TabsTrigger value="quotes" className="gap-1.5" disabled={data.quoted_by.length === 0}>
                                    <Quote className="h-3.5 w-3.5" />
                                    Цитати ({data.quoted_by.length})
                                </TabsTrigger>
                            </TabsList>

                            {/* Replies */}
                            <TabsContent value="replies" className="space-y-3 mt-4">
                                {data.stats?.repliers && (
                                    <StatsPanel stats={data.stats.repliers} label="Профіль авторів коментарів" />
                                )}
                                {data.replies.length === 0 ? (
                                    <EmptyState message="Немає коментарів" />
                                ) : (
                                    <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
                                        {data.replies.map((r) => <ReplyCard key={r.id} reply={r} />)}
                                    </div>
                                )}
                            </TabsContent>

                            {/* Reposters */}
                            <TabsContent value="reposts" className="space-y-3 mt-4">
                                {data.stats?.reposters && (
                                    <StatsPanel stats={data.stats.reposters} label="Профіль репостерів" />
                                )}
                                {data.reposted_by.length === 0 ? (
                                    <EmptyState message="Немає репостів" />
                                ) : (
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-[400px] overflow-y-auto pr-1">
                                        {data.reposted_by.map((u) => <UserProfileCard key={u.id} user={u} />)}
                                    </div>
                                )}
                            </TabsContent>

                            {/* Likers */}
                            <TabsContent value="likes" className="space-y-3 mt-4">
                                {data.stats?.likers && (
                                    <StatsPanel stats={data.stats.likers} label="Профіль лайкерів" />
                                )}
                                {data.liked_by.length === 0 ? (
                                    <EmptyState message="Немає лайків або обмежений доступ" />
                                ) : (
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-[400px] overflow-y-auto pr-1">
                                        {data.liked_by.map((u) => <UserProfileCard key={u.id} user={u} />)}
                                    </div>
                                )}
                            </TabsContent>

                            {/* Quoters (Bluesky only) */}
                            <TabsContent value="quotes" className="space-y-3 mt-4">
                                {data.stats?.quoters && (
                                    <StatsPanel stats={data.stats.quoters} label="Профіль цитаторів" />
                                )}
                                {data.quoted_by.length === 0 ? (
                                    <EmptyState message="Немає цитат (або Mastodon не підтримує)" />
                                ) : (
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-[400px] overflow-y-auto pr-1">
                                        {data.quoted_by.map((u) => <UserProfileCard key={u.id} user={u} />)}
                                    </div>
                                )}
                            </TabsContent>
                        </Tabs>
                    </div>
                )}
            </DialogContent>
        </Dialog>
    );
}

// ── Subcomponents ─────────────────────────────────────────────────────────

function EmptyState({ message }: { message: string }) {
    return (
        <div className="text-center py-8 text-sm text-muted-foreground border border-dashed rounded-md">
            {message}
        </div>
    );
}

function FetchedLimitsInfo({ limits }: { limits: PostDetailsResponse["fetched_limits"] }) {
    if (!limits) return null;
    const items = [];
    if (limits.likes_total != null && limits.likes_fetched != null && limits.likes_total > limits.likes_fetched) {
        items.push(`отримано ${limits.likes_fetched} з ${limits.likes_total} лайків`);
    }
    if (limits.reposts_total != null && limits.reposts_fetched != null && limits.reposts_total > limits.reposts_fetched) {
        items.push(`${limits.reposts_fetched} з ${limits.reposts_total} репостів`);
    }
    if (limits.replies_total != null && limits.replies_fetched != null && limits.replies_total > limits.replies_fetched) {
        items.push(`${limits.replies_fetched} з ${limits.replies_total} коментарів`);
    }
    if (items.length === 0) return null;
    return (
        <div className="flex items-start gap-2 p-2.5 rounded-md bg-muted/50 text-xs text-muted-foreground">
            <Info className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
            <span>Показано sample: {items.join(", ")}. API обмежує кількість на запит.</span>
        </div>
    );
}

function StatsPanel({ stats, label }: { stats: ProfileGroupStats; label: string }) {
    if (!stats || stats.total === 0) return null;

    // Detect warning signals
    const warnings: string[] = [];
    if (stats.young_accounts_30d_pct != null && stats.young_accounts_30d_pct >= 30) {
        warnings.push(`${stats.young_accounts_30d_pct}% акаунтів молодше 30 днів`);
    }
    if (stats.low_followers_pct != null && stats.low_followers_pct >= 50) {
        warnings.push(`${stats.low_followers_pct}% мають <50 фоловерів`);
    }
    if (stats.suspicious_ratio_pct != null && stats.suspicious_ratio_pct >= 30) {
        warnings.push(`${stats.suspicious_ratio_pct}% мають підозрілий followers/following`);
    }

    return (
        <Card className={cn(warnings.length > 0 && "border-amber-300 dark:border-amber-900")}>
            <CardContent className="p-4 space-y-3">
                <div className="flex items-center justify-between">
                    <h4 className="text-sm font-semibold">{label}</h4>
                    <Badge variant="outline" className="text-xs">
                        {stats.total} користувачів
                    </Badge>
                </div>

                {warnings.length > 0 && (
                    <div className="flex items-start gap-2 p-2 rounded-md bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900">
                        <AlertTriangle className="h-3.5 w-3.5 text-amber-700 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                        <ul className="text-xs text-amber-800 dark:text-amber-300 space-y-0.5">
                            {warnings.map((w, i) => <li key={i}>{w}</li>)}
                        </ul>
                    </div>
                )}

                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                    {stats.verified_pct != null && (
                        <StatCell icon={CheckCircle2} label="Верифіковані" value={`${stats.verified_pct}%`} />
                    )}
                    {stats.custom_domain_pct != null && stats.custom_domain_pct > 0 && (
                        <StatCell icon={Globe} label="Custom domain" value={`${stats.custom_domain_pct}%`} />
                    )}
                    {stats.avg_account_age_days != null && (
                        <StatCell
                            icon={Clock}
                            label="Середній вік"
                            value={`${Math.round(stats.avg_account_age_days / 30)} міс.`}
                            hint={`${stats.avg_account_age_days} днів`}
                        />
                    )}
                    {stats.avg_followers_count != null && (
                        <StatCell
                            icon={Users}
                            label="Фоловери (avg)"
                            value={stats.avg_followers_count.toLocaleString()}
                            hint={stats.median_followers_count != null ? `median ${stats.median_followers_count}` : undefined}
                        />
                    )}
                    {stats.young_accounts_30d_pct != null && stats.young_accounts_30d_pct > 0 && (
                        <StatCell
                            icon={AlertTriangle}
                            label="Нові (<30д)"
                            value={`${stats.young_accounts_30d_pct}%`}
                            warn={stats.young_accounts_30d_pct >= 20}
                        />
                    )}
                    {stats.low_followers_pct != null && stats.low_followers_pct > 0 && (
                        <StatCell
                            icon={TrendingUp}
                            label="<50 фоловерів"
                            value={`${stats.low_followers_pct}%`}
                            warn={stats.low_followers_pct >= 40}
                        />
                    )}
                    {stats.bots_count != null && stats.bots_count > 0 && (
                        <StatCell icon={Bot} label="Боти" value={stats.bots_count.toString()} warn />
                    )}
                    {stats.suspicious_ratio_pct != null && stats.suspicious_ratio_pct > 0 && (
                        <StatCell
                            icon={ShieldAlert}
                            label="Підозрілі"
                            value={`${stats.suspicious_ratio_pct}%`}
                            warn={stats.suspicious_ratio_pct >= 20}
                            hint="followers/following >10"
                        />
                    )}
                </div>
            </CardContent>
        </Card>
    );
}

function StatCell({
    icon: Icon,
    label,
    value,
    hint,
    warn,
}: {
    icon: any;
    label: string;
    value: string;
    hint?: string;
    warn?: boolean;
}) {
    return (
        <div className={cn(
            "rounded-md border p-2",
            warn
                ? "border-amber-300 bg-amber-50/50 dark:bg-amber-950/20 dark:border-amber-900"
                : "border-border bg-card"
        )}>
            <div className="flex items-center gap-1.5 mb-0.5">
                <Icon className={cn(
                    "h-3 w-3",
                    warn ? "text-amber-700 dark:text-amber-400" : "text-muted-foreground"
                )} />
                <p className="text-[10px] font-medium text-muted-foreground truncate">{label}</p>
            </div>
            <p className={cn(
                "text-sm font-semibold leading-tight",
                warn && "text-amber-800 dark:text-amber-300"
            )}>
                {value}
            </p>
            {hint && <p className="text-[9px] text-muted-foreground mt-0.5 truncate">{hint}</p>}
        </div>
    );
}

function UserProfileCard({ user }: { user: UserProfile }) {
    return (
        <div className="rounded-md border border-border p-2.5 hover:bg-muted/30 transition-colors text-xs space-y-1.5">
            <div className="flex items-center gap-1.5 flex-wrap">
                <span className="font-medium truncate">
                    {user.display_name || user.handle || "Unknown"}
                </span>
                {user.is_verified && (
                    <Badge className="bg-blue-100 text-blue-700 border-blue-200 text-[9px] px-1 py-0 dark:bg-blue-950/40 dark:text-blue-300">
                        <CheckCircle2 className="h-2 w-2 mr-0.5" />
                        verified
                    </Badge>
                )}
                {user.has_custom_domain && (
                    <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200 text-[9px] px-1 py-0 dark:bg-emerald-950/40 dark:text-emerald-300">
                        <Globe className="h-2 w-2 mr-0.5" />
                        domain
                    </Badge>
                )}
                {user.is_bot && (
                    <Badge className="bg-amber-50 text-amber-700 border-amber-200 text-[9px] px-1 py-0 dark:bg-amber-950/40 dark:text-amber-300">
                        <Bot className="h-2 w-2 mr-0.5" />
                        bot
                    </Badge>
                )}
            </div>
            {user.handle && user.handle !== user.display_name && (
                <p className="text-muted-foreground truncate text-[11px]">{user.handle}</p>
            )}
            <div className="flex items-center gap-2.5 text-[10px] text-muted-foreground flex-wrap">
                {user.followers_count != null && (
                    <span className="flex items-center gap-0.5">
                        <Users className="h-2.5 w-2.5" />
                        {user.followers_count.toLocaleString()}
                    </span>
                )}
                {user.account_age_days != null && (
                    <span className="flex items-center gap-0.5">
                        <Clock className="h-2.5 w-2.5" />
                        {user.account_age_days < 30
                            ? `${user.account_age_days}д`
                            : `${Math.round(user.account_age_days / 30)}м`}
                    </span>
                )}
                {user.posts_count != null && (
                    <span>{user.posts_count.toLocaleString()} постів</span>
                )}
            </div>
        </div>
    );
}

function ReplyCard({ reply }: { reply: Reply }) {
    const text = reply.text.length > 250 ? reply.text.slice(0, 250) + "…" : reply.text;
    return (
        <Card className="hover:bg-muted/30 transition-colors">
            <CardContent className="p-3 space-y-2">
                {/* Author row */}
                {reply.author && (
                    <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="text-xs font-medium truncate">
                            {reply.author.display_name || reply.author.handle || "Unknown"}
                        </span>
                        {reply.author.is_verified && (
                            <Badge className="bg-blue-100 text-blue-700 border-blue-200 text-[9px] px-1 py-0">
                                <CheckCircle2 className="h-2 w-2 mr-0.5" />
                                verified
                            </Badge>
                        )}
                        {reply.author.has_custom_domain && (
                            <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200 text-[9px] px-1 py-0">
                                <Globe className="h-2 w-2 mr-0.5" />
                                domain
                            </Badge>
                        )}
                        {reply.author.is_bot && (
                            <Badge className="bg-amber-50 text-amber-700 border-amber-200 text-[9px] px-1 py-0">
                                <Bot className="h-2 w-2 mr-0.5" />
                                bot
                            </Badge>
                        )}
                        {reply.author.account_age_days != null && reply.author.account_age_days < 30 && (
                            <Badge className="bg-red-50 text-red-700 border-red-200 text-[9px] px-1 py-0">
                                <AlertTriangle className="h-2 w-2 mr-0.5" />
                                новий ({reply.author.account_age_days}д)
                            </Badge>
                        )}
                        {reply.url && (
                            <a href={reply.url} target="_blank" rel="noopener noreferrer"
                                className="ml-auto text-[11px] text-primary hover:underline">
                                Відкрити →
                            </a>
                        )}
                    </div>
                )}

                {/* Text */}
                <p className="text-sm leading-snug">{text}</p>

                {/* Engagement + author metadata */}
                <div className="flex items-center gap-3 text-[10px] text-muted-foreground flex-wrap">
                    {reply.likes_count != null && (
                        <span className="flex items-center gap-0.5">
                            <Heart className="h-2.5 w-2.5" />
                            {reply.likes_count.toLocaleString()}
                        </span>
                    )}
                    {reply.reposts_count != null && (
                        <span className="flex items-center gap-0.5">
                            <Repeat2 className="h-2.5 w-2.5" />
                            {reply.reposts_count.toLocaleString()}
                        </span>
                    )}
                    {reply.author?.followers_count != null && (
                        <span className="flex items-center gap-0.5 ml-auto">
                            <Users className="h-2.5 w-2.5" />
                            {reply.author.followers_count.toLocaleString()} фол.
                        </span>
                    )}
                </div>
            </CardContent>
        </Card>
    );
}