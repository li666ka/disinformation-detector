
import React, { useEffect, useState } from "react";
import api from "./api";
import { Button } from "./components/ui/button";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
} from "./components/ui/dialog";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Legend,
    PieChart,
    Pie,
    Cell,
} from "recharts";
import { Loader2, RefreshCw, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { cn } from "./lib/utils";

interface AnalyticsData {
    overview: {
        total_articles: number;
        total_tweets: number;
        total_users: number;
        label_distribution: Record<string, number>;
        source_distribution: Record<string, number>;
        source_label_breakdown: Record<string, { total: number; fake: number; real: number }>;

        coverage_pct?: number;
        coverage_gap_fake_real?: number;
        synthetic_count?: number;
    };
    text_stats: Record<
        string,
        {
            fake: { mean: number; std: number; median: number };
            real: { mean: number; std: number; median: number };
            histogram_fake: { edges: number[]; counts: number[] };
            histogram_real: { edges: number[]; counts: number[] };
        }
    >;
    emotional_features: Record<
        string,
        {
            fake: { mean: number; std: number; median: number };
            real: { mean: number; std: number; median: number };
            histogram_fake: { edges: number[]; counts: number[] };
            histogram_real: { edges: number[]; counts: number[] };
        }
    >;
    top_words: {
        fake: Array<{ word: string; count: number }>;
        real: Array<{ word: string; count: number }>;
        most_fake_indicative: Array<{ word: string; fake_freq: number; real_freq: number; ratio_fake_over_real: number }>;
        most_real_indicative: Array<{ word: string; fake_freq: number; real_freq: number; ratio_fake_over_real: number }>;
    };
    user_analysis: Record<string, any>;
    engagement: Record<string, any>;
    elapsed_seconds: number;
}

interface AnalyticsModalProps {
    open: boolean;
    onClose: () => void;
    datasetId: number;
    datasetName: string;
}

const FAKE_COLOR = "#ef4444";
const REAL_COLOR = "#22c55e";


function buildHistogramData(
    hist: { edges: number[]; counts: number[] } | undefined,
    otherHist: { edges: number[]; counts: number[] } | undefined,
): Array<{ bin: string; fake: number; real: number }> {
    if (!hist || !otherHist || !hist.edges?.length) return [];
    const bins = hist.counts.length;
    const edges = hist.edges;
    const data = [];
    for (let i = 0; i < bins; i++) {
        const mid = (edges[i] + edges[i + 1]) / 2;
        data.push({
            bin: mid.toFixed(3),
            fake: hist.counts[i] || 0,
            real: otherHist.counts[i] || 0,
        });
    }
    return data;
}

export default function AnalyticsModal({ open, onClose, datasetId, datasetName }: AnalyticsModalProps) {
    const [data, setData] = useState<AnalyticsData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const fetchAnalytics = async (refresh = false) => {
        setLoading(true);
        setError(null);
        try {
            const { data } = await api.get<AnalyticsData>(
                `/datasets/${datasetId}/analytics`,
                { params: { refresh } },
            );
            setData(data);
        } catch (err: any) {
            const msg = err.response?.data?.detail || err.message || "Помилка завантаження";
            setError(msg);
            toast.error(msg);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (open && datasetId) fetchAnalytics(false);
    }, [open, datasetId]);

    return (
        <Dialog open={open} onOpenChange={onClose}>
            <DialogContent className="max-w-5xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle>Аналітика датасету: {datasetName}</DialogTitle>
                    <DialogDescription>
                        Розподіли характеристик FAKE vs REAL для exploratory data analysis
                    </DialogDescription>
                </DialogHeader>

                {loading && (
                    <div className="flex items-center justify-center py-12">
                        <Loader2 className="h-6 w-6 animate-spin mr-2" />
                        <span>Обчислення статистик... (може зайняти до хвилини)</span>
                    </div>
                )}

                {error && (
                    <div className="py-8 text-center">
                        <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-2" />
                        <p className="text-sm text-red-600 mb-4">{error}</p>
                        <Button variant="outline" size="sm" onClick={() => fetchAnalytics(true)}>
                            <RefreshCw className="h-4 w-4 mr-2" />
                            Повторити
                        </Button>
                    </div>
                )}

                {data && !loading && (
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <p className="text-xs text-muted-foreground">
                                Обчислено за {data.elapsed_seconds?.toFixed(1)}с
                            </p>
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={() => fetchAnalytics(true)}
                                disabled={loading}
                            >
                                <RefreshCw className="h-4 w-4 mr-2" />
                                Перерахувати
                            </Button>
                        </div>

                        <Tabs defaultValue="overview" className="w-full">
                            <TabsList className="grid grid-cols-5 w-full">
                                <TabsTrigger value="overview">Огляд</TabsTrigger>
                                <TabsTrigger value="emotional">Емоції</TabsTrigger>
                                <TabsTrigger value="text">Текст</TabsTrigger>
                                <TabsTrigger value="words">Слова</TabsTrigger>
                                <TabsTrigger value="users">Соц. аналіз</TabsTrigger>
                            </TabsList>

                            {}
                            <TabsContent value="overview" className="space-y-4">
                                <div className="grid grid-cols-3 gap-3">
                                    <Card>
                                        <CardContent className="pt-4 text-center">
                                            <p className="text-xs text-muted-foreground">Articles</p>
                                            <p className="text-2xl font-bold">{data.overview.total_articles.toLocaleString()}</p>
                                        </CardContent>
                                    </Card>
                                    <Card>
                                        <CardContent className="pt-4 text-center">
                                            <p className="text-xs text-muted-foreground">Tweets</p>
                                            <p className="text-2xl font-bold">{data.overview.total_tweets.toLocaleString()}</p>
                                        </CardContent>
                                    </Card>
                                    <Card>
                                        <CardContent className="pt-4 text-center">
                                            <p className="text-xs text-muted-foreground">Users</p>
                                            <p className="text-2xl font-bold">{data.overview.total_users.toLocaleString()}</p>
                                        </CardContent>
                                    </Card>
                                </div>

                                {}
                                {data.overview.coverage_gap_fake_real != null && (
                                    <Card className={data.overview.coverage_gap_fake_real > 15
                                        ? "border-amber-500"
                                        : "border-green-500"}>
                                        <CardContent className="pt-4">
                                            <div className="flex items-start gap-3">
                                                <AlertCircle className={cn(
                                                    "h-5 w-5 mt-0.5 shrink-0",
                                                    data.overview.coverage_gap_fake_real > 15 ? "text-amber-500" : "text-green-500"
                                                )} />
                                                <div className="text-sm space-y-1">
                                                    <p className="font-medium">
                                                        User profile coverage: {data.overview.coverage_pct?.toFixed(1)}%
                                                    </p>
                                                    <p className="text-xs text-muted-foreground">
                                                        Gap FAKE vs REAL: {data.overview.coverage_gap_fake_real.toFixed(1)}%
                                                        {data.overview.coverage_gap_fake_real > 15
                                                            ? " — підозра на data leakage у social features"
                                                            : " — нормальний рівень"}
                                                    </p>
                                                    {data.overview.synthetic_count != null && data.overview.synthetic_count > 0 && (
                                                        <p className="text-xs text-muted-foreground">
                                                            Synthetic articles: {data.overview.synthetic_count.toLocaleString()}
                                                            (з реземплованим engagement)
                                                        </p>
                                                    )}
                                                </div>
                                            </div>
                                        </CardContent>
                                    </Card>
                                )}

                                <Card>
                                    <CardHeader>
                                        <CardTitle className="text-base">Розподіл FAKE/REAL</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <ResponsiveContainer width="100%" height={200}>
                                            <PieChart>
                                                <Pie
                                                    data={Object.entries(data.overview.label_distribution).map(([k, v]) => ({ name: k, value: v }))}
                                                    dataKey="value"
                                                    nameKey="name"
                                                    cx="50%" cy="50%"
                                                    outerRadius={80}
                                                    label
                                                >
                                                    {Object.entries(data.overview.label_distribution).map(([k], i) => (
                                                        <Cell key={i} fill={k === "FAKE" ? FAKE_COLOR : REAL_COLOR} />
                                                    ))}
                                                </Pie>
                                                <Tooltip />
                                                <Legend />
                                            </PieChart>
                                        </ResponsiveContainer>
                                    </CardContent>
                                </Card>

                                <Card>
                                    <CardHeader>
                                        <CardTitle className="text-base">Розподіл по джерелах</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <ResponsiveContainer width="100%" height={240}>
                                            <BarChart data={Object.entries(data.overview.source_label_breakdown || {}).map(([k, v]) => ({
                                                source: k, fake: v.fake, real: v.real,
                                            }))}>
                                                <XAxis dataKey="source" />
                                                <YAxis />
                                                <Tooltip />
                                                <Legend />
                                                <Bar dataKey="fake" fill={FAKE_COLOR} name="FAKE" />
                                                <Bar dataKey="real" fill={REAL_COLOR} name="REAL" />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </CardContent>
                                </Card>
                            </TabsContent>

                            {}
                            <TabsContent value="emotional" className="space-y-3">
                                {Object.keys(data.emotional_features).length === 0 || "error" in data.emotional_features ? (
                                    <p className="text-sm text-muted-foreground text-center py-6">
                                        {(data.emotional_features as any).error || "Emotional features недоступні"}
                                    </p>
                                ) : (
                                    <>
                                        {}
                                        <Card>
                                            <CardHeader>
                                                <CardTitle className="text-base">Середні значення</CardTitle>
                                            </CardHeader>
                                            <CardContent>
                                                <table className="w-full text-sm">
                                                    <thead>
                                                        <tr className="border-b">
                                                            <th className="text-left py-1">Feature</th>
                                                            <th className="text-right py-1" style={{ color: FAKE_COLOR }}>FAKE mean</th>
                                                            <th className="text-right py-1" style={{ color: REAL_COLOR }}>REAL mean</th>
                                                            <th className="text-right py-1">Diff</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        {Object.entries(data.emotional_features).map(([feat, s]: any) => {
                                                            const fake = s.fake?.mean || 0;
                                                            const real = s.real?.mean || 0;
                                                            const diff = fake - real;
                                                            return (
                                                                <tr key={feat} className="border-b">
                                                                    <td className="py-1">{feat}</td>
                                                                    <td className="text-right py-1">{fake.toFixed(4)}</td>
                                                                    <td className="text-right py-1">{real.toFixed(4)}</td>
                                                                    <td
                                                                        className="text-right py-1 font-medium"
                                                                        style={{ color: diff > 0 ? FAKE_COLOR : REAL_COLOR }}
                                                                    >
                                                                        {diff > 0 ? "+" : ""}{diff.toFixed(4)}
                                                                    </td>
                                                                </tr>
                                                            );
                                                        })}
                                                    </tbody>
                                                </table>
                                            </CardContent>
                                        </Card>

                                        {}
                                        {["anger_score", "fear_score", "joy_score", "sentiment_score", "emotion_intensity"]
                                            .filter((f) => data.emotional_features[f])
                                            .map((feat) => {
                                                const d = data.emotional_features[feat] as any;
                                                const hist = buildHistogramData(d.histogram_fake, d.histogram_real);
                                                return (
                                                    <Card key={feat}>
                                                        <CardHeader>
                                                            <CardTitle className="text-sm">{feat}</CardTitle>
                                                        </CardHeader>
                                                        <CardContent>
                                                            <ResponsiveContainer width="100%" height={200}>
                                                                <BarChart data={hist}>
                                                                    <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                                                                    <YAxis tick={{ fontSize: 10 }} />
                                                                    <Tooltip />
                                                                    <Legend />
                                                                    <Bar dataKey="fake" fill={FAKE_COLOR} fillOpacity={0.6} name="FAKE" />
                                                                    <Bar dataKey="real" fill={REAL_COLOR} fillOpacity={0.6} name="REAL" />
                                                                </BarChart>
                                                            </ResponsiveContainer>
                                                        </CardContent>
                                                    </Card>
                                                );
                                            })}
                                    </>
                                )}
                            </TabsContent>

                            {}
                            <TabsContent value="text" className="space-y-3">
                                <Card>
                                    <CardHeader>
                                        <CardTitle className="text-base">Текстові характеристики</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <table className="w-full text-sm">
                                            <thead>
                                                <tr className="border-b">
                                                    <th className="text-left py-1">Stat</th>
                                                    <th className="text-right py-1" style={{ color: FAKE_COLOR }}>FAKE</th>
                                                    <th className="text-right py-1" style={{ color: REAL_COLOR }}>REAL</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {Object.entries(data.text_stats).map(([feat, s]: any) => (
                                                    <tr key={feat} className="border-b">
                                                        <td className="py-1">{feat} (mean)</td>
                                                        <td className="text-right py-1">{s.fake?.mean?.toFixed(2) || "—"}</td>
                                                        <td className="text-right py-1">{s.real?.mean?.toFixed(2) || "—"}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </CardContent>
                                </Card>

                                {["length_chars", "exclamation_count", "caps_ratio"]
                                    .filter((f) => data.text_stats[f])
                                    .map((feat) => {
                                        const d = data.text_stats[feat] as any;
                                        const hist = buildHistogramData(d.histogram_fake, d.histogram_real);
                                        return (
                                            <Card key={feat}>
                                                <CardHeader>
                                                    <CardTitle className="text-sm">{feat}</CardTitle>
                                                </CardHeader>
                                                <CardContent>
                                                    <ResponsiveContainer width="100%" height={180}>
                                                        <BarChart data={hist}>
                                                            <XAxis dataKey="bin" tick={{ fontSize: 10 }} />
                                                            <YAxis tick={{ fontSize: 10 }} />
                                                            <Tooltip />
                                                            <Legend />
                                                            <Bar dataKey="fake" fill={FAKE_COLOR} fillOpacity={0.6} name="FAKE" />
                                                            <Bar dataKey="real" fill={REAL_COLOR} fillOpacity={0.6} name="REAL" />
                                                        </BarChart>
                                                    </ResponsiveContainer>
                                                </CardContent>
                                            </Card>
                                        );
                                    })}
                            </TabsContent>

                            {}
                            <TabsContent value="words" className="space-y-3">
                                <div className="grid grid-cols-2 gap-3">
                                    <Card>
                                        <CardHeader>
                                            <CardTitle className="text-sm" style={{ color: FAKE_COLOR }}>
                                                Top FAKE indicators
                                            </CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <div className="flex flex-wrap gap-1">
                                                {(data.top_words.most_fake_indicative || []).map((w) => (
                                                    <Badge key={w.word} variant="outline" className="text-xs">
                                                        {w.word} ({w.ratio_fake_over_real.toFixed(1)}x)
                                                    </Badge>
                                                ))}
                                            </div>
                                        </CardContent>
                                    </Card>

                                    <Card>
                                        <CardHeader>
                                            <CardTitle className="text-sm" style={{ color: REAL_COLOR }}>
                                                Top REAL indicators
                                            </CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <div className="flex flex-wrap gap-1">
                                                {(data.top_words.most_real_indicative || []).map((w) => (
                                                    <Badge key={w.word} variant="outline" className="text-xs">
                                                        {w.word} ({(1 / w.ratio_fake_over_real).toFixed(1)}x)
                                                    </Badge>
                                                ))}
                                            </div>
                                        </CardContent>
                                    </Card>
                                </div>

                                <Card>
                                    <CardHeader>
                                        <CardTitle className="text-sm">Top words FAKE</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <ResponsiveContainer width="100%" height={300}>
                                            <BarChart data={data.top_words.fake?.slice(0, 20)} layout="vertical">
                                                <XAxis type="number" />
                                                <YAxis dataKey="word" type="category" width={80} tick={{ fontSize: 10 }} />
                                                <Tooltip />
                                                <Bar dataKey="count" fill={FAKE_COLOR} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </CardContent>
                                </Card>
                            </TabsContent>

                            {}
                            <TabsContent value="users" className="space-y-3">
                                {"error" in data.user_analysis ? (
                                    <p className="text-sm text-muted-foreground text-center py-6">
                                        User analysis недоступний (users.csv відсутній або немає потрібних полів)
                                    </p>
                                ) : (
                                    <Card>
                                        <CardHeader>
                                            <CardTitle className="text-base">Профілі користувачів</CardTitle>
                                        </CardHeader>
                                        <CardContent>
                                            <table className="w-full text-sm">
                                                <thead>
                                                    <tr className="border-b">
                                                        <th className="text-left py-1">Metric</th>
                                                        <th className="text-right py-1" style={{ color: FAKE_COLOR }}>FAKE</th>
                                                        <th className="text-right py-1" style={{ color: REAL_COLOR }}>REAL</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {Object.entries(data.user_analysis).map(([feat, s]: any) => {
                                                        if (feat === "error") return null;
                                                        if (typeof s === "number") {
                                                            return (
                                                                <tr key={feat} className="border-b">
                                                                    <td className="py-1">{feat}</td>
                                                                    <td colSpan={2} className="text-right py-1">{s}</td>
                                                                </tr>
                                                            );
                                                        }
                                                        if (s?.fake?.mean != null) {
                                                            return (
                                                                <tr key={feat} className="border-b">
                                                                    <td className="py-1">{feat} (mean)</td>
                                                                    <td className="text-right py-1">{s.fake.mean.toFixed(1)}</td>
                                                                    <td className="text-right py-1">{s.real.mean.toFixed(1)}</td>
                                                                </tr>
                                                            );
                                                        }
                                                        if (s?.fake != null && s?.real != null) {
                                                            return (
                                                                <tr key={feat} className="border-b">
                                                                    <td className="py-1">{feat}</td>
                                                                    <td className="text-right py-1">{(s.fake * 100).toFixed(1)}%</td>
                                                                    <td className="text-right py-1">{(s.real * 100).toFixed(1)}%</td>
                                                                </tr>
                                                            );
                                                        }
                                                        return null;
                                                    })}
                                                </tbody>
                                            </table>
                                        </CardContent>
                                    </Card>
                                )}

                                {Object.keys(data.engagement).length > 0 && (
                                    <Card>
                                        <CardHeader>
                                            <CardTitle className="text-base flex items-center gap-2">
                                                Engagement metrics
                                                <Badge variant="outline" className="text-xs">FAKE vs REAL</Badge>
                                            </CardTitle>
                                            <p className="text-xs text-muted-foreground">
                                                Розподіл взаємодій (likes, retweets, replies). Більший mean у FAKE
                                                підтверджує findings Vosoughi et al. 2018 (Science).
                                            </p>
                                        </CardHeader>
                                        <CardContent>
                                            <table className="w-full text-sm">
                                                <thead>
                                                    <tr className="border-b">
                                                        <th className="text-left py-1">Метрика</th>
                                                        <th className="text-right py-1 text-xs" style={{ color: FAKE_COLOR }}>FAKE mean</th>
                                                        <th className="text-right py-1 text-xs" style={{ color: FAKE_COLOR }}>FAKE median</th>
                                                        <th className="text-right py-1 text-xs" style={{ color: REAL_COLOR }}>REAL mean</th>
                                                        <th className="text-right py-1 text-xs" style={{ color: REAL_COLOR }}>REAL median</th>
                                                        <th className="text-right py-1 text-xs">Δ</th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    {Object.entries(data.engagement).map(([feat, s]: any) => {
                                                        const fakeM = s.fake?.mean ?? 0;
                                                        const realM = s.real?.mean ?? 0;
                                                        const diff = realM > 0 ? ((fakeM - realM) / realM * 100) : 0;
                                                        return (
                                                            <tr key={feat} className="border-b">
                                                                <td className="py-1 font-mono text-xs">{feat}</td>
                                                                <td className="text-right py-1">{s.fake?.mean?.toFixed(2) ?? "—"}</td>
                                                                <td className="text-right py-1 text-muted-foreground">{s.fake?.median?.toFixed(0) ?? "—"}</td>
                                                                <td className="text-right py-1">{s.real?.mean?.toFixed(2) ?? "—"}</td>
                                                                <td className="text-right py-1 text-muted-foreground">{s.real?.median?.toFixed(0) ?? "—"}</td>
                                                                <td className={cn(
                                                                    "text-right py-1 text-xs",
                                                                    Math.abs(diff) > 50 ? "font-semibold" : "text-muted-foreground"
                                                                )}>
                                                                    {diff > 0 ? "+" : ""}{diff.toFixed(0)}%
                                                                </td>
                                                            </tr>
                                                        );
                                                    })}
                                                </tbody>
                                            </table>
                                        </CardContent>
                                    </Card>
                                )}
                            </TabsContent>
                        </Tabs>
                    </div>
                )}
            </DialogContent>
        </Dialog>
    );
}