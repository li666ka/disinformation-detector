import React, { useEffect, useState } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Checkbox } from "./components/ui/checkbox";
import { Search, Link, Clock, Globe, Loader2, Rss } from "lucide-react";
import { toast } from "sonner";
import PostCard from "./PostCard";
import VerifyModal from "./VerifyModal";
import type { ModelRecord, NewsItem, ClassifiedPost, SourceType } from "./types";
import PostDetailsModal from "./PostDetailsModal";



type Mode = "search" | "recent" | "url";

function BlueskyIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 600 530" className={className} fill="currentColor">
      <path d="m135.72 44.03c66.496 49.921 138.02 151.14 164.28 205.46 26.262-54.316 97.782-155.54 164.28-205.46 47.98-36.021 125.72-63.892 125.72 24.795 0 17.712-10.155 148.79-16.111 170.07-20.703 73.984-96.144 92.854-163.25 81.433 117.3 19.964 147.14 86.092 82.697 152.22-122.39 125.59-175.91-31.511-189.63-71.766-2.514-7.3797-3.6904-10.832-3.7077-7.8964-0.0174-2.9357-1.1937 0.51669-3.7077 7.8964-13.714 40.255-67.233 197.36-189.63 71.766-64.444-66.128-34.605-132.26 82.697-152.22-67.108 11.421-142.55-7.4491-163.25-81.433-5.9562-21.282-16.111-152.36-16.111-170.07 0-88.687 77.742-60.816 125.72-24.795z" />
    </svg>
  );
}

function MastodonIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="currentColor">
      <path d="M23.268 5.313c-.35-2.578-2.617-4.61-5.304-5.004C17.51.242 15.792 0 11.813 0h-.03c-3.98 0-4.835.242-5.288.309C3.882.692 1.496 2.518.917 5.127.64 6.412.61 7.837.661 9.143c.074 1.874.088 3.745.26 5.611.118 1.24.325 2.47.62 3.68.55 2.237 2.777 4.098 4.96 4.857 2.336.792 4.849.923 7.256.38.265-.061.527-.132.786-.213.585-.184 1.27-.39 1.774-.753a.057.057 0 0 0 .023-.043v-1.809a.052.052 0 0 0-.02-.041.053.053 0 0 0-.046-.01 20.282 20.282 0 0 1-4.709.545c-2.73 0-3.463-1.284-3.674-1.818a5.593 5.593 0 0 1-.319-1.433.053.053 0 0 1 .066-.054 19.648 19.648 0 0 0 4.168.507h.313c1.598-.009 3.186-.064 4.759-.29 1.699-.243 3.23-1.057 3.418-4.023.09-1.396.079-2.84.012-4.295-.014-.316-.012-.636-.012-.958ZM19.07 12.77h-2.29V7.63c0-1.084-.459-1.635-1.375-1.635-1.013 0-1.521.654-1.521 1.946v2.746h-2.278V7.94c0-1.292-.508-1.946-1.521-1.946-.92 0-1.378.55-1.378 1.635v5.14H6.414V7.47c0-1.084.277-1.946.832-2.586.572-.64 1.32-.968 2.25-.968 1.074 0 1.887.412 2.428 1.236l.523.876.523-.876c.54-.824 1.354-1.236 2.428-1.236.93 0 1.678.328 2.25.968.555.64.832 1.502.832 2.586v5.3Z" />
    </svg>
  );
}

const SOURCE_CONFIG: Record<SourceType, { label: string; icon: any; color: string; bg: string }> = {
  bluesky: {
    label: "Bluesky",
    icon: BlueskyIcon,
    color: "text-sky-600 dark:text-sky-400",
    bg: "bg-sky-50 dark:bg-sky-950/30 border-sky-200 dark:border-sky-800",
  },
  mastodon: {
    label: "Mastodon",
    icon: MastodonIcon,
    color: "text-indigo-600 dark:text-indigo-400",
    bg: "bg-indigo-50 dark:bg-indigo-950/30 border-indigo-200 dark:border-indigo-800",
  },
  rss: {
    label: "RSS",
    icon: Rss,
    color: "text-orange-600 dark:text-orange-400",
    bg: "bg-orange-50 dark:bg-orange-950/30 border-orange-200 dark:border-orange-800",
  },
};

export default function SourcesPage() {

  const [mode, setMode] = useState<Mode>("search");
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [modelId, setModelId] = useState<number | null>(null);
  const [loadingModels, setLoadingModels] = useState<boolean>(true);
  const [posts, setPosts] = useState<ClassifiedPost[]>([]);
  const [loadingPosts, setLoadingPosts] = useState<boolean>(false);
  const [query, setQuery] = useState<string>("");
  const [sources, setSources] = useState<Set<SourceType>>(new Set<SourceType>(["bluesky", "mastodon", "rss"]));
  const [limit, setLimit] = useState<number>(20);
  const [postUrl, setPostUrl] = useState<string>("");
  const [classifyingIds, setClassifyingIds] = useState<Set<string>>(new Set());
  const [batchClassifying, setBatchClassifying] = useState<boolean>(false);
  const [verifyOpen, setVerifyOpen] = useState(false);
  const [verifyTarget, setVerifyTarget] = useState<{
    url?: string;
    title?: string;
    text?: string;
  } | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [detailsPostId, setDetailsPostId] = useState<string | null>(null);

  const openDetails = (post: any) => {
    setDetailsPostId(post.id);
    setDetailsOpen(true);
  };

  const openVerify = (post: any) => {
    setVerifyTarget({
      url: post.url,
      title: post.title || undefined,
      text: post.title ? undefined : post.text,
    });
    setVerifyOpen(true);
  };

  useEffect(() => {
    api.get("/models")
      .then(({ data }) => {
        setModels(data);
        const active = data.find((m: ModelRecord) => m.is_active);
        if (active) setModelId(active.id);
        else if (data.length > 0) setModelId(data[0].id);
      })
      .catch(() => toast.error("Не вдалося завантажити моделі"))
      .finally(() => setLoadingModels(false));
  }, []);

  const toggleSource = (src: SourceType) => {
    setSources((prev) => {
      const next = new Set(prev);
      if (next.has(src)) next.delete(src); else next.add(src);
      return next;
    });
  };

  const handleSearch = async () => {
    if (!query.trim()) { toast.error("Введіть пошуковий запит"); return; }
    if (sources.size === 0) { toast.error("Оберіть хоча б одне джерело"); return; }
    setLoadingPosts(true); setPosts([]);
    try {
      const { data } = await api.get<{ posts: NewsItem[] }>("/sources/search", {
        params: { query, sources: Array.from(sources).join(","), limit },
      });
      setPosts(data.posts.map((p) => ({ ...p, classification: null })));
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка пошуку");
    } finally { setLoadingPosts(false); }
  };

  const handleFetchRecent = async () => {
    if (sources.size === 0) { toast.error("Оберіть хоча б одне джерело"); return; }
    setLoadingPosts(true); setPosts([]);
    try {
      const { data } = await api.get<{ posts: NewsItem[] }>("/sources/recent", {
        params: { sources: Array.from(sources).join(","), limit },
      });
      setPosts(data.posts.map((p) => ({ ...p, classification: null })));
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Не вдалося отримати пости");
    } finally { setLoadingPosts(false); }
  };

  const handleFetchUrl = async () => {
    if (!postUrl.trim()) { toast.error("Введіть URL поста"); return; }
    setLoadingPosts(true); setPosts([]);
    try {
      const { data } = await api.post<NewsItem>("/sources/fetch-url", { url: postUrl });
      setPosts([{ ...data, classification: null }]);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Не вдалося завантажити пост");
    } finally { setLoadingPosts(false); }
  };

  const classifyPost = async (post: ClassifiedPost) => {
    if (!modelId) { toast.error("Оберіть модель для класифікації"); return; }
    setClassifyingIds((prev) => new Set(prev).add(post.id));
    try {
      const { data } = await api.post("/analyze", { text: post.text, model_id: modelId });
      setPosts((prev) => prev.map((p) =>
        p.id === post.id ? { ...p, classification: { label: data.label, confidence: data.confidence, probability: data.probability } } : p
      ));
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка класифікації");
    } finally {
      setClassifyingIds((prev) => { const next = new Set(prev); next.delete(post.id); return next; });
    }
  };

  const classifyAll = async () => {
    if (!modelId) { toast.error("Оберіть модель для класифікації"); return; }
    const unclassified = posts.filter((p) => !p.classification);
    if (unclassified.length === 0) return;
    setBatchClassifying(true);
    for (const post of unclassified) { await classifyPost(post); }
    setBatchClassifying(false);
  };

  const classifiedPosts = posts.filter((p) => p.classification);
  const fakeCount = classifiedPosts.filter((p) => p.classification?.label === "FAKE").length;
  const realCount = classifiedPosts.filter((p) => p.classification?.label === "REAL").length;

  const MODES = [
    { id: "search" as Mode, icon: Search, title: "Пошук за словами", desc: "Знайти пости за ключовими словами" },
    { id: "recent" as Mode, icon: Clock, title: "Свіжі пости", desc: "Останні публікації" },
    { id: "url" as Mode, icon: Link, title: "За URL", desc: "Завантажити за посиланням" },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Реальні дані</h2>
        <p className="text-muted-foreground">Аналізуйте пости з соціальних мереж та RSS-стрічок</p>
      </div>

      {/* Mode selector */}
      <div className="grid grid-cols-3 gap-3">
        {MODES.map((m) => {
          const Icon = m.icon;
          return (
            <Card
              key={m.id}
              className={cn(
                "cursor-pointer transition-all hover:shadow-md",
                mode === m.id && "ring-2 ring-primary"
              )}
              onClick={() => { setMode(m.id); setPosts([]); }}
            >
              <CardContent className="p-4 text-center">
                <Icon className={cn("h-5 w-5 mx-auto mb-2", mode === m.id ? "text-primary" : "text-muted-foreground")} />
                <p className="text-sm font-semibold">{m.title}</p>
                <p className="text-xs text-muted-foreground mt-1">{m.desc}</p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Sources selection */}
      {mode !== "url" && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Джерела даних</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-3 gap-3">
              {(Object.keys(SOURCE_CONFIG) as SourceType[]).map((src) => {
                const cfg = SOURCE_CONFIG[src];
                const Icon = cfg.icon;
                const active = sources.has(src);
                return (
                  <div
                    key={src}
                    onClick={() => toggleSource(src)}
                    className={cn(
                      "flex items-center gap-3 p-4 rounded-lg border-2 cursor-pointer transition-all",
                      active
                        ? cn("ring-2 ring-primary", cfg.bg)
                        : "border-border bg-muted/30 hover:bg-muted/50 opacity-60"
                    )}
                  >
                    <div className={cn(
                      "w-10 h-10 rounded-lg flex items-center justify-center shrink-0",
                      active ? cfg.color : "text-muted-foreground",
                      active ? cfg.bg : "bg-muted"
                    )}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className={cn("font-semibold text-sm", active ? cfg.color : "text-muted-foreground")}>
                        {cfg.label}
                      </p>
                    </div>
                    <Checkbox
                      checked={active}
                      onCheckedChange={() => toggleSource(src)}
                    />
                  </div>
                );
              })}
            </div>
            <div className="flex items-center gap-3">
              <span className="text-sm text-muted-foreground">Кількість:</span>
              <Input
                type="number"
                min={1}
                max={500}
                value={limit}
                onChange={(e) => setLimit(Math.max(1, parseInt(e.target.value) || 1))}
                className="w-24"
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Search input */}
      {mode === "search" && (
        <Card>
          <CardContent className="p-4">
            <div className="flex gap-2">
              <Input
                placeholder="Наприклад: election, COVID, climate change..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSearch()}
                className="flex-1"
              />
              <Button onClick={handleSearch} disabled={loadingPosts || !query.trim()}>
                {loadingPosts ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4 mr-2" />}
                {loadingPosts ? "" : "Шукати"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent mode */}
      {mode === "recent" && (
        <Button onClick={handleFetchRecent} disabled={loadingPosts}>
          {loadingPosts && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          {loadingPosts ? "Завантаження..." : "Завантажити"}
        </Button>
      )}

      {/* URL mode */}
      {mode === "url" && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">URL поста або статті</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Підтримуються: Bluesky, Mastodon та RSS-статті
            </p>
            <div className="flex gap-2">
              <Input
                type="url"
                placeholder="https://bsky.app/profile/user.bsky.social/post/..."
                value={postUrl}
                onChange={(e) => setPostUrl(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleFetchUrl()}
                className="flex-1"
              />
              <Button onClick={handleFetchUrl} disabled={loadingPosts || !postUrl.trim()}>
                {loadingPosts ? <Loader2 className="h-4 w-4 animate-spin" /> : "Завантажити"}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Model selector */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Модель для класифікації</CardTitle>
        </CardHeader>
        <CardContent>
          {loadingModels ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Завантаження моделей...
            </div>
          ) : models.length === 0 ? (
            <p className="text-sm text-muted-foreground">Навчених моделей немає.</p>
          ) : (
            <Select value={modelId?.toString() || ""} onValueChange={(v) => setModelId(Number(v))}>
              <SelectTrigger className="max-w-sm">
                <SelectValue placeholder="Оберіть модель" />
              </SelectTrigger>
              <SelectContent>
                {models.map((m) => (
                  <SelectItem key={m.id} value={m.id.toString()}>
                    {m.name || m.model_type}
                    {m.accuracy != null ? ` — ${(m.accuracy * 100).toFixed(1)}%` : ""}
                    {m.is_active ? " (активна)" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </CardContent>
      </Card>

      {/* Results header + batch classify */}
      {posts.length > 0 && (
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <span className="font-semibold">Знайдено постів: {posts.length}</span>
                {classifiedPosts.length > 0 && (
                  <span className="text-sm text-muted-foreground ml-3">
                    (проаналізовано: {classifiedPosts.length})
                  </span>
                )}
              </div>
              {modelId && (
                <Button size="sm" onClick={classifyAll} disabled={batchClassifying || classifyingIds.size > 0}>
                  {batchClassifying && <Loader2 className="mr-2 h-3 w-3 animate-spin" />}
                  {batchClassifying ? `Аналіз... (${classifiedPosts.length}/${posts.length})` : "Аналізувати всі"}
                </Button>
              )}
            </div>

            {classifiedPosts.length > 0 && (
              <div className="grid grid-cols-3 gap-3">
                <div className="text-center p-3 rounded-lg bg-red-50 dark:bg-red-950/20">
                  <div className="text-2xl font-bold text-red-600 dark:text-red-400">{fakeCount}</div>
                  <div className="text-xs text-muted-foreground">Фейки</div>
                </div>
                <div className="text-center p-3 rounded-lg bg-green-50 dark:bg-green-950/20">
                  <div className="text-2xl font-bold text-green-600 dark:text-green-400">{realCount}</div>
                  <div className="text-xs text-muted-foreground">Достовірні</div>
                </div>
                <div className="text-center p-3 rounded-lg bg-blue-50 dark:bg-blue-950/20">
                  <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                    {classifiedPosts.length > 0 ? `${((fakeCount / classifiedPosts.length) * 100).toFixed(0)}%` : "—"}
                  </div>
                  <div className="text-xs text-muted-foreground">% фейків</div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Posts list */}
      {posts.map((post) => (
        <PostCard
          key={post.id}
          post={post}
          classifying={classifyingIds.has(post.id)}
          canClassify={!!modelId}
          onClassify={() => classifyPost(post)}
          onVerify={() => openVerify(post)}
          onDetails={() => openDetails(post)}
        />
      ))}

      {/* Empty state */}
      {posts.length === 0 && !loadingPosts && (
        <div className="text-center py-12">
          <Globe className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <p className="text-muted-foreground">
            {mode === "search" && "Введіть запит і натисніть «Шукати»"}
            {mode === "recent" && "Натисніть «Завантажити» щоб отримати свіжі пости"}
            {mode === "url" && "Вставте URL поста і натисніть «Завантажити»"}
          </p>
        </div>
      )}

      {/* Verify modal */}
      {verifyTarget && (
        <VerifyModal
          open={verifyOpen}
          onClose={() => setVerifyOpen(false)}
          url={verifyTarget.url}
          title={verifyTarget.title}
          text={verifyTarget.text}
        />
      )}

      {detailsPostId && (
        <PostDetailsModal
          open={detailsOpen}
          onClose={() => setDetailsOpen(false)}
          postId={detailsPostId}
        />
      )}
    </div>


  );
}
