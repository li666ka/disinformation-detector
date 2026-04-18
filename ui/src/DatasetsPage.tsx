import React, { useEffect, useState, useCallback } from "react";
import api from "./api";
import { cn } from "./lib/utils";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Input } from "./components/ui/input";
import { Textarea } from "./components/ui/textarea";
import { Label } from "./components/ui/label";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "./components/ui/table";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter,
  DialogHeader, DialogTitle,
} from "./components/ui/dialog";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "./components/ui/dropdown-menu";
import {
  Database, Upload, Download, Trash2, CheckCircle2, MoreVertical,
  FileText, Users, MessageCircle, Heart, Repeat2, Loader2, BarChart3,
  Edit3, AlertTriangle, Shield,
} from "lucide-react";
import { toast } from "sonner";
import type { Dataset, DatasetUploadResponse, DatasetStats } from "./types";

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("uk-UA");
}

const COMPONENT_BADGES: { key: keyof Dataset; label: string; icon: any }[] = [
  { key: "has_news", label: "News", icon: FileText },
  { key: "has_tweets", label: "Tweets", icon: MessageCircle },
  { key: "has_users", label: "Users", icon: Users },
  { key: "has_replies", label: "Replies", icon: MessageCircle },
  { key: "has_likes", label: "Likes", icon: Heart },
  { key: "has_retweets", label: "Retweets", icon: Repeat2 },
  { key: "has_evidence", label: "Evidence", icon: Shield },
];

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);

  // Upload modal
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [uploadDesc, setUploadDesc] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<DatasetUploadResponse | null>(null);

  // Stats modal
  const [statsOpen, setStatsOpen] = useState(false);
  const [statsDataset, setStatsDataset] = useState<Dataset | null>(null);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(false);

  // Rename modal
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameDataset, setRenameDataset] = useState<Dataset | null>(null);
  const [renameName, setRenameName] = useState("");
  const [renameDesc, setRenameDesc] = useState("");
  const [renaming, setRenaming] = useState(false);

  const fetchDatasets = useCallback(async () => {
    try {
      const { data } = await api.get<Dataset[]>("/datasets");
      setDatasets(data);
    } catch {
      toast.error("Не вдалося завантажити датасети");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchDatasets(); }, [fetchDatasets]);

  const handleActivate = async (id: number) => {
    try {
      await api.post(`/datasets/${id}/activate`);
      await fetchDatasets();
      toast.success("Датасет активовано");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка активації");
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm("Видалити датасет? Цю дію не можна скасувати.")) return;
    try {
      await api.delete(`/datasets/${id}`);
      await fetchDatasets();
      toast.success("Датасет видалено");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка видалення");
    }
  };

  const handleShowStats = async (ds: Dataset) => {
    setStatsDataset(ds);
    setStats(null);
    setStatsOpen(true);
    setStatsLoading(true);
    try {
      const { data } = await api.get<DatasetStats>(`/datasets/${ds.id}/stats`);
      setStats(data);
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Не вдалося завантажити статистику");
      setStatsOpen(false);
    } finally {
      setStatsLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!uploadFile || !uploadName.trim()) return;
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);
      formData.append("name", uploadName.trim());
      if (uploadDesc.trim()) formData.append("description", uploadDesc.trim());
      const { data } = await api.post<DatasetUploadResponse>("/datasets/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadResult(data);
      await fetchDatasets();
      toast.success("Датасет завантажено успішно!");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка завантаження");
    } finally {
      setUploading(false);
    }
  };

  const handleRename = async () => {
    if (!renameDataset || !renameName.trim()) return;
    setRenaming(true);
    try {
      await api.patch(`/datasets/${renameDataset.id}`, {
        name: renameName.trim(),
        description: renameDesc.trim() || null,
      });
      await fetchDatasets();
      setRenameOpen(false);
      toast.success("Датасет оновлено");
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка оновлення");
    } finally {
      setRenaming(false);
    }
  };

  const handleDownloadTemplate = () => {
    const token = localStorage.getItem("token");
    window.open(`http://localhost:8000/datasets/template/download?token=${token}`, "_blank");
  };

  const resetUploadModal = () => {
    setUploadOpen(false);
    setUploadFile(null);
    setUploadName("");
    setUploadDesc("");
    setUploadResult(null);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Датасети</h2>
          <p className="text-muted-foreground">Управління навчальними даними</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleDownloadTemplate}>
            <Download className="mr-2 h-4 w-4" /> Шаблон
          </Button>
          <Button size="sm" onClick={() => setUploadOpen(true)}>
            <Upload className="mr-2 h-4 w-4" /> Завантажити датасет
          </Button>
        </div>
      </div>

      {/* Content */}
      {datasets.length === 0 ? (
        <div className="text-center py-16">
          <Database className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">Датасетів ще немає</h3>
          <p className="text-muted-foreground mb-4">
            Завантажте ZIP-архів з файлом news.csv для початку роботи
          </p>
          <Button onClick={() => setUploadOpen(true)}>
            <Upload className="mr-2 h-4 w-4" /> Завантажити датасет
          </Button>
        </div>
      ) : (
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Назва</TableHead>
                <TableHead className="text-center">Новини</TableHead>
                <TableHead className="text-center">FAKE / REAL</TableHead>
                <TableHead>Компоненти</TableHead>
                <TableHead className="text-right">Розмір</TableHead>
                <TableHead className="text-center">Статус</TableHead>
                <TableHead className="w-10" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {datasets.map((ds) => (
                <TableRow key={ds.id} className={cn(ds.is_active && "bg-primary/5")}>
                  <TableCell>
                    <div>
                      <p className="font-medium text-sm">{ds.name}</p>
                      {ds.description && (
                        <p className="text-xs text-muted-foreground truncate max-w-[200px]">
                          {ds.description}
                        </p>
                      )}
                      <p className="text-xs text-muted-foreground">{formatDate(ds.created_at)}</p>
                    </div>
                  </TableCell>
                  <TableCell className="text-center font-medium">
                    {ds.total_news.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-center">
                    <span className="text-red-600 dark:text-red-400 font-medium">{ds.fake_count}</span>
                    {" / "}
                    <span className="text-green-600 dark:text-green-400 font-medium">{ds.real_count}</span>
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1">
                      {COMPONENT_BADGES.map(({ key, label }) =>
                        ds[key] ? (
                          <Badge key={key} variant="secondary" className="text-[10px] px-1.5 py-0">
                            {label}
                          </Badge>
                        ) : null
                      )}
                    </div>
                  </TableCell>
                  <TableCell className="text-right text-sm">
                    {formatBytes(ds.file_size_bytes)}
                  </TableCell>
                  <TableCell className="text-center">
                    {ds.is_active ? (
                      <Badge className="bg-green-100 text-green-800 dark:bg-green-950 dark:text-green-400 border-green-300">
                        <CheckCircle2 className="h-3 w-3 mr-1" /> Активний
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="text-muted-foreground">
                        Неактивний
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon" className="h-8 w-8">
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        {!ds.is_active && (
                          <DropdownMenuItem onClick={() => handleActivate(ds.id)}>
                            <CheckCircle2 className="mr-2 h-4 w-4" /> Активувати
                          </DropdownMenuItem>
                        )}
                        <DropdownMenuItem onClick={() => handleShowStats(ds)}>
                          <BarChart3 className="mr-2 h-4 w-4" /> Статистика
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => {
                          setRenameDataset(ds);
                          setRenameName(ds.name);
                          setRenameDesc(ds.description || "");
                          setRenameOpen(true);
                        }}>
                          <Edit3 className="mr-2 h-4 w-4" /> Редагувати
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          className="text-destructive"
                          onClick={() => handleDelete(ds.id)}
                        >
                          <Trash2 className="mr-2 h-4 w-4" /> Видалити
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      {/* ── Upload Dialog ──────────────────────────────────────────────── */}
      <Dialog open={uploadOpen} onOpenChange={(open) => { if (!open) resetUploadModal(); }}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Завантажити датасет</DialogTitle>
            <DialogDescription>
              Очікується ZIP з файлами news.csv (обов'язково) + опц. tweets.csv, users.csv та ін.
            </DialogDescription>
          </DialogHeader>

          {!uploadResult ? (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Назва датасету *</Label>
                <Input
                  value={uploadName}
                  onChange={(e) => setUploadName(e.target.value)}
                  placeholder="Наприклад: FakeNewsNet v2"
                />
              </div>
              <div className="space-y-2">
                <Label>Опис (необов'язково)</Label>
                <Textarea
                  value={uploadDesc}
                  onChange={(e) => setUploadDesc(e.target.value)}
                  placeholder="Короткий опис датасету..."
                  rows={2}
                />
              </div>
              <div className="space-y-2">
                <Label>ZIP-файл *</Label>
                <div
                  className={cn(
                    "border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors",
                    uploadFile
                      ? "border-primary bg-primary/5"
                      : "border-border hover:border-primary/50"
                  )}
                  onClick={() => document.getElementById("dataset-file-input")?.click()}
                  onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
                  onDrop={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const file = e.dataTransfer.files[0];
                    if (file) setUploadFile(file);
                  }}
                >
                  {uploadFile ? (
                    <div className="flex items-center justify-center gap-2">
                      <FileText className="h-5 w-5 text-primary" />
                      <span className="text-sm font-medium">{uploadFile.name}</span>
                      <span className="text-xs text-muted-foreground">
                        ({formatBytes(uploadFile.size)})
                      </span>
                    </div>
                  ) : (
                    <>
                      <Upload className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                      <p className="text-sm text-muted-foreground">
                        Перетягніть файл сюди або клікніть для вибору
                      </p>
                    </>
                  )}
                  <input
                    id="dataset-file-input"
                    type="file"
                    accept=".zip"
                    className="hidden"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) setUploadFile(file);
                    }}
                  />
                </div>
              </div>

              <DialogFooter>
                <Button variant="outline" onClick={resetUploadModal}>Скасувати</Button>
                <Button
                  onClick={handleUpload}
                  disabled={!uploadFile || !uploadName.trim() || uploading}
                >
                  {uploading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  {uploading ? "Завантаження..." : "Завантажити"}
                </Button>
              </DialogFooter>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                <CheckCircle2 className="h-5 w-5" />
                <span className="font-medium">Датасет завантажено успішно!</span>
              </div>

              {uploadResult.warnings.length > 0 && (
                <div className="rounded-lg bg-amber-50 dark:bg-amber-950/20 p-3 space-y-1">
                  {uploadResult.warnings.map((w, i) => (
                    <div key={i} className="flex items-start gap-2 text-sm text-amber-800 dark:text-amber-300">
                      <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
                      <span>{w}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Preview table */}
              {uploadResult.preview.news_head.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium">Перші рядки news.csv:</p>
                  <div className="rounded-md border overflow-auto max-h-48">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          {uploadResult.preview.news_columns.slice(0, 5).map((col) => (
                            <TableHead key={col} className="text-xs whitespace-nowrap">{col}</TableHead>
                          ))}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {uploadResult.preview.news_head.slice(0, 5).map((row, i) => (
                          <TableRow key={i}>
                            {uploadResult!.preview.news_columns.slice(0, 5).map((col) => (
                              <TableCell key={col} className="text-xs max-w-[200px] truncate">
                                {String(row[col] ?? "")}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}

              <DialogFooter>
                <Button onClick={resetUploadModal}>Готово</Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* ── Stats Dialog ───────────────────────────────────────────────── */}
      <Dialog open={statsOpen} onOpenChange={setStatsOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Статистика: {statsDataset?.name}</DialogTitle>
          </DialogHeader>

          {statsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : stats ? (
            <div className="space-y-4">
              {/* Main metrics */}
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Всього новин", value: stats.total_news.toLocaleString() },
                  { label: "FAKE", value: stats.fake_count.toLocaleString(), color: "text-red-600 dark:text-red-400" },
                  { label: "REAL", value: stats.real_count.toLocaleString(), color: "text-green-600 dark:text-green-400" },
                  { label: "Без мітки", value: stats.unlabeled_count.toLocaleString() },
                ].map((m) => (
                  <Card key={m.label}>
                    <CardContent className="p-3 text-center">
                      <p className={cn("text-xl font-bold", m.color)}>{m.value}</p>
                      <p className="text-xs text-muted-foreground">{m.label}</p>
                    </CardContent>
                  </Card>
                ))}
              </div>

              {/* Text length stats */}
              <div>
                <p className="text-sm font-medium mb-2">Довжина тексту (символів)</p>
                <div className="grid grid-cols-5 gap-2 text-center text-xs">
                  {[
                    { label: "Min", value: stats.text_length_stats.min },
                    { label: "Median", value: Math.round(stats.text_length_stats.median) },
                    { label: "Mean", value: Math.round(stats.text_length_stats.mean) },
                    { label: "P95", value: Math.round(stats.text_length_stats.p95) },
                    { label: "Max", value: stats.text_length_stats.max },
                  ].map((s) => (
                    <div key={s.label} className="p-2 bg-muted rounded">
                      <p className="font-semibold">{s.value.toLocaleString()}</p>
                      <p className="text-muted-foreground">{s.label}</p>
                    </div>
                  ))}
                </div>
              </div>

              {(stats.empty_text_count > 0 || stats.duplicate_text_count > 0) && (
                <div className="text-sm space-y-1">
                  {stats.empty_text_count > 0 && (
                    <p className="text-amber-600">Порожніх текстів: {stats.empty_text_count}</p>
                  )}
                  {stats.duplicate_text_count > 0 && (
                    <p className="text-amber-600">Дублікатів: {stats.duplicate_text_count}</p>
                  )}
                </div>
              )}

              {/* Tweets section */}
              {stats.total_tweets != null && (
                <div>
                  <p className="text-sm font-medium mb-2">Твіти</p>
                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="p-2 bg-muted rounded">
                      <p className="font-semibold">{stats.total_tweets?.toLocaleString()}</p>
                      <p className="text-muted-foreground">Всього</p>
                    </div>
                    {stats.avg_tweets_per_news != null && (
                      <div className="p-2 bg-muted rounded">
                        <p className="font-semibold">{stats.avg_tweets_per_news?.toFixed(1)}</p>
                        <p className="text-muted-foreground">Сер. на новину</p>
                      </div>
                    )}
                    {stats.news_with_tweets_pct != null && (
                      <div className="p-2 bg-muted rounded">
                        <p className="font-semibold">{stats.news_with_tweets_pct?.toFixed(1)}%</p>
                        <p className="text-muted-foreground">З твітами</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Users section */}
              {stats.total_users != null && (
                <div>
                  <p className="text-sm font-medium mb-2">Користувачі</p>
                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="p-2 bg-muted rounded">
                      <p className="font-semibold">{stats.total_users?.toLocaleString()}</p>
                      <p className="text-muted-foreground">Всього</p>
                    </div>
                    {stats.verified_users_pct != null && (
                      <div className="p-2 bg-muted rounded">
                        <p className="font-semibold">{stats.verified_users_pct?.toFixed(1)}%</p>
                        <p className="text-muted-foreground">Верифіковані</p>
                      </div>
                    )}
                    {stats.avg_followers_count != null && (
                      <div className="p-2 bg-muted rounded">
                        <p className="font-semibold">{Math.round(stats.avg_followers_count!).toLocaleString()}</p>
                        <p className="text-muted-foreground">Сер. фоловерів</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Top domains */}
              {stats.top_domains && stats.top_domains.length > 0 && (
                <div>
                  <p className="text-sm font-medium mb-2">Топ домени</p>
                  <div className="rounded-md border overflow-auto max-h-40">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="text-xs">Домен</TableHead>
                          <TableHead className="text-xs text-right">Кількість</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {stats.top_domains.slice(0, 10).map((d) => (
                          <TableRow key={d.domain}>
                            <TableCell className="text-xs">{d.domain}</TableCell>
                            <TableCell className="text-xs text-right">{d.count}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}
            </div>
          ) : null}
        </DialogContent>
      </Dialog>

      {/* ── Rename Dialog ──────────────────────────────────────────────── */}
      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <DialogTitle>Редагувати датасет</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Назва</Label>
              <Input
                value={renameName}
                onChange={(e) => setRenameName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Опис</Label>
              <Textarea
                value={renameDesc}
                onChange={(e) => setRenameDesc(e.target.value)}
                rows={2}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameOpen(false)}>Скасувати</Button>
            <Button onClick={handleRename} disabled={!renameName.trim() || renaming}>
              {renaming && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Зберегти
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
