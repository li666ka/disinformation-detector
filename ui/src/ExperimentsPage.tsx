import React, { useEffect, useState } from "react";
import api from "./api";
import { Button } from "./components/ui/button";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "./components/ui/dialog";
import { BarChart3, Loader2, Trash2, X } from "lucide-react";
import { toast } from "sonner";
import type { Experiment } from "./types";

function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedExp, setSelectedExp] = useState<Experiment | null>(null);

  const loadExperiments = () => {
    setLoading(true);
    api
      .get("/experiments")
      .then(({ data }) => setExperiments(data))
      .catch(() => toast.error("Не вдалося завантажити експерименти"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { loadExperiments(); }, []);

  const deleteExperiment = async (id: number) => {
    try {
      await api.delete(`/experiments/${id}`);
      setExperiments((prev) => prev.filter((e) => e.id !== id));
      toast.success("Експеримент видалено");
    } catch {
      toast.error("Не вдалося видалити експеримент");
    }
  };

  const deleteAll = async () => {
    if (!window.confirm("Видалити всі експерименти?")) return;
    try {
      await api.delete("/experiments");
      setExperiments([]);
      toast.success("Всі експерименти видалено");
    } catch {
      toast.error("Не вдалося видалити експерименти");
    }
  };

  const formatDate = (iso: string) => new Date(iso).toLocaleString("uk-UA");

  // Stats
  const successExps = experiments.filter((e) => e.status === "success");
  const avgAccuracy = successExps.length > 0
    ? successExps.reduce((sum, e) => sum + (e.accuracy || 0), 0) / successExps.length
    : 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Експерименти</h2>
          <p className="text-muted-foreground">Журнал навчання моделей</p>
        </div>
        {experiments.length > 0 && (
          <Button variant="destructive" size="sm" onClick={deleteAll}>
            <Trash2 className="mr-2 h-4 w-4" />
            Видалити всі
          </Button>
        )}
      </div>

      {/* Stats */}
      {experiments.length > 0 && (
        <div className="grid grid-cols-3 gap-4">
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold">{experiments.length}</div>
              <div className="text-xs text-muted-foreground mt-1">Всього</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                {successExps.length}
              </div>
              <div className="text-xs text-muted-foreground mt-1">Успішних</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                {avgAccuracy > 0 ? `${(avgAccuracy * 100).toFixed(1)}%` : "—"}
              </div>
              <div className="text-xs text-muted-foreground mt-1">Середня accuracy</div>
            </CardContent>
          </Card>
        </div>
      )}

      {experiments.length === 0 ? (
        <div className="text-center py-16">
          <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <p className="text-muted-foreground">
            Експериментів ще немає. Запустіть перше навчання у вкладці "Навчання моделі".
          </p>
        </div>
      ) : (
        <Card>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Тип моделі</TableHead>
                <TableHead>Accuracy</TableHead>
                <TableHead>F1</TableHead>
                <TableHead>Статус</TableHead>
                <TableHead>Дата</TableHead>
                <TableHead className="w-[50px]"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {experiments.map((exp) => (
                <TableRow
                  key={exp.id}
                  className="cursor-pointer"
                  onClick={() => setSelectedExp(exp)}
                >
                  <TableCell className="font-mono text-xs">
                    {String(exp.experiment_id).slice(0, 8)}...
                  </TableCell>
                  <TableCell>{exp.model_type}</TableCell>
                  <TableCell>
                    {exp.accuracy != null ? `${(exp.accuracy * 100).toFixed(1)}%` : "—"}
                  </TableCell>
                  <TableCell>
                    {exp.f1_score != null ? `${(exp.f1_score * 100).toFixed(1)}%` : "—"}
                  </TableCell>
                  <TableCell>
                    <Badge variant={exp.status === "success" ? "default" : "destructive"} className={exp.status === "success" ? "bg-green-100 text-green-700 dark:bg-green-950 dark:text-green-300 hover:bg-green-100" : ""}>
                      {exp.status === "success" ? "Успішно" : "Помилка"}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground text-xs">
                    {formatDate(exp.created_at)}
                  </TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-destructive hover:text-destructive"
                      onClick={(e) => { e.stopPropagation(); deleteExperiment(exp.id); }}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Card>
      )}

      {/* Detail Dialog */}
      <Dialog open={!!selectedExp} onOpenChange={() => setSelectedExp(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Деталі експерименту</DialogTitle>
            <DialogDescription>
              {selectedExp?.experiment_id}
            </DialogDescription>
          </DialogHeader>
          {selectedExp && (
            <div className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <span className="text-muted-foreground">Тип моделі</span>
                  <p className="font-medium">{selectedExp.model_type}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Статус</span>
                  <p className="font-medium">{selectedExp.status === "success" ? "Успішно" : "Помилка"}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Accuracy</span>
                  <p className="font-medium">{selectedExp.accuracy != null ? `${(selectedExp.accuracy * 100).toFixed(1)}%` : "—"}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">F1 Score</span>
                  <p className="font-medium">{selectedExp.f1_score != null ? `${(selectedExp.f1_score * 100).toFixed(1)}%` : "—"}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Precision</span>
                  <p className="font-medium">{selectedExp.precision != null ? `${(selectedExp.precision * 100).toFixed(1)}%` : "—"}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Recall</span>
                  <p className="font-medium">{selectedExp.recall != null ? `${(selectedExp.recall * 100).toFixed(1)}%` : "—"}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Тренувальний набір</span>
                  <p className="font-medium">{selectedExp.train_size || "—"}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Тестовий набір</span>
                  <p className="font-medium">{selectedExp.test_size || "—"}</p>
                </div>
              </div>
              <div>
                <span className="text-muted-foreground">Дата</span>
                <p className="font-medium">{formatDate(selectedExp.created_at)}</p>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default ExperimentsPage;
