
import React from "react";
import { Button } from "./components/ui/button";
import {
    Dialog,
    DialogContent,
    DialogHeader,
    DialogTitle,
    DialogDescription,
} from "./components/ui/dialog";
import { Card, CardContent } from "./components/ui/card";
import { Badge } from "./components/ui/badge";
import { CheckCircle2, XCircle, Target, Sparkles } from "lucide-react";

interface EvaluationResult {
    model_id: number;
    model_name: string;
    model_type: string;
    dataset_id: number;
    dataset_name?: string;
    metrics: {
        accuracy: number;
        precision: number;
        recall: number;
        f1_score: number;

        f1_macro?: number;
        roc_auc?: number;
        best_epoch?: number;
        confusion_matrix?: { tn: number; fp: number; fn: number; tp: number };
        samples_evaluated?: number;
        training_time?: number;
    };
    samples?: Array<{
        index: number;
        text: string;
        true_label: number;
        pred_label: number;
        pred_str: string;
        true_str: string;
        confidence: number;
        reason?: string;
        correct: boolean;
    }>;
}

interface EvaluationResultsModalProps {
    open: boolean;
    onClose: () => void;
    result: EvaluationResult | null;
}

export default function EvaluationResultsModal({
    open,
    onClose,
    result,
}: EvaluationResultsModalProps) {
    if (!result) return null;

    const m = result.metrics;
    const cm = m.confusion_matrix;

    return (
        <Dialog open={open} onOpenChange={onClose}>
            <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
                <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                        <Target className="h-5 w-5" />
                        Результати оцінювання
                    </DialogTitle>
                    <DialogDescription>
                        <span className="font-medium">{result.model_name}</span>
                        {" — "}
                        <Badge variant="outline" className="ml-1 text-xs">
                            {result.model_type}
                        </Badge>
                        {result.dataset_name && (
                            <>
                                {" · на датасеті "}
                                <span className="italic">{result.dataset_name}</span>
                            </>
                        )}
                    </DialogDescription>
                </DialogHeader>

                <div className="space-y-4">
                    {}
                    <div className="grid grid-cols-4 gap-3">
                        {[
                            { label: "Accuracy", value: m.accuracy, color: "text-green-600" },
                            { label: "Precision (FAKE)", value: m.precision, color: "text-blue-600" },
                            { label: "Recall (FAKE)", value: m.recall, color: "text-violet-600" },
                            { label: "F1 (FAKE)", value: m.f1_score, color: "text-amber-600" },
                        ].map((metric) => (
                            <Card key={metric.label}>
                                <CardContent className="pt-5 pb-4 text-center">
                                    <div className={`text-2xl font-bold ${metric.color}`}>
                                        {metric.value != null ? (metric.value * 100).toFixed(1) + "%" : "—"}
                                    </div>
                                    <div className="text-xs text-muted-foreground mt-1">{metric.label}</div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>

                    {}
                    {(m.f1_macro != null || m.roc_auc != null) && (
                        <div>
                            <h3 className="text-sm font-semibold mb-2">Додаткові метрики</h3>
                            <div className="grid grid-cols-2 gap-3">
                                {[
                                    { label: "F1 (macro)", value: m.f1_macro, color: "text-emerald-600" },
                                    { label: "ROC AUC", value: m.roc_auc, color: "text-indigo-600" },
                                ].map((metric) => (
                                    <Card key={metric.label}>
                                        <CardContent className="pt-5 pb-4 text-center">
                                            <div className={`text-2xl font-bold ${metric.color}`}>
                                                {metric.value != null ? (metric.value * 100).toFixed(1) + "%" : "—"}
                                            </div>
                                            <div className="text-xs text-muted-foreground mt-1">{metric.label}</div>
                                        </CardContent>
                                    </Card>
                                ))}
                            </div>
                            {m.best_epoch != null && (
                                <p className="text-xs text-muted-foreground mt-2">
                                    Best epoch: <strong>{m.best_epoch}</strong>
                                </p>
                            )}
                        </div>
                    )}

                    {}
                    {cm && (
                        <div>
                            <h3 className="text-sm font-semibold mb-1">Матриця помилок</h3>
                            <p className="text-xs text-muted-foreground mb-2">FAKE = позитивний клас</p>
                            <div className="overflow-hidden rounded-lg border">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="bg-muted/50">
                                            <th className="p-2 text-left font-medium"></th>
                                            <th className="p-2 text-center font-medium">Передбачено: REAL</th>
                                            <th className="p-2 text-center font-medium">Передбачено: FAKE</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr className="border-t">
                                            <th className="p-2 text-left font-medium bg-muted/50">
                                                Фактично: REAL
                                            </th>
                                            <td className="p-3 text-center font-semibold text-green-600 bg-green-50 dark:bg-green-950/20">
                                                {cm.tn}
                                            </td>
                                            <td className="p-3 text-center font-semibold text-red-600 bg-red-50 dark:bg-red-950/20">
                                                {cm.fp}
                                            </td>
                                        </tr>
                                        <tr className="border-t">
                                            <th className="p-2 text-left font-medium bg-muted/50">
                                                Фактично: FAKE
                                            </th>
                                            <td className="p-3 text-center font-semibold text-red-600 bg-red-50 dark:bg-red-950/20">
                                                {cm.fn}
                                            </td>
                                            <td className="p-3 text-center font-semibold text-green-600 bg-green-50 dark:bg-green-950/20">
                                                {cm.tp}
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {}
                    <Card>
                        <CardContent className="p-3 text-xs text-muted-foreground">
                            <div className="flex items-center justify-between">
                                {m.samples_evaluated != null && (
                                    <span>Оцінено зразків: <strong>{m.samples_evaluated}</strong></span>
                                )}
                                {m.training_time != null && (
                                    <span>
                                        Час: <strong>{m.training_time.toFixed(1)}s</strong>
                                    </span>
                                )}
                            </div>
                        </CardContent>
                    </Card>

                    {}
                    {result.samples && result.samples.length > 0 && (
                        <div>
                            <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
                                <Sparkles className="h-4 w-4" />
                                Приклади передбачень
                            </h3>
                            <div className="space-y-2 max-h-96 overflow-y-auto">
                                {result.samples.slice(0, 20).map((s) => (
                                    <Card
                                        key={s.index}
                                        className={s.correct ? "border-green-300" : "border-red-300"}
                                    >
                                        <CardContent className="p-3 space-y-1.5">
                                            <div className="flex items-start gap-2">
                                                {s.correct ? (
                                                    <CheckCircle2 className="h-4 w-4 text-green-600 shrink-0 mt-0.5" />
                                                ) : (
                                                    <XCircle className="h-4 w-4 text-red-600 shrink-0 mt-0.5" />
                                                )}
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-xs text-muted-foreground line-clamp-2">
                                                        {s.text}
                                                    </p>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2 text-xs">
                                                <Badge variant={s.true_str === "FAKE" ? "destructive" : "secondary"} className="text-[10px]">
                                                    Справжнє: {s.true_str}
                                                </Badge>
                                                <Badge variant={s.pred_str === "FAKE" ? "destructive" : "secondary"} className="text-[10px]">
                                                    Передбачено: {s.pred_str}
                                                </Badge>
                                                <span className="text-muted-foreground">
                                                    confidence: {(s.confidence * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                            {s.reason && (
                                                <p className="text-xs italic text-muted-foreground pl-6">
                                                    "{s.reason}"
                                                </p>
                                            )}
                                        </CardContent>
                                    </Card>
                                ))}
                                {result.samples.length > 20 && (
                                    <p className="text-xs text-center text-muted-foreground py-2">
                                        ... ще {result.samples.length - 20} прикладів
                                    </p>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                <div className="flex justify-end pt-2">
                    <Button onClick={onClose}>Закрити</Button>
                </div>
            </DialogContent>
        </Dialog>
    );
}