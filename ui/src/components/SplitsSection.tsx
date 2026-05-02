import React, { useEffect, useState } from "react";
import api from "../api";
import { Card, CardContent } from "./ui/card";
import { Badge } from "./ui/badge";
import { Label } from "./ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "./ui/select";
import { Loader2, Layers, AlertTriangle } from "lucide-react";
import { toast } from "sonner";
import type { Dataset, DatasetSplitsResponse, SplitInfo } from "../types";

const AUTO_VALUE = "__auto__";

function capitalize(s: string): string {
  if (!s) return s;
  return s
    .split(/[_\s]+/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

export interface SplitsSectionProps {
  dataset: Dataset;
  /** Called after successful PATCH so the parent can refetch dataset list. */
  onChange?: () => void | Promise<void>;
}

export default function SplitsSection({ dataset, onChange }: SplitsSectionProps) {
  const [data, setData] = useState<DatasetSplitsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .get<DatasetSplitsResponse>(`/datasets/${dataset.id}/splits`)
      .then((resp) => {
        if (!cancelled) setData(resp.data);
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.response?.data?.detail || "Не вдалось завантажити splits");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset.id]);

  const handleChange = async (rawValue: string) => {
    const splitName = rawValue === AUTO_VALUE ? null : rawValue;
    setSaving(true);
    try {
      await api.patch(`/datasets/${dataset.id}/active-split`, { split_name: splitName });
      toast.success(
        splitName
          ? `Split встановлено: ${capitalize(splitName)}`
          : "Auto-split увімкнено",
      );
      if (onChange) await onChange();
    } catch (err: any) {
      toast.error(err.response?.data?.detail || "Помилка збереження split");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        <div className="flex items-center gap-2">
          <Layers className="h-4 w-4 text-primary" />
          <p className="font-medium text-sm">Splits активного датасету</p>
          <span className="text-xs text-muted-foreground">— {dataset.name}</span>
        </div>

        {loading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Завантаження splits…
          </div>
        ) : error ? (
          <div className="flex items-start gap-2 text-sm text-amber-700 dark:text-amber-300">
            <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
            <div>
              <p>{error}</p>
              <p className="text-xs text-muted-foreground mt-1">
                Тренування все одно можливе — буде використано auto-split 70/15/15.
              </p>
            </div>
          </div>
        ) : data && data.splits.length === 0 && !data.has_legacy_splits ? (
          <Badge variant="outline" className="text-xs">
            Auto-split (no custom splits configured)
          </Badge>
        ) : data ? (
          <SplitsSelector
            splits={data.splits}
            value={dataset.active_split ?? null}
            disabled={saving}
            onChange={handleChange}
          />
        ) : null}
      </CardContent>
    </Card>
  );
}

interface SplitsSelectorProps {
  splits: SplitInfo[];
  value: string | null;
  disabled?: boolean;
  onChange: (rawValue: string) => void;
}

function SplitsSelector({ splits, value, disabled, onChange }: SplitsSelectorProps) {
  const selectValue = value ?? AUTO_VALUE;
  return (
    <div className="space-y-2">
      <Label className="text-xs">Active split</Label>
      <Select value={selectValue} disabled={disabled} onValueChange={onChange}>
        <SelectTrigger className="w-full max-w-md">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={AUTO_VALUE}>Auto-split (70/15/15)</SelectItem>
          {splits.map((s) => (
            <SelectItem key={s.name} value={s.name}>
              {capitalize(s.name)}{" "}
              <span className="text-muted-foreground ml-1">
                ({s.train_count}/{s.val_count}/{s.test_count})
              </span>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        {value
          ? `Тренування використає фіксований split: ${capitalize(value)}.`
          : "Тренування виконає авто-розбиття 70/15/15 на льоту."}
      </p>
    </div>
  );
}
