/**
 * Header model bar: UI preset, checkpoint (searchable, with architecture and
 * size), VAE / text encoders (`forge_additional_modules`, applied when the
 * picker closes so one change loads the model once) and "Diffusion in Low
 * Bits". While the server loads a model, `model.loading` shows its stage.
 */
import { useQuery } from "@tanstack/react-query"
import { Box, Layers, Loader2, SlidersHorizontal } from "lucide-react"
import { useEffect, useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useClientOption } from "@/api/app-options"
import { useProgress } from "@/api/progress"
import { modelsQuery } from "@/api/queries"
import { SearchSelect } from "@/components/SearchSelect"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { cn } from "@/lib/utils"
import { checkpointLabel, fileName, formatBytes, MODEL_OPTION_KEYS, presetLabel, useApplyPreset, useOptionWrite, writeModules } from "./model-options"

const DEFAULT_PRESETS = ["sd", "xl", "flux", "anima"]
const DEFAULT_DTYPES = ["Automatic", "Automatic (fp16 LoRA)", "float8-e4m3fn", "float8-e5m2"]

const useModelBusy = () => useProgress((s) => s.modelLoading)

function PresetSelect({ className }: { className?: string }) {
  const { t } = useTranslation()
  const models = useQuery(modelsQuery)
  const current = useClientOption("forge_preset")
  const apply = useApplyPreset()
  const presets = models.data?.presets ?? DEFAULT_PRESETS
  const value = apply.isPending ? (apply.variables ?? current) : current
  const items = presets.map((id) => ({ value: id, label: presetLabel(id) }))
  return (
    <Select<string> items={items} value={value} onValueChange={(next) => next && next !== current && apply.mutate(next)} disabled={apply.isPending}>
      <SelectTrigger size="sm" aria-label={t("UI preset")} className={cn("min-w-24", className)}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent alignItemWithTrigger={false} align="start">
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function CheckpointPicker({ className }: { className?: string }) {
  const { t } = useTranslation()
  const models = useQuery(modelsQuery)
  const current = useClientOption("sd_model_checkpoint")
  const write = useOptionWrite<string>(MODEL_OPTION_KEYS.checkpoint)
  const loading = useModelBusy()
  const options = (models.data?.checkpoints ?? []).map((m) => ({
    value: m.title ?? m.name,
    label: checkpointLabel(m.name),
    hint: [m.arch ? presetLabel(m.arch) : null, formatBytes(m.size)].filter(Boolean).join(" · ") || undefined,
  }))
  const busy = write.isPending || loading !== null
  return (
    <SearchSelect
      aria-label={t("Checkpoint")}
      size="sm"
      className={className}
      options={options}
      value={write.isPending ? (write.variables ?? null) : current}
      placeholder={models.isPending ? t("Loading…") : t("Select a checkpoint")}
      onValueChange={(name) => name !== current && write.mutate(name)}
      disabled={write.isPending}
      prefix={
        busy ? (
          <Loader2 className="size-3.5 shrink-0 animate-spin text-primary" aria-hidden />
        ) : (
          <Box className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
        )
      }
    />
  )
}

function ModulesPicker({ className }: { className?: string }) {
  const { t } = useTranslation()
  const id = useId()
  const models = useQuery(modelsQuery)
  // The setting holds paths; the server takes (and lists) file names.
  const current = useClientOption("forge_additional_modules").map(fileName)
  const [draft, setDraft] = useState<string[] | null>(null)
  /** The uncommitted selection (a ref too: the unmount cleanup must see the latest). */
  const pending = useRef<string[] | null>(null)
  const [query, setQuery] = useState("")
  const selected = draft ?? current
  const modules = (models.data?.modules ?? [...(models.data?.vae ?? []), ...(models.data?.text_encoders ?? [])]).map((m) => ({
    value: m.name,
    label: m.title ?? m.name,
    size: formatBytes(m.size),
  }))
  const q = query.trim().toLowerCase()
  const visible = q ? modules.filter((m) => m.label.toLowerCase().includes(q)) : modules
  // Selected files that are no longer listed stay visible so they can be removed.
  const missing = selected.filter((value) => !modules.some((m) => m.value === value))

  const edit = (next: string[]) => {
    pending.current = next
    setDraft(next)
  }
  const commit = () => {
    const next = pending.current
    pending.current = null
    setDraft(null)
    setQuery("")
    if (next) writeModules(next)
  }
  const toggle = (value: string, on: boolean) => {
    const base = draft ?? current
    edit(on ? [...base, value] : base.filter((v) => v !== value))
  }
  // Nested in the narrow header's popover, the picker can unmount without closing itself.
  useEffect(
    () => () => {
      const next = pending.current
      pending.current = null
      if (next) writeModules(next)
    },
    [],
  )

  return (
    <Popover onOpenChange={(open) => !open && commit()}>
      <PopoverTrigger
        render={
          <Button variant="outline" size="sm" className={cn("gap-1.5 font-normal", className)} aria-label={t("VAE / Text encoders")} />
        }
      >
        <Layers className="text-muted-foreground" aria-hidden />
        <span className="hidden 2xl:inline">{t("VAE / TE")}</span>
        <Badge variant={selected.length > 0 ? "default" : "secondary"} className="h-4 min-w-4 px-1 text-[0.65rem] tabular-nums">
          {selected.length}
        </Badge>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[min(26rem,calc(100vw-2rem))] gap-2 p-2">
        <div className="flex items-center justify-between gap-2 px-1">
          <p id={`${id}-title`} className="text-sm font-medium">
            {t("VAE / Text encoders")}
          </p>
          {selected.length > 0 ? (
            <Button variant="ghost" size="xs" onClick={() => edit([])}>
              {t("Clear")}
            </Button>
          ) : null}
        </div>
        <Input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t("Search…")} aria-label={t("Search modules")} className="h-8" />
        <ul role="group" aria-labelledby={`${id}-title`} className="flex max-h-72 flex-col gap-0.5 overflow-y-auto overscroll-contain">
          {[...missing.map((value) => ({ value, label: value, size: null, missing: true })), ...visible.map((m) => ({ ...m, missing: false }))].map(
            (m, index) => {
              const checkId = `${id}-module-${index}` // paths contain spaces and slashes
              return (
                <li key={m.value} className="flex items-center gap-2 rounded-md px-1.5 py-1 hover:bg-accent">
                  <Checkbox id={checkId} checked={selected.includes(m.value)} onCheckedChange={(on) => toggle(m.value, on)} />
                  <label htmlFor={checkId} className={cn("min-w-0 flex-1 truncate text-sm", m.missing && "text-muted-foreground line-through")}>
                    {m.label}
                  </label>
                  {m.size ? <span className="shrink-0 text-xs text-muted-foreground tabular-nums">{m.size}</span> : null}
                </li>
              )
            },
          )}
          {modules.length === 0 ? <li className="px-2 py-4 text-center text-sm text-muted-foreground">{t("No modules found")}</li> : null}
        </ul>
        <p className="px-1 text-xs text-muted-foreground">{t("Applied when this panel closes.")}</p>
      </PopoverContent>
    </Popover>
  )
}

function DtypeSelect({ className }: { className?: string }) {
  const { t } = useTranslation()
  const models = useQuery(modelsQuery)
  const current = useClientOption("forge_unet_storage_dtype")
  const write = useOptionWrite<string>(MODEL_OPTION_KEYS.dtype)
  const dtypes = models.data?.unet_dtypes ?? DEFAULT_DTYPES
  // "Automatic" is the one word among the dtype names
  const items = dtypes.map((d) => ({ value: d, label: d === "Automatic" ? t("Automatic") : d }))
  const value = write.isPending ? (write.variables ?? current) : current
  return (
    <Select<string> items={items} value={value} onValueChange={(next) => next && next !== current && write.mutate(next)} disabled={write.isPending}>
      <SelectTrigger size="sm" aria-label={t("Diffusion in Low Bits")} className={cn("max-w-48", className)}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent alignItemWithTrigger={false} align="end">
        {items.map((item) => (
          <SelectItem key={item.value} value={item.value}>
            {item.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function LoadingStatus({ className }: { className?: string }) {
  const { t } = useTranslation()
  const loading = useModelBusy()
  if (!loading) return null
  const percent = typeof loading.progress === "number" ? Math.round(loading.progress * 100) : null
  return (
    <span role="status" className={cn("max-w-40 truncate text-xs text-muted-foreground", className)}>
      {t(loading.stage)}
      {percent !== null ? ` ${percent}%` : ""}
    </span>
  )
}

/** The compact controls in one popover (narrow headers, phones). */
function ModelSettingsPopover({ className }: { className?: string }) {
  const { t } = useTranslation()
  return (
    <Popover>
      <PopoverTrigger render={<Button variant="ghost" size="icon-sm" className={className} aria-label={t("Model settings")} />}>
        <SlidersHorizontal />
      </PopoverTrigger>
      <PopoverContent align="end" className="w-[min(22rem,calc(100vw-2rem))] gap-3">
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">{t("UI preset")}</span>
          <PresetSelect className="w-full" />
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">{t("Diffusion in Low Bits")}</span>
          <DtypeSelect className="w-full max-w-none" />
        </div>
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">{t("VAE / Text encoders")}</span>
          <ModulesPicker className="w-full justify-start" />
        </div>
      </PopoverContent>
    </Popover>
  )
}

export default function ModelBar() {
  const { t } = useTranslation()
  return (
    <div role="group" aria-label={t("Model")} className="flex min-w-0 items-center gap-1.5">
      <PresetSelect className="hidden xl:flex" />
      <CheckpointPicker className="w-[min(22rem,36vw)]" />
      <ModulesPicker className="hidden xl:inline-flex" />
      <DtypeSelect className="hidden xl:flex" />
      <ModelSettingsPopover className="xl:hidden" />
      <LoadingStatus className="hidden xl:inline" />
    </div>
  )
}

/** Stacked variant for the generate page on phones (the header hides the bar there). */
export function ModelBarStacked() {
  const { t } = useTranslation()
  return (
    <div role="group" aria-label={t("Model")} className="flex flex-col gap-2">
      <CheckpointPicker className="w-full" />
      <div className="flex flex-wrap items-center gap-1.5">
        <PresetSelect />
        <ModulesPicker />
        <DtypeSelect className="min-w-0 flex-1" />
      </div>
      <LoadingStatus />
    </div>
  )
}

