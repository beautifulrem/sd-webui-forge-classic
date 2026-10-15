/**
 * An item's metadata: file facts, training facts read from the kohya /
 * modelspec header, the dataset tags as weighted chips (click copies one,
 * "Copy all" copies them as a prompt) and the raw header as a filterable
 * table.
 */
import { useQuery } from "@tanstack/react-query"
import { ChevronDown, Copy, Search } from "lucide-react"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import type { NetworkItem, NetworkMeta } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { t as translate } from "@/i18n"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { formatBytes } from "@/features/model-bar/model-options"
import { useArchLabel } from "./labels"
import { Preview } from "./NetworkCard"
import { metaQuery } from "./queries"
import { TagChips } from "./TagChips"
import { escapeTag } from "./tags"

async function copy(text: string, what: string) {
  try {
    await navigator.clipboard.writeText(text)
    toast.success(what)
  } catch {
    toast.error(translate("Could not copy to the clipboard"))
  }
}

type Training = Record<string, unknown>

function useTrainingRows(training: Training) {
  const { t, i18n } = useTranslation()
  const text = (value: unknown) => (value === undefined || value === null || value === "" ? null : String(value))
  const resolutions = Array.isArray(training.resolutions) ? (training.resolutions as { resolution: string; count: number }[]) : []
  const trained = typeof training.trained_at === "string" ? new Date(training.trained_at) : null
  const rows: [string, string | null][] = [
    [t("Output name"), text(training.output_name)],
    [t("Title"), text(training.title)],
    [t("Author"), text(training.author)],
    [t("Network type"), text(training.network_type)],
    [t("Module"), text(training.module)],
    [t("Base model"), text(training.base_model)],
    [t("Base model version"), text(training.base_model_version)],
    [t("Architecture"), text(training.architecture)],
    [t("Dimension / alpha"), training.dim !== undefined ? `${String(training.dim)} / ${text(training.alpha) ?? "–"}` : null],
    [t("Clip skip"), text(training.clip_skip)],
    [t("Epochs"), text(training.epochs)],
    [t("Steps"), text(training.steps)],
    [t("Learning rate"), text(training.learning_rate)],
    [t("Optimizer"), text(training.optimizer)],
    [t("Training images"), text(training.dataset_size ?? training.train_images)],
    [t("Trained"), trained && !Number.isNaN(trained.getTime()) ? trained.toLocaleString(i18n.language) : null],
    [
      resolutions.length > 1 ? t("Resolutions") : t("Resolution"),
      resolutions.length ? resolutions.slice(0, 6).map((r) => `${r.resolution} (${r.count})`).join(", ") + (resolutions.length > 6 ? ", …" : "") : null,
    ],
    [t("License"), text(training.license)],
  ]
  return rows.filter((row): row is [string, string] => row[1] !== null)
}

function Facts({ rows }: { rows: [string, string][] }) {
  return (
    <dl className="grid grid-cols-[minmax(7rem,auto)_1fr] gap-x-4 gap-y-1.5 text-[0.8rem]">
      {rows.map(([label, value]) => (
        <div key={label} className="contents">
          <dt className="text-muted-foreground">{label}</dt>
          <dd className="min-w-0 [overflow-wrap:anywhere]">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

function RawMetadata({ metadata }: { metadata: Record<string, unknown> }) {
  const { t } = useTranslation()
  const [filter, setFilter] = useState("")
  const entries = Object.entries(metadata).filter(([key, value]) => {
    if (!filter.trim()) return true
    const needle = filter.toLowerCase()
    return key.toLowerCase().includes(needle) || (typeof value === "string" && value.toLowerCase().includes(needle))
  })
  const render = (value: unknown) => (typeof value === "string" ? value : JSON.stringify(value, null, 2))
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder={t("Filter keys and values")} aria-label={t("Filter keys and values")} className="h-7 pl-7 text-xs" />
        </div>
        <Button variant="outline" size="sm" onClick={() => void copy(JSON.stringify(metadata, null, 2), t("Metadata copied"))}>
          <Copy aria-hidden />
          {t("Copy JSON")}
        </Button>
      </div>
      <div className="max-h-80 overflow-auto rounded-lg border">
        <table className="w-full text-xs">
          <tbody>
            {entries.map(([key, value]) => (
              <tr key={key} className="border-b align-top last:border-0">
                <th scope="row" className="w-1/3 bg-muted/40 px-2 py-1.5 text-left font-mono font-normal text-muted-foreground [overflow-wrap:anywhere]">
                  {key}
                </th>
                <td className="px-2 py-1.5">
                  <pre className="max-h-48 overflow-auto font-mono whitespace-pre-wrap [overflow-wrap:anywhere]">{render(value)}</pre>
                </td>
              </tr>
            ))}
            {entries.length === 0 ? (
              <tr>
                <td className="px-2 py-3 text-center text-muted-foreground">{t("Nothing matches")}</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function Body({ meta }: { meta: NetworkMeta }) {
  const { t, i18n } = useTranslation()
  const archLabel = useArchLabel()
  const rows = useTrainingRows(meta.training as Training)
  const tags = meta.dataset_tags
  const keys = Object.keys(meta.metadata)
  const fileRows: [string, string][] = [
    [t("Path"), meta.path],
    ...(meta.alias && meta.alias !== meta.name ? ([[t("Alias"), meta.alias]] as [string, string][]) : []),
    ...(meta.hash ? ([[t("Hash"), meta.hash]] as [string, string][]) : []),
    ...(meta.size ? ([[t("Size"), formatBytes(meta.size) ?? ""]] as [string, string][]) : []),
    ...(meta.mtime ? ([[t("Modified"), new Date(meta.mtime * 1000).toLocaleString(i18n.language)]] as [string, string][]) : []),
  ]
  return (
    <div className="flex flex-col gap-5">
      <div className="flex gap-4">
        <Preview item={meta} width={160} rounded="rounded-lg" className="h-32 w-24 shrink-0 ring-1 ring-border" />
        <div className="flex min-w-0 flex-1 flex-col gap-2">
          <div className="flex flex-wrap gap-1">
            {meta.arch ? <Badge variant="secondary">{archLabel(meta.arch)}</Badge> : null}
            {meta.subtype ? <Badge variant="outline">{meta.subtype}</Badge> : null}
            {meta.tags.map((tag) => (
              <Badge key={tag} variant="outline">
                {tag}
              </Badge>
            ))}
          </div>
          <Facts rows={fileRows} />
        </div>
      </div>
      {meta.description ? <p className="text-[0.8rem] whitespace-pre-wrap text-foreground/85">{meta.description}</p> : null}
      {rows.length ? (
        <section aria-labelledby="network-meta-training" className="flex flex-col gap-2">
          <h3 id="network-meta-training" className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
            {t("Training")}
          </h3>
          <Facts rows={rows} />
        </section>
      ) : null}
      {tags.length ? (
        <section aria-labelledby="network-meta-tags" className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <h3 id="network-meta-tags" className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
              {t("Training tags")}
            </h3>
            <span className="text-xs text-muted-foreground tabular-nums">{tags.length}</span>
            <Button variant="ghost" size="xs" className="ml-auto" onClick={() => void copy(tags.slice(0, 40).map((tag) => escapeTag(tag.tag)).join(", "), t("Tags copied"))}>
              <Copy aria-hidden />
              {t("Copy top 40")}
            </Button>
            <Button variant="ghost" size="xs" onClick={() => void copy(tags.map((tag) => escapeTag(tag.tag)).join(", "), t("Tags copied"))}>
              <Copy aria-hidden />
              {t("Copy all")}
            </Button>
          </div>
          <TagChips tags={tags} pickLabel={t("Click a tag to copy it")} onPick={(tag) => void copy(escapeTag(tag), t("Copied {{tag}}", { tag }))} />
        </section>
      ) : null}
      {keys.length ? (
        <Collapsible>
          <CollapsibleTrigger render={<Button variant="ghost" size="sm" className="group/raw -ml-2" />}>
            <ChevronDown className="transition-transform group-data-panel-open/raw:rotate-180" aria-hidden />
            {t("Raw metadata ({{count}} keys)", { count: keys.length })}
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-2">
            <RawMetadata metadata={meta.metadata} />
          </CollapsibleContent>
        </Collapsible>
      ) : (
        <p className="text-xs text-muted-foreground">{t("The file has no embedded metadata.")}</p>
      )}
    </div>
  )
}

export function MetadataDialog({ item, open, onOpenChange }: { item: NetworkItem | null; open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t } = useTranslation()
  const meta = useQuery({ ...metaQuery(item?.kind ?? "", item?.name ?? ""), enabled: Boolean(item && open) })
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[min(88dvh,52rem)] flex-col gap-4 sm:max-w-3xl">
        <DialogHeader className="pr-8">
          <DialogTitle className="truncate">{item?.title ?? ""}</DialogTitle>
          <DialogDescription>{t("File information, training details and dataset tags")}</DialogDescription>
        </DialogHeader>
        <div className="-mx-4 min-h-0 flex-1 overflow-y-auto px-4 pb-1">
          {meta.data ? (
            <Body meta={meta.data} />
          ) : meta.isError ? (
            <p className="text-sm text-destructive">{meta.error.message}</p>
          ) : (
            <div className="flex flex-col gap-3" aria-busy="true">
              <Skeleton className="h-32 w-full" />
              <Skeleton className="h-24 w-full" />
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
