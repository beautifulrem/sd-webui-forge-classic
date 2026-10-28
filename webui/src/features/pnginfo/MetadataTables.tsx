/** PNG Info tables: parsed parameters, every metadata entry by group, and the PNG chunk list. */
import { Check, Copy } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import type { ImageMetadata, MetadataEntry } from "@/api/types"
import { Button } from "@/components/ui/button"
import { copyText } from "@/features/gallery/files"
import { cn } from "@/lib/utils"
import { infotextKeyLabel } from "@/features/infotext/display"
import { groupEntries } from "./group"

function CopyValue({ value, label }: { value: string; label: string }) {
  const { t } = useTranslation()
  const [copied, setCopied] = useState(false)
  return (
    <Button
      variant="ghost"
      size="icon-xs"
      aria-label={t("Copy {{name}}", { name: label })}
      className="opacity-0 group-hover/row:opacity-100 focus-visible:opacity-100"
      onClick={() =>
        void copyText(value).then((ok) => {
          if (!ok) return
          setCopied(true)
          setTimeout(() => setCopied(false), 1200)
        })
      }
    >
      {copied ? <Check className="text-success" /> : <Copy />}
    </Button>
  )
}

/** `label`: what the key column shows (a translated infotext key); the raw key is its tooltip. */
function Rows({ rows, caption }: { rows: { key: string; label?: string; value: string; kind?: string }[]; caption: string }) {
  const { t } = useTranslation()
  if (rows.length === 0) return <p className="px-1 py-3 text-sm text-muted-foreground">{t("Nothing found in the image.")}</p>
  return (
    <table className="w-full table-fixed border-collapse text-sm">
      <caption className="sr-only">{caption}</caption>
      <colgroup>
        <col className="w-[34%] sm:w-48" />
        <col />
        <col className="w-8" />
      </colgroup>
      <tbody>
        {rows.map((row, index) => (
          <tr key={`${row.key}-${index}`} className="group/row border-b border-border/60 last:border-0 hover:bg-muted/40">
            <th scope="row" className="py-1.5 pr-3 text-left align-top font-medium break-words text-muted-foreground" title={row.label && row.label !== row.key ? row.key : undefined}>
              {row.label ?? row.key}
            </th>
            <td className={cn("py-1.5 align-top break-words whitespace-pre-wrap select-text", row.kind === "binary" && "font-mono text-xs text-muted-foreground", row.kind === "number" && "tabular-nums")}>
              {row.value.length > 4000 ? `${row.value.slice(0, 4000)}…` : row.value}
            </td>
            <td className="py-1 align-top">{row.kind === "binary" ? null : <CopyValue value={row.value} label={row.label ?? row.key} />}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function ParametersTable({ params }: { params: ImageMetadata["params"] }) {
  const { t } = useTranslation()
  const rows = Object.entries(params).map(([key, value]) => ({ key, label: infotextKeyLabel(t, key), value }))
  return <Rows rows={rows} caption={t("Generation parameters")} />
}

export function EntriesTable({ entries }: { entries: readonly MetadataEntry[] }) {
  const { t } = useTranslation()
  const groups = groupEntries(entries)
  return (
    <div className="flex flex-col gap-4">
      {groups.map(([group, rows]) => (
        <section key={group} aria-label={t(group)}>
          <h3 className="mb-1 text-[0.8rem] font-medium text-muted-foreground">{t(group)}</h3>
          <Rows rows={rows} caption={t(group)} />
        </section>
      ))}
    </div>
  )
}

export function ChunksTable({ chunks }: { chunks: ImageMetadata["chunks"] }) {
  const { t, i18n } = useTranslation()
  if (chunks.length === 0) return <p className="px-1 py-3 text-sm text-muted-foreground">{t("Not a PNG file: it has no chunks.")}</p>
  return (
    <table className="w-full border-collapse text-sm">
      <caption className="sr-only">{t("PNG chunks")}</caption>
      <thead>
        <tr className="border-b text-left text-xs text-muted-foreground">
          <th scope="col" className="py-1.5 pr-3 font-medium">
            {t("Chunk")}
          </th>
          <th scope="col" className="py-1.5 pr-3 font-medium">
            {t("Keyword")}
          </th>
          <th scope="col" className="py-1.5 text-right font-medium">
            {t("Bytes")}
          </th>
        </tr>
      </thead>
      <tbody>
        {chunks.map((chunk, index) => (
          <tr key={index} className="border-b border-border/60 last:border-0">
            <td className="py-1.5 pr-3 font-mono">
              {chunk.type}
              {chunk.count > 1 ? <span className="ml-1 text-xs text-muted-foreground">× {chunk.count}</span> : null}
            </td>
            <td className="py-1.5 pr-3">{chunk.key ?? "—"}</td>
            <td className="py-1.5 text-right tabular-nums">{chunk.length.toLocaleString(i18n.language)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
