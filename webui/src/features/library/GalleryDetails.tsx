/** The viewer's side panel for a gallery image: file facts, star, parameters and library actions ("show in folder" only where folders may be opened). */
import { useQuery } from "@tanstack/react-query"
import { FileSearch, FolderOpen, Star, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { bootstrapQuery } from "@/api/queries"
import type { GalleryCategory, GalleryItem } from "@/api/types"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { InfotextPanel } from "@/features/infotext/InfotextPanel"
import { CATEGORY_LABELS, formatBytes } from "./labels"
import { openInPngInfo, revealOutput, starOutputs } from "./queries"

type GalleryDetailsProps = {
  item: GalleryItem
  onDelete: (item: GalleryItem) => void
  onClose: () => void
}

export function GalleryDetails({ item, onDelete, onClose }: GalleryDetailsProps) {
  const { t, i18n } = useTranslation()
  // "folders.open" is gone under --hide-ui-dir-config (the server answers 403 then).
  const canOpenFolders = useQuery({ ...bootstrapQuery, select: (data) => data.features.includes("folders.open") }).data ?? false
  const created = new Date(item.created_at)
  const facts: [string, string][] = [
    [t("File"), item.name],
    [t("Size"), `${item.width ?? "?"} × ${item.height ?? "?"} · ${formatBytes(item.size, i18n.language)}`],
    [t("Created"), Number.isNaN(created.getTime()) ? item.created_at : created.toLocaleString(i18n.language)],
    [t("From"), t(CATEGORY_LABELS[item.category as GalleryCategory] ?? item.category)],
  ]
  if (item.model) facts.push([t("Model"), item.model])
  const tone = "text-white/85 hover:bg-white/12 hover:text-white"
  return (
    <div className="flex flex-col gap-3 text-white/90">
      <div className="flex items-center gap-1">
        <Button variant="ghost" size="sm" className={tone} aria-pressed={item.starred} onClick={() => void starOutputs([item.id], !item.starred)}>
          <Star className={cn(item.starred && "fill-amber-400 text-amber-400")} aria-hidden />
          {item.starred ? t("Starred") : t("Star")}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className={tone}
          onClick={() => {
            onClose()
            openInPngInfo(item.id)
          }}
        >
          <FileSearch aria-hidden />
          {t("PNG Info")}
        </Button>
        {canOpenFolders ? (
          <Button variant="ghost" size="icon-sm" className={tone} aria-label={t("Show in folder")} title={t("Show in folder")} onClick={() => void revealOutput(item.id)}>
            <FolderOpen />
          </Button>
        ) : null}
        <Button variant="ghost" size="icon-sm" className={cn(tone, "ml-auto hover:text-red-300")} aria-label={t("Delete")} title={t("Delete")} onClick={() => onDelete(item)}>
          <Trash2 />
        </Button>
      </div>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        {facts.map(([key, value]) => (
          <div key={key} className="contents">
            <dt className="text-white/55">{key}</dt>
            <dd className="min-w-0 truncate select-text" title={value}>
              {value}
            </dd>
          </div>
        ))}
      </dl>
      <InfotextPanel infotext={item.infotext} className="[&_.bg-muted]:bg-white/10 [&_.text-muted-foreground]:text-white/55" />
    </div>
  )
}
