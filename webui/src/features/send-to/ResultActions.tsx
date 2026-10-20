/**
 * Actions on one result: download, save (to the Save folder, or its whole
 * generation as a zip), open, copy parameters, reuse seed, use its
 * parameters, Hires. fix it (txt2img's ✨), send to img2img / inpaint /
 * extras, remove from history.
 */
import { ArrowRightLeft, ClipboardPaste, Copy, Download, ExternalLink, FileArchive, ImagePlus, Paintbrush, Recycle, Save, Sparkles, Trash2, Wand } from "lucide-react"
import type { ReactNode } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { clientOption } from "@/api/app-options"
import type { FormTab } from "@/api/types"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { copyText, downloadFile, openInNewTab } from "@/features/gallery/files"
import { type ResultItem, useSession } from "@/features/gallery/session-store"
import { upscaleResult } from "@/features/progress/run-controller"
import { applyParameters, reuseSeed, saveResults, sendWithFeedback } from "./result-actions"

function Action(props: { label: string; shortcut?: string; onClick: () => void; disabled?: boolean; children: ReactNode; tone?: "light" }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={props.label}
            aria-keyshortcuts={props.shortcut}
            onClick={props.onClick}
            disabled={props.disabled}
            className={cn(props.tone === "light" && "text-white/85 hover:bg-white/12 hover:text-white")}
          />
        }
      >
        {props.children}
      </TooltipTrigger>
      <TooltipContent>{props.label}</TooltipContent>
    </Tooltip>
  )
}

type ResultActionsProps = {
  item: ResultItem | null
  /** The form "reuse seed" and "use parameters" write to. */
  tab: FormTab
  /** Light icons over dark backdrops (lightbox). */
  tone?: "light"
  /** Hide "remove from history" (e.g. inside the lightbox). */
  removable?: boolean
  /** Offer the ✨ button (Hires. fix the result with the txt2img form's values) for txt2img results. */
  upscalable?: boolean
  /** After "Send to" switched pages (the lightbox closes itself). */
  onSent?: () => void
  className?: string
}

export function ResultActions({ item, tab, tone, removable = true, upscalable = false, onSent, className }: ResultActionsProps) {
  const { t } = useTranslation()
  const disabled = !item
  const infotext = item?.infotext ?? null
  const send = (...args: Parameters<typeof sendWithFeedback>) => {
    sendWithFeedback(...args)
    onSent?.()
  }
  // An img2img result sends its image only, unless `allow_i2i_send_info` (A1111's behaviour).
  const sendImage = (target: "img2img" | "inpaint" | "extras", result: ResultItem) =>
    send(target, tab === "img2img" && !clientOption("allow_i2i_send_info") ? { fileId: result.fileId, width: result.width, height: result.height } : result)

  return (
    <div role="toolbar" aria-label={t("Image actions")} className={cn("flex items-center gap-0.5", className)}>
      <Action label={t("Download")} shortcut="S" tone={tone} disabled={disabled} onClick={() => item && downloadFile(item.fileId)}>
        <Download />
      </Action>
      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={t("Save")}
              disabled={disabled}
              className={cn(tone === "light" && "text-white/85 hover:bg-white/12 hover:text-white")}
            />
          }
        >
          <Save />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-52">
          <DropdownMenuItem onClick={() => item && void saveResults([item.fileId])}>
            <Save />
            {t("Save to the Save folder")}
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => {
              if (!item) return
              const ids = useSession.getState().items.filter((other) => other.jobId === item.jobId).map((other) => other.fileId)
              void saveResults(ids, true)
            }}
          >
            <FileArchive />
            {t("Save this generation as zip")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <Action label={t("Open in new tab")} tone={tone} disabled={disabled} onClick={() => item && openInNewTab(item.fileId)}>
        <ExternalLink />
      </Action>
      <Action
        label={t("Copy generation parameters")}
        tone={tone}
        disabled={!infotext}
        onClick={() => {
          if (infotext) void copyText(infotext).then((ok) => (ok ? toast.success(t("Copied")) : toast.error(t("Could not copy"))))
        }}
      >
        <Copy />
      </Action>
      <Action
        label={t("Reuse seed")}
        tone={tone}
        disabled={!infotext}
        onClick={() => {
          if (reuseSeed(tab, infotext)) toast.success(t("Seed reused"))
          else toast.error(t("This image has no seed to reuse"))
        }}
      >
        <Recycle />
      </Action>
      <Action
        label={t("Use these parameters")}
        tone={tone}
        disabled={!infotext || tab === "extras"}
        onClick={() => infotext && tab !== "extras" && applyParameters(tab, infotext)}
      >
        <ClipboardPaste />
      </Action>
      {upscalable && tab === "txt2img" ? (
        <Action
          label={t("Upscale with Hires. fix")}
          tone={tone}
          // The grid, extra images (detected maps) and video frames are not first passes.
          disabled={!item || item.tab !== "txt2img" || Boolean(item.role) || Boolean(item.videoId)}
          onClick={() => item && upscaleResult({ fileId: item.fileId, key: item.key })}
        >
          <Sparkles />
        </Action>
      ) : null}
      <DropdownMenu>
        <DropdownMenuTrigger
          render={
            <Button
              variant="ghost"
              size="sm"
              disabled={disabled}
              className={cn("gap-1 px-2", tone === "light" && "text-white/85 hover:bg-white/12 hover:text-white")}
            />
          }
        >
          <ArrowRightLeft />
          <span>{t("Send to")}</span>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-44">
          <DropdownMenuItem onClick={() => item && sendImage("img2img", item)}>
            <ImagePlus />
            {t("img2img")}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => item && sendImage("inpaint", item)}>
            <Paintbrush />
            {t("Inpaint")}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => item && sendImage("extras", item)}>
            <Wand />
            {t("Extras")}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem disabled={!infotext} onClick={() => infotext && send("txt2img", { infotext })}>
            <ClipboardPaste />
            {t("txt2img (parameters)")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      {removable ? (
        <Action label={t("Remove from history")} tone={tone} disabled={disabled} onClick={() => item && useSession.getState().remove(item.key)}>
          <Trash2 />
        </Action>
      ) : null}
    </div>
  )
}
