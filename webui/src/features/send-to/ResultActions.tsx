/**
 * Actions on one result: download, save (to the Save folder, or its whole
 * generation as a zip), open, copy parameters, reuse seed, use its
 * parameters, Hires. fix it (txt2img's ✨), send to img2img / inpaint /
 * extras, remove from history.
 */
import { ArrowRightLeft, ClipboardPaste, Copy, Download, Ellipsis, ExternalLink, FileArchive, ImagePlus, Paintbrush, Recycle, Save, Sparkles, Trash2, Wand } from "@/components/icons"
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

function Action(props: { label: string; shortcut?: string; onClick: () => void; disabled?: boolean; children: ReactNode; tone?: "light"; className?: string }) {
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
            className={cn(props.tone === "light" && "text-white/85 hover:bg-white/12 hover:text-white", props.className)}
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
  /**
   * Narrow screens (below the `sm` breakpoint): the less used actions move
   * into a "More actions" menu and "Send to" shows its icon only, so the bar
   * fits beside the viewer's own buttons.
   */
  collapsible?: boolean
  className?: string
}

type SecondaryAction = { label: string; icon: ReactNode; disabled: boolean; run: () => void }

export function ResultActions({ item, tab, tone, removable = true, upscalable = false, onSent, collapsible = false, className }: ResultActionsProps) {
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

  const secondary: SecondaryAction[] = [
    { label: t("Open in new tab"), icon: <ExternalLink />, disabled, run: () => item && openInNewTab(item.fileId) },
    {
      label: t("Copy generation parameters"),
      icon: <Copy />,
      disabled: !infotext,
      run: () => {
        if (infotext) void copyText(infotext).then((ok) => (ok ? toast.success(t("Copied")) : toast.error(t("Could not copy"))))
      },
    },
    {
      label: t("Reuse seed"),
      icon: <Recycle />,
      disabled: !infotext,
      run: () => {
        if (reuseSeed(tab, infotext)) toast.success(t("Seed reused"))
        else toast.error(t("This image has no seed to reuse"))
      },
    },
    { label: t("Use these parameters"), icon: <ClipboardPaste />, disabled: !infotext || tab === "extras", run: () => infotext && tab !== "extras" && applyParameters(tab, infotext) },
  ]
  if (upscalable && tab === "txt2img") {
    secondary.push({
      label: t("Upscale with Hires. fix"),
      icon: <Sparkles />,
      // The grid, extra images (detected maps) and video frames are not first passes.
      disabled: !item || item.tab !== "txt2img" || Boolean(item.role) || Boolean(item.videoId),
      run: () => item && upscaleResult({ fileId: item.fileId, key: item.key }),
    })
  }

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
      {secondary.map((action) => (
        <Action key={action.label} label={action.label} tone={tone} disabled={action.disabled} onClick={action.run} className={cn(collapsible && "hidden sm:inline-flex")}>
          {action.icon}
        </Action>
      ))}
      {collapsible ? (
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={t("More actions")}
                disabled={disabled}
                className={cn("sm:hidden", tone === "light" && "text-white/85 hover:bg-white/12 hover:text-white")}
              />
            }
          >
            <Ellipsis />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="min-w-52">
            {secondary.map((action) => (
              <DropdownMenuItem key={action.label} disabled={action.disabled} onClick={action.run}>
                {action.icon}
                {action.label}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
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
          <span className={cn(collapsible && "sr-only sm:not-sr-only")}>{t("Send to")}</span>
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
