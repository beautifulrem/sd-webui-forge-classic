/**
 * `pp-send`: Post Processing Suite's "Apply PP → Send to img2img". Runs the
 * suite's action `props.action` on this form's result selected on the stage
 * (the session's image results of this tab, as the old button used the
 * txt2img gallery; the newest when the selection is another tab's), which
 * answers the processed image's upload id as the output `sent`; the image
 * then goes to `props.target`'s source image like any "Send to".
 *
 * props: { action: string, target?: "img2img" | "inpaint" }   value: unused (null)
 */
import { ImageUp, Loader2 } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import type { ActionExtra, FormTab } from "@/api/types"
import { Button } from "@/components/ui/button"
import { useSession } from "@/features/gallery/session-store"
import { deliver, formOf, imageEntry, type SendTarget } from "@/features/send-to/send-to"
import { useForm, useRunAction } from "@/ui/hooks"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { sourceItem } from "./source"

const NS = registerWidgetCatalogs("pp-send", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Props = { action?: string; target?: "img2img" | "inpaint" }

export default function PpSend({ props, disabled, id }: WidgetProps<null, Props>) {
  const { t: tw } = useTranslation(NS)
  const { formId } = useForm()
  const run = useRunAction()
  const item = useSession((s) => sourceItem(s, formId as FormTab))
  const [busy, setBusy] = useState(false)
  const target: SendTarget = props.target ?? "img2img"
  const inpaint = target === "inpaint"

  const send = async () => {
    if (!item) return
    setBusy(true)
    const progress = toast.loading(tw("Applying post-processing…"))
    try {
      // A script action's own argument (ActionExtra types the core actions' only)
      const answer = await run(props.action ?? "apply_send", { file_id: item.fileId } as unknown as ActionExtra)
      const sent = answer?.outputs?.sent
      if (typeof sent !== "string") {
        toast.dismiss(progress)
        return
      }
      const entry = imageEntry(target, sent)
      const delivered = entry ? await deliver(formOf(target), [entry]) : 0
      if (delivered > 0) toast.success(inpaint ? tw("Post-processed image sent to inpaint") : tw("Post-processed image sent to img2img"), { id: progress })
      else toast.error(tw("Could not send the post-processed image"), { id: progress })
    } catch (error) {
      toast.error(tw("Could not send the post-processed image"), { id: progress, description: error instanceof Error ? error.message : String(error) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex items-center gap-2">
      {item ? (
        <img
          src={api.thumbUrl(item.fileId, 64)}
          alt={tw("Selected result")}
          className="size-9 shrink-0 rounded-md border bg-canvas object-cover"
          loading="lazy"
        />
      ) : null}
      <Button id={id} variant="default" disabled={disabled || busy || !item} aria-busy={busy || undefined} onClick={() => void send()}>
        {busy ? <Loader2 className="animate-spin" aria-hidden /> : <ImageUp aria-hidden />}
        {inpaint ? tw("Apply PP → Send to inpaint") : tw("Apply PP → Send to img2img")}
      </Button>
      {item ? null : <span className="text-xs text-muted-foreground">{tw("Generate an image first")}</span>}
    </div>
  )
}
