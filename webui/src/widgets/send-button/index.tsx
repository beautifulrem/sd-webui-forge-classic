/**
 * `send-button`: sends one of this owner's values somewhere else.
 *
 * props (`to` picks the kind; `source` names a param of the same owner):
 *   { to: "prompt", source, target?: "txt2img" | "img2img" (txt2img), mode?: "append" | "replace" (append),
 *     navigate?: boolean (true) }   — the text into that form's prompt
 *   { to: "page", source, page, param, action? }  — an image (upload id) into another extension page's
 *     `param` (`sendToPage`); `action` first runs this owner's action (e.g. a download filling `source`;
 *     nothing is sent when it answers an error or warning toast)
 *   { to: "url", source }           — opens the http(s) URL in a new tab
 *   all: label (English, translated), variant?: "primary" | "secondary" | "outline" | "ghost" (secondary)
 * value: unused (null)
 */
import { ExternalLink, Loader2, Send } from "lucide-react"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { useQuery } from "@tanstack/react-query"
import { bootstrapQuery } from "@/api/queries"
import { Button } from "@/components/ui/button"
import { sendToPage } from "@/pages/ext/deliver"
import { useForm, useOwner, useRunAction } from "@/ui/hooks"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { openUrl, sendToPrompt } from "./send"

const NS = registerWidgetCatalogs("send-button", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Common = { source: string; label?: string; variant?: "primary" | "secondary" | "outline" | "ghost" }
type Props = Common &
  (
    | { to: "prompt"; target?: "txt2img" | "img2img"; mode?: "append" | "replace"; navigate?: boolean }
    | { to: "page"; page: string; param: string; action?: string }
    | { to: "url" }
  )

const VARIANT = { primary: "default", secondary: "secondary", outline: "outline", ghost: "ghost" } as const

export default function SendButton({ props, disabled, id }: WidgetProps<null, Props>) {
  const { t: tw } = useTranslation(NS)
  const { store } = useForm()
  const owner = useOwner()
  const run = useRunAction()
  const pages = useQuery(bootstrapQuery).data?.pages
  const [busy, setBusy] = useState(false)
  const read = () => store.getState().values[owner]?.[props.source]
  const label = props.label ? tw(props.label) : tw("Send")

  // A page target that is not installed (its extension disabled): no button.
  if (props.to === "page" && pages && !pages.some((page) => page.id === props.page)) return null

  const send = async () => {
    if (props.to === "prompt") {
      const text = read()
      const target = props.target ?? "txt2img"
      if (typeof text !== "string" || !sendToPrompt(target, text, props.mode ?? "append", props.navigate ?? true)) {
        toast.info(tw("Nothing to send"))
      } else if (props.navigate === false) {
        toast.success(tw("Added to the {{target}} prompt", { target: tw(target) }))
      }
      return
    }
    if (props.to === "url") {
      const url = read()
      if (typeof url !== "string" || !openUrl(url)) toast.info(tw("No link to open"))
      return
    }
    setBusy(true)
    try {
      if (props.action) {
        // A failed action (an error or warning toast, already shown) sends nothing:
        // the source may still hold an earlier, unrelated image.
        const answer = await run(props.action)
        if (!answer || answer.toast?.level === "error" || answer.toast?.level === "warning") return
      }
      const image = read()
      if (typeof image !== "string" || !image) {
        toast.info(tw("No image to send"))
        return
      }
      const count = await sendToPage(props.page, { [props.param]: image })
      if (count === 0) toast.error(tw("Could not send the image"))
    } finally {
      setBusy(false)
    }
  }

  const Icon = busy ? Loader2 : props.to === "url" ? ExternalLink : Send
  return (
    <Button id={id} variant={VARIANT[props.variant ?? "secondary"]} disabled={disabled || busy} aria-busy={busy || undefined} onClick={() => void send()}>
      <Icon className={busy ? "animate-spin" : undefined} aria-hidden />
      {label}
    </Button>
  )
}
