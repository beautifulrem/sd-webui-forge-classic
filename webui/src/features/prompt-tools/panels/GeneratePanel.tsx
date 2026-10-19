/**
 * "Write a prompt": describe the image (any language) and let the
 * OpenAI-compatible model set up for translation write the prompt, from
 * scratch or by changing the current one; the answer can be edited before it
 * replaces the prompt or is added to it (Prompt All-in-One's "ChatGPT prompt").
 */
import { Loader2, Sparkles } from "lucide-react"
import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { promptToolsApi } from "../api"
import { PanelHeader } from "./panel"
import { appendText, type PanelProps } from "./helpers"

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export default function GeneratePanel({ value, setValue, kind, disabled, close }: PanelProps) {
  const { t } = useTranslation()
  const id = useId()
  const [description, setDescription] = useState("")
  const [buildOn, setBuildOn] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const negative = kind === "negative"

  const run = async () => {
    if (!description.trim() || busy) return
    setBusy(true)
    setError(null)
    try {
      setResult(await promptToolsApi.generate({ description: description.trim(), prompt: buildOn ? value : "", negative }))
    } catch (failure) {
      setError(errorText(failure))
    } finally {
      setBusy(false)
    }
  }

  const use = (text: string) => {
    setValue(text)
    close()
  }

  return (
    <div className="flex min-h-0 flex-col">
      <PanelHeader title={negative ? t("Write a negative prompt") : t("Write a prompt")} />
      <form
        className="flex flex-col gap-2 p-3"
        onSubmit={(event) => {
          event.preventDefault()
          void run()
        }}
      >
        <Label htmlFor={`${id}-description`}>{t("Describe the image")}</Label>
        <Textarea
          id={`${id}-description`}
          value={description}
          rows={3}
          maxLength={4000}
          placeholder={t("in any language, e.g. a girl reading under a cherry tree at dusk")}
          onChange={(event) => setDescription(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
              event.preventDefault()
              void run()
            }
          }}
        />
        {value.trim() ? (
          <Label className="font-normal text-muted-foreground">
            <Checkbox checked={buildOn} onCheckedChange={(checked) => setBuildOn(checked === true)} />
            {t("Change the current prompt instead of writing a new one")}
          </Label>
        ) : null}
        <div className="flex items-center gap-2">
          <Button type="submit" size="sm" disabled={!description.trim() || busy}>
            {busy ? <Loader2 className="animate-spin" aria-hidden /> : <Sparkles aria-hidden />}
            {t("Write")}
          </Button>
          <p className="text-xs text-muted-foreground">{t("Sent to the OpenAI-compatible service set up for translation.")}</p>
        </div>
        {error ? (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        ) : null}
      </form>
      {result !== null ? (
        <div className="flex flex-col gap-2 border-t p-3">
          <Label htmlFor={`${id}-result`}>{t("Written prompt")}</Label>
          <Textarea id={`${id}-result`} value={result} rows={4} onChange={(event) => setResult(event.target.value)} />
          <div className="flex flex-wrap gap-2">
            <Button size="sm" disabled={disabled || !result.trim()} onClick={() => use(result.trim())}>
              {t("Replace the prompt")}
            </Button>
            <Button size="sm" variant="outline" disabled={disabled || !result.trim()} onClick={() => use(appendText(value, result))}>
              {t("Add to the prompt")}
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  )
}
