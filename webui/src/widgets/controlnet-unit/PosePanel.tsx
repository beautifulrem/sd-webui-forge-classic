/**
 * The OpenPose editor: the preview's pose as JSON (edit, upload, download);
 * "Apply" draws it on the server and makes it the unit's input, as the old
 * editor's render did.
 */
import { Download, Play, Upload } from "lucide-react"
import { useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { errorTextId } from "@/ui/a11y"
import { FieldError } from "@/ui/Field"
import { NS } from "./i18n"
import { parsePose, poseText } from "./pose"

type PosePanelProps = {
  pose: unknown
  busy: boolean
  disabled?: boolean
  onApply: (pose: unknown) => void
}

export function PosePanel({ pose, busy, disabled, onApply }: PosePanelProps) {
  const { t: tw } = useTranslation(NS)
  const id = useId()
  const initial = poseText(pose)
  const [text, setText] = useState(initial)
  const [error, setError] = useState<string | null>(null)
  const file = useRef<HTMLInputElement>(null)

  const apply = (source: string) => {
    const parsed = parsePose(source)
    if (!parsed.ok) {
      setError(tw("Not a valid pose: {{error}}", { error: parsed.error }))
      return
    }
    setError(null)
    onApply(parsed.pose)
  }

  const upload = async (files: FileList | null) => {
    const picked = files?.[0]
    if (file.current) file.current.value = ""
    if (!picked) return
    const source = await picked.text()
    setText(source)
    apply(source)
  }

  const download = () => {
    const url = URL.createObjectURL(new Blob([text], { type: "application/json" }))
    const link = document.createElement("a")
    link.href = url
    link.download = "pose.json"
    document.body.append(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
  }

  return (
    <div role="group" aria-labelledby={`${id}-title`} className="flex flex-col gap-2 rounded-lg border p-2.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <span id={`${id}-title`} className="mr-auto text-[0.8rem] font-medium">
          {tw("Pose (OpenPose JSON)")}
        </span>
        <Button size="xs" variant="ghost" onClick={() => file.current?.click()} disabled={disabled || busy}>
          <Upload aria-hidden />
          {tw("Upload JSON")}
        </Button>
        <Button size="xs" variant="ghost" onClick={download} disabled={!text.trim()}>
          <Download aria-hidden />
          {tw("Download JSON")}
        </Button>
        <Button size="xs" variant="secondary" onClick={() => apply(text)} disabled={disabled || busy || !text.trim()}>
          <Play aria-hidden />
          {tw("Apply pose")}
        </Button>
        <input ref={file} type="file" accept=".json,application/json" hidden onChange={(event) => void upload(event.target.files)} />
      </div>
      <Textarea
        id={id}
        aria-label={tw("Pose (OpenPose JSON)")}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? errorTextId(id) : undefined}
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder={tw("Run an openpose preprocessor, or upload a pose JSON")}
        spellCheck={false}
        disabled={disabled}
        className="max-h-56 min-h-24 font-mono text-xs"
      />
      {error ? <FieldError id={id} message={error} /> : null}
    </div>
  )
}
