/**
 * `text-file`: a text area that can also load a text file. The file is read
 * in the browser (nothing is uploaded) and replaces the text, each line
 * stripped. The value stays a plain string.
 *
 * props: { accept?: string, lines?: number, max_lines?: number, placeholder?: string }   value: string
 */
import { FileText } from "@/components/icons"
import { useRef } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { showToast } from "@/lib/toast"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { fileLines } from "./lines"

const NS = registerWidgetCatalogs("text-file", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

/** Matches the generic text input's line height (rem). */
const LINE_HEIGHT_REM = 1.25

type Props = { accept?: string; lines?: number; max_lines?: number; placeholder?: string }

export default function TextFile({ value, onChange, props, disabled, id, label }: WidgetProps<string | null, Props>) {
  const { t: tw } = useTranslation(NS)
  const picker = useRef<HTMLInputElement>(null)
  const lines = Math.max(1, props.lines ?? 2)

  const load = async (file: File | undefined) => {
    if (!file) return
    try {
      onChange(fileLines(await file.text()))
    } catch (error) {
      showToast("error", tw("Could not read the file"), error instanceof Error ? error.message : String(error))
    }
  }

  return (
    <div className="flex flex-col gap-1.5">
      <Textarea
        id={id}
        value={typeof value === "string" ? value : ""}
        onChange={(event) => onChange(event.target.value)}
        placeholder={props.placeholder}
        disabled={disabled}
        spellCheck={false}
        aria-label={label ? undefined : tw("Text")}
        className="resize-y"
        style={{
          minHeight: `${lines * LINE_HEIGHT_REM + 1}rem`,
          maxHeight: props.max_lines ? `${props.max_lines * LINE_HEIGHT_REM + 1}rem` : undefined,
        }}
      />
      <input
        ref={picker}
        type="file"
        accept={props.accept}
        hidden
        data-testid="text-file-picker"
        onChange={(event) => {
          void load(event.target.files?.[0])
          event.target.value = ""
        }}
      />
      <Button type="button" variant="outline" size="sm" className="self-start" disabled={disabled} onClick={() => picker.current?.click()}>
        <FileText aria-hidden />
        {tw("Load from a text file")}
      </Button>
    </div>
  )
}
