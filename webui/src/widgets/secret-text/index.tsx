/**
 * `secret-text`: a masked text field (Gradio's `Textbox(type="password")`)
 * with a show / hide toggle. Declare it `persist=False, submit=False` so the
 * secret never lands in drafts or jobs; a page action reads it and stores it
 * server side.
 *
 * props: { placeholder?: string (English, translated) }
 * value: string
 */
import { Eye, EyeOff } from "lucide-react"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"

const NS = registerWidgetCatalogs("secret-text", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Props = { placeholder?: string }

export default function SecretText({ value, onChange, props, disabled, id, label }: WidgetProps<string | null, Props>) {
  const { t: tw } = useTranslation(NS)
  const [shown, setShown] = useState(false)
  return (
    <InputGroup>
      <InputGroupInput
        id={id}
        type={shown ? "text" : "password"}
        value={typeof value === "string" ? value : ""}
        onChange={(event) => onChange(event.target.value)}
        placeholder={props.placeholder ? tw(props.placeholder) : undefined}
        disabled={disabled}
        autoComplete="off"
        spellCheck={false}
        aria-label={label || undefined}
      />
      <InputGroupAddon align="inline-end">
        <InputGroupButton
          size="icon-xs"
          aria-label={shown ? tw("Hide") : tw("Show")}
          aria-pressed={shown}
          disabled={disabled}
          onClick={() => setShown((current) => !current)}
        >
          {shown ? <EyeOff aria-hidden /> : <Eye aria-hidden />}
        </InputGroupButton>
      </InputGroupAddon>
    </InputGroup>
  )
}
