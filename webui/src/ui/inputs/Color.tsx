import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { Input } from "@/components/ui/input"
import { useParam } from "../hooks"
import { describedBy, hintRef } from "../a11y"
import { Field } from "../Field"
import type { ColorNode } from "../schema"
import { isHexColor } from "../values"
import { useFallbackName } from "../fallback-name"

/** #rgb → #rrggbb; strips alpha (the native picker has none). */
function toPickerValue(color: string): string {
  if (/^#[0-9a-f]{3}$/i.test(color)) return `#${[...color.slice(1)].map((c) => c + c).join("")}`
  return color.slice(0, 7)
}

/** Native colour picker swatch plus a hex field (accepts #rgb, #rrggbb, #rrggbbaa). */
export function Color({ node }: { node: ColorNode }) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<string>(node)
  const value = isHexColor(param.value) ? param.value : "#000000"
  const [draft, setDraft] = useState<string | null>(null)
  const info = param.info
  const disabled = !param.enabled
  const label = t(param.label)
  const fallbackName = useFallbackName(node.name)

  return (
    <Field
      id={id}
      label={label}
      info={info}
      hint={param.hint}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={param.reset}
      disabled={disabled}
    >
      <div className="flex items-center gap-1.5">
        <div className="relative size-8 shrink-0 cursor-pointer overflow-hidden rounded-lg border border-input bg-checker focus-within:ring-3 focus-within:ring-ring/50">
          <span aria-hidden className="absolute inset-0" style={{ backgroundColor: value }} />
          <input
            type="color"
            value={toPickerValue(value)}
            onChange={(event) => {
              // Keep an existing alpha channel when picking a new colour.
              const alpha = value.length === 9 ? value.slice(7) : ""
              param.setValue(event.target.value + alpha)
            }}
            disabled={disabled}
            aria-label={label ? t("Pick a color for {{label}}", { label }) : t("Pick a color")}
            className="absolute inset-0 size-full cursor-pointer opacity-0"
          />
        </div>
        <Input
          id={id}
          value={draft ?? value}
          spellCheck={false}
          aria-label={label ? undefined : fallbackName}
          aria-describedby={describedBy(id, info, param.error, hintRef(id, param.hint))}
          aria-invalid={(draft !== null && !isHexColor(draft)) || Boolean(param.error)}
          onChange={(event) => {
            const next = event.target.value.trim()
            setDraft(next)
            if (isHexColor(next)) param.setValue(next.toLowerCase())
          }}
          onBlur={() => setDraft(null)}
          disabled={disabled}
          className="font-mono text-xs uppercase"
        />
      </div>
    </Field>
  )
}
