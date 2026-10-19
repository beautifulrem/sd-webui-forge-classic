import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { Textarea } from "@/components/ui/textarea"
import { useParam } from "../hooks"
import { describedBy, hintRef } from "../a11y"
import { Field } from "../Field"
import type { InputNode } from "../schema"
import { useFallbackName } from "../fallback-name"

/** JSON escape hatch: edited as text, committed whenever it parses. */
export function JsonInput({ node }: { node: InputNode }) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<unknown>(node)
  const [draft, setDraft] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const info = param.info
  const label = t(param.label)
  const fallbackName = useFallbackName(node.name)
  const text = draft ?? JSON.stringify(param.value ?? null, null, 2)

  return (
    <Field
      id={id}
      label={label}
      info={info}
      hint={param.hint}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={() => {
        setDraft(null)
        setError(null)
        param.reset()
      }}
      disabled={!param.enabled}
    >
      <Textarea
        id={id}
        value={text}
        spellCheck={false}
        aria-label={label ? undefined : fallbackName}
        aria-invalid={error !== null || Boolean(param.error)}
        aria-describedby={describedBy(id, info, param.error, hintRef(id, param.hint), error ? `${id}-parse` : undefined)}
        onChange={(event) => {
          setDraft(event.target.value)
          try {
            param.setValue(JSON.parse(event.target.value))
            setError(null)
          } catch (e) {
            setError(e instanceof Error ? e.message : String(e))
          }
        }}
        onBlur={() => error === null && setDraft(null)}
        disabled={!param.enabled}
        className="min-h-20 font-mono text-xs"
      />
      {error ? (
        <p id={`${id}-parse`} role="alert" className="text-xs text-destructive">
          {t("Invalid JSON")}: {error}
        </p>
      ) : null}
    </Field>
  )
}
