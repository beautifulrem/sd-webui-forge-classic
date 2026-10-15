import { lazy, Suspense, useId, useRef } from "react"
import { useTranslation } from "react-i18next"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { PromptDecorators } from "@/features/prompt-editor/PromptDecorators"
import type { PromptEditorHandle } from "@/features/prompt-editor/PromptEditor"
import { useParam } from "../hooks"
import { describedBy } from "../a11y"
import { Field } from "../Field"
import type { TextNode } from "../schema"

// CodeMirror is a separate chunk, fetched when the first prompt field mounts.
const PromptEditor = lazy(() => import("../prompt/PromptEditor"))

const LINE_HEIGHT_REM = 1.25

export function Text({ node }: { node: TextNode }) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<string>(node)
  const value = typeof param.value === "string" ? param.value : ""
  const info = param.info
  const placeholder = node.placeholder ? t(node.placeholder) : undefined
  const lines = Math.max(1, node.lines ?? 1)
  const label = t(param.label)
  const ariaDescribedBy = describedBy(id, info, param.error)
  const invalid = param.error ? true : undefined
  const handle = useRef<PromptEditorHandle | null>(null)

  let control
  if (node.prompt) {
    const minHeight = `${Math.max(lines, 2) * LINE_HEIGHT_REM + 1}rem`
    const kind = /negative/i.test(node.name) ? ("negative" as const) : ("positive" as const)
    control = (
      <PromptDecorators
        value={value}
        setValue={param.setValue}
        kind={kind}
        editor={() => handle.current}
        label={label || node.name}
        disabled={!param.enabled}
        fieldId={id}
      >
        <Suspense
          fallback={<div aria-hidden className="rounded-lg border border-input bg-input/30" style={{ minHeight }} />}
        >
          <PromptEditor
            ref={handle}
            kind={kind}
            id={id}
            value={value}
            onChange={param.setValue}
            placeholder={placeholder}
            disabled={!param.enabled}
            minLines={lines}
            maxLines={node.max_lines ?? undefined}
            ariaLabel={label || node.name}
            ariaDescribedBy={ariaDescribedBy}
            ariaInvalid={invalid}
          />
        </Suspense>
      </PromptDecorators>
    )
  } else if (lines === 1) {
    control = (
      <Input
        id={id}
        value={value}
        placeholder={placeholder}
        onChange={(event) => param.setValue(event.target.value)}
        disabled={!param.enabled}
        aria-label={label ? undefined : node.name}
        aria-describedby={ariaDescribedBy}
        aria-invalid={invalid}
        spellCheck={false}
      />
    )
  } else {
    control = (
      <Textarea
        id={id}
        value={value}
        placeholder={placeholder}
        onChange={(event) => param.setValue(event.target.value)}
        disabled={!param.enabled}
        aria-label={label ? undefined : node.name}
        aria-describedby={ariaDescribedBy}
        aria-invalid={invalid}
        spellCheck={false}
        className="resize-y"
        style={{
          minHeight: `${lines * LINE_HEIGHT_REM + 1}rem`,
          maxHeight: node.max_lines ? `${node.max_lines * LINE_HEIGHT_REM + 1}rem` : undefined,
        }}
      />
    )
  }

  return (
    <Field
      id={id}
      label={label}
      info={info}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={param.reset}
      disabled={!param.enabled}
      labelFor={!node.prompt}
    >
      {control}
    </Field>
  )
}
