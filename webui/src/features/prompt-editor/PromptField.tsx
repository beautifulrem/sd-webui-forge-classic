/**
 * A prompt input bound to a form param: label, modified dot, token counter,
 * page actions, the CodeMirror editor, a bracket-balance summary and image
 * drops (an image dropped anywhere on the field is handed to `onDropImage`).
 */
import { ImageDown } from "lucide-react"
import { type ReactNode, type Ref, Suspense, useId, useImperativeHandle, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { describedBy, infoTextId } from "@/ui/a11y"
import { FieldError, ModifiedDot } from "@/ui/Field"
import { useParam } from "@/ui/hooks"
import type { TextNode } from "@/ui/schema"
import { PromptDecorators } from "./PromptDecorators"
import { LazyPromptEditor, type PromptEditorHandle } from "./index"
import type { PromptKind } from "./facets"
import { TokenCounter } from "./TokenCounter"
import { bracketProblems } from "./tokenize"
import { useDebouncedValue } from "./use-debounced-value"

type PromptFieldProps = {
  node: TextNode
  kind?: PromptKind
  /** Sampling steps and styles for the token count. */
  steps?: number
  styles?: readonly string[]
  /** Extra controls at the end of the label row. */
  actions?: ReactNode
  onDropImage?: (file: File) => void
  editorRef?: Ref<PromptEditorHandle>
  minLines?: number
  className?: string
}

const LINE_REM = 1.25

function BracketSummary({ id, value }: { id: string; value: string }) {
  const { t } = useTranslation()
  // Only after typing pauses: an opening bracket is not an error yet.
  const settled = useDebouncedValue(value, 600)
  const problems = bracketProblems(settled)
  // A live region must exist before its message so screen readers announce it.
  return (
    <p
      id={id}
      role="status"
      aria-live="polite"
      className="text-xs leading-snug text-[color-mix(in_oklch,var(--warning)_65%,var(--foreground))] empty:hidden"
    >
      {problems.length > 0 ? t("Unbalanced brackets: {{count}}", { count: problems.length }) : null}
    </p>
  )
}

export function PromptField({ node, kind = "positive", steps, styles, actions, onDropImage, editorRef, minLines, className }: PromptFieldProps) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<string>(node)
  const value = typeof param.value === "string" ? param.value : ""
  const label = t(param.label)
  const placeholder = node.placeholder ? t(node.placeholder) : undefined
  const lines = minLines ?? Math.max(node.lines ?? 2, 2)
  const [dragging, setDragging] = useState(false)
  const depth = useRef(0)
  const bracketId = `${id}-brackets`
  const handle = useRef<PromptEditorHandle | null>(null)
  // The page's ref reaches the editor through this field's own handle.
  useImperativeHandle(
    editorRef,
    () => ({
      focus: () => handle.current?.focus(),
      insertText: (text: string) => handle.current?.insertText(text),
      view: () => handle.current?.view() ?? null,
    }),
    [],
  )
  const ariaDescribedBy = describedBy(id, param.info, param.error, bracketId)

  const hasImage = (event: React.DragEvent) => [...event.dataTransfer.items].some((item) => item.kind === "file" && item.type.startsWith("image/"))
  const dropImages = (files: File[]) => {
    const [file] = files
    if (file) onDropImage?.(file)
  }

  return (
    <div
      className={cn("group/field relative flex min-w-0 flex-col gap-1.5", className)}
      data-invalid={param.error ? true : undefined}
      onDragEnter={(event) => {
        if (!onDropImage || !hasImage(event)) return
        depth.current++
        setDragging(true)
      }}
      onDragLeave={() => {
        depth.current = Math.max(0, depth.current - 1)
        if (depth.current === 0) setDragging(false)
      }}
      onDragOver={(event) => {
        if (onDropImage && hasImage(event)) event.preventDefault()
      }}
      onDrop={(event) => {
        depth.current = 0
        setDragging(false)
        if (!onDropImage || event.defaultPrevented) return
        const files = [...event.dataTransfer.files].filter((file) => file.type.startsWith("image/"))
        if (files.length === 0) return
        event.preventDefault()
        dropImages(files)
      }}
    >
      <div className="flex min-h-6 items-center gap-1.5">
        <label id={`${id}-label`} htmlFor={id} className="truncate text-[0.8rem] leading-5 font-medium text-foreground/85 select-none">
          {label}
        </label>
        {param.modified ? <ModifiedDot onReset={param.reset} defaultValue={param.defaultValue} label={label} disabled={!param.enabled} /> : null}
        <div className="ml-auto flex shrink-0 items-center gap-1">
          {actions}
          <TokenCounter text={value} steps={steps} styles={styles} negative={kind === "negative"} />
        </div>
      </div>
      <PromptDecorators
        value={value}
        setValue={param.setValue}
        kind={kind}
        editor={() => handle.current}
        label={label || node.name}
        disabled={!param.enabled}
        fieldId={id}
        main
      >
        <Suspense fallback={<div aria-hidden className="rounded-lg border border-input bg-input/30" style={{ minHeight: `${lines * LINE_REM + 0.9}rem` }} />}>
          <LazyPromptEditor
            ref={handle}
            id={id}
            kind={kind}
            value={value}
            onChange={param.setValue}
            placeholder={placeholder}
            disabled={!param.enabled}
            minLines={lines}
            maxLines={node.max_lines ?? 16}
            ariaLabel={label || node.name}
            ariaDescribedBy={ariaDescribedBy}
            ariaInvalid={param.error ? true : undefined}
            onDropFiles={onDropImage ? dropImages : undefined}
          />
        </Suspense>
      </PromptDecorators>
      <BracketSummary id={bracketId} value={value} />
      {param.info ? (
        <p id={infoTextId(id)} className="text-xs leading-snug text-muted-foreground">
          {t(param.info)}
        </p>
      ) : null}
      {param.error ? <FieldError id={id} message={param.error} /> : null}
      {dragging ? (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center gap-2 rounded-lg border-2 border-dashed border-primary bg-background/80 text-sm font-medium text-primary backdrop-blur-[2px]"
        >
          <ImageDown className="size-4" />
          {t("Drop an image to read its parameters")}
        </div>
      ) : null}
    </div>
  )
}
