/**
 * The form's `styles` multiselect plus "apply styles to the prompts"
 * (materialize: writes the selected styles into the prompts and clears the
 * selection) and the style editor.
 */
import { Pencil, Stamp } from "@/components/icons"
import { lazy, Suspense, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useForm, useFormState } from "@/ui/hooks"
import { RenderNode } from "@/ui/Renderer"
import type { MultiChoiceNode } from "@/ui/schema"

type StylesFieldProps = {
  node: MultiChoiceNode
  /** Param names of the prompts the styles apply to. */
  prompt?: string
  negative?: string
}

const StyleEditorDialog = lazy(() => import("./StyleEditorDialog").then((m) => ({ default: m.StyleEditorDialog })))

function ToolButton({ label, onClick, disabled, children }: { label: string; onClick: () => void; disabled?: boolean; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger render={<Button variant="outline" size="icon" aria-label={label} onClick={onClick} disabled={disabled} className="shrink-0" />}>
        {children}
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

export function StylesField({ node, prompt = "prompt", negative = "negative_prompt" }: StylesFieldProps) {
  const { t } = useTranslation()
  const { store } = useForm()
  const [editing, setEditing] = useState(false)
  const selected = useFormState((s) => s.values.core?.[node.name])
  const count = Array.isArray(selected) ? selected.length : 0

  const currentPrompts = () => {
    const core = store.getState().values.core ?? {}
    return { prompt: String(core[prompt] ?? ""), negative: String(core[negative] ?? "") }
  }

  const materialize = async () => {
    const core = store.getState().values.core ?? {}
    const names = Array.isArray(core[node.name]) ? (core[node.name] as string[]) : []
    if (names.length === 0) return
    const current = currentPrompts()
    try {
      // The server applies styles exactly as generation does.
      const next = await api.styles.apply({ prompt: current.prompt, negative_prompt: current.negative, styles: names })
      store.getState().setValues("core", { [prompt]: next.prompt, [negative]: next.negative_prompt, [node.name]: [] })
    } catch (error) {
      toast.error(t("Could not apply the styles"), { description: error instanceof Error ? error.message : String(error) })
    }
  }

  return (
    <div className="flex items-end gap-1.5">
      <div className="min-w-0 flex-1">
        <RenderNode node={node} />
      </div>
      <ToolButton label={t("Apply the selected styles to the prompts")} onClick={() => void materialize()} disabled={count === 0}>
        <Stamp />
      </ToolButton>
      <ToolButton label={t("Edit styles")} onClick={() => setEditing(true)}>
        <Pencil />
      </ToolButton>
      {editing ? (
        <Suspense fallback={null}>
          <StyleEditorDialog
            open={editing}
            onOpenChange={setEditing}
            currentPrompts={currentPrompts}
            initial={Array.isArray(selected) && typeof selected[0] === "string" ? selected[0] : null}
          />
        </Suspense>
      ) : null}
    </div>
  )
}
