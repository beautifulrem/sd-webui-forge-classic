import { createContext, type ReactNode, use } from "react"
import { useTranslation } from "react-i18next"
import { evaluate } from "../conditions"
import { OwnerProvider } from "../context"
import { useFormState } from "../hooks"
import { scriptsForSlot } from "../form-index"
import { RenderLayout } from "../Renderer"
import type { Node, ScriptSchema } from "../schema"

/** Scripts of the tab being rendered; core-form `slot` nodes place them. */
const ScriptsContext = createContext<readonly ScriptSchema[]>([])

export function ScriptsProvider({ scripts, children }: { scripts: readonly ScriptSchema[]; children: ReactNode }) {
  return <ScriptsContext value={scripts}>{children}</ScriptsContext>
}

function ScriptView({ script }: { script: ScriptSchema }) {
  const { t } = useTranslation()
  const visible = useFormState((s) => evaluate(script.visible_if, script.id, (o, n) => s.values[o]?.[n]))
  if (!visible || !script.layout) return null
  const layout = script.layout
  // A section or titled group names the script itself; anything else (e.g. a
  // legacy script's untitled group of "#i" inputs) is framed with the title.
  const panel = layout.type === "section" || (layout.type === "group" && Boolean(layout.title))
  // The frame replaces an untitled group's box (keeping its conditions).
  const body: Node = layout.type === "group" && !panel ? { ...layout, type: "column" } : layout
  return (
    <OwnerProvider owner={script.id}>
      <div data-script={script.id} className="min-w-0">
        {panel ? (
          <RenderLayout layout={layout} />
        ) : (
          // A bare layout gets a light frame so scripts stay distinguishable.
          <div role="group" aria-label={t(script.title)} className="flex flex-col gap-3 rounded-xl border bg-card p-3">
            <div className="text-sm font-medium">{t(script.title)}</div>
            <RenderLayout layout={body} />
          </div>
        )}
      </div>
    </OwnerProvider>
  )
}

export function Slot({ section }: { section: string }) {
  const scripts = use(ScriptsContext)
  const slots = useFormState((s) => s.index.slots)
  const placed = scriptsForSlot(scripts, section, slots)
  if (placed.length === 0) return null
  return (
    <div className="flex flex-col gap-3" data-slot-section={section}>
      {placed.map((script) => (
        <ScriptView key={script.id} script={script} />
      ))}
    </div>
  )
}
