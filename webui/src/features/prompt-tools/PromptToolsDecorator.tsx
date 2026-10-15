/**
 * Wraps every prompt editor: the editor (always mounted, hidden in the tag
 * view so its undo history survives), the tag-chips view, and the prompt
 * tools bar under them. Small and in the shell; the bar and the chips view
 * are lazy chunks loaded once the server says the tools are on.
 */
import { lazy, Suspense } from "react"
import type { PromptDecoratorProps } from "@/features/prompt-editor/decorators"
import { useToolsConfig } from "./queries"
import { usePromptViews, viewKey } from "./view-store"

const PromptToolbar = lazy(() => import("./PromptToolbar"))
const ChipsView = lazy(() => import("./chips/ChipsView"))

export function PromptToolsDecorator(props: PromptDecoratorProps) {
  const config = useToolsConfig()
  const key = viewKey(props.kind, props.label)
  const stored = usePromptViews((state) => state.views[key])
  // the bar (and with it the tag view) on every prompt field, or on the main prompts only
  const on = Boolean(config?.tools.enabled) && (config?.tools.toolbar !== "Main prompts only" || Boolean(props.main))
  const chips = on && stored === "chips"

  return (
    <div data-slot="prompt-tools" className="flex min-w-0 flex-col gap-1">
      <div hidden={chips} className="min-w-0">
        {props.children}
      </div>
      {chips && config ? (
        <Suspense fallback={<div aria-hidden className="min-h-16 rounded-lg border border-input bg-input/30" />}>
          <ChipsView {...props} config={config} />
        </Suspense>
      ) : null}
      {on && config ? (
        <Suspense fallback={<div aria-hidden className="h-6" />}>
          <PromptToolbar {...props} config={config} viewKey={key} view={chips ? "chips" : "text"} />
        </Suspense>
      ) : null}
    </div>
  )
}
