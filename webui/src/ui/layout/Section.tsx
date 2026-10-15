import { Collapsible } from "@base-ui/react/collapsible"
import { ChevronRight } from "lucide-react"
import { useId } from "react"
import { useTranslation } from "react-i18next"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import { resetParams } from "../actions"
import { useForm, useFormState, useOwner, useParam } from "../hooks"
import { ModifiedDot } from "../Field"
import { inputsIn, sectionKey } from "../form-index"
import { anyModified } from "../form-store"
import { RenderChildren } from "../Renderer"
import type { SectionNode, ToggleNode } from "../schema"

function EnableSwitch({ node, title, onEnabled }: { node: ToggleNode; title: string; onEnabled: () => void }) {
  const { t } = useTranslation()
  const param = useParam<boolean>(node)
  return (
    <Switch
      checked={Boolean(param.value)}
      onCheckedChange={(checked) => {
        param.setValue(checked)
        if (checked) onEnabled()
      }}
      disabled={!param.enabled}
      aria-label={t("Enable {{title}}", { title })}
      aria-invalid={param.error ? true : undefined}
      title={param.error}
    />
  )
}

/**
 * Collapsible panel. With `enable`, the header carries the owner's enable
 * toggle (the former InputAccordion); switching it on also opens the panel.
 * The header's reset restores every param of the section and reruns their
 * `on_change` actions and conflict locks.
 */
export function Section({ node }: { node: SectionNode }) {
  const { t } = useTranslation()
  const id = useId()
  const { store, formId } = useForm()
  const owner = useOwner()
  const key = sectionKey(owner, node)
  const open = useFormState((s) => s.open[key] ?? node.open ?? false)
  const names = inputsIn(node).map((input) => input.name)
  const modified = useFormState((s) => anyModified(s, owner, names))
  const enabled = useFormState((s) => (node.enable ? Boolean(s.values[owner]?.[node.enable.name]) : false))
  const title = t(node.title)
  const setOpen = (next: boolean) => store.getState().setOpen(key, next)

  return (
    <Collapsible.Root
      open={open}
      onOpenChange={setOpen}
      render={<section aria-labelledby={`${id}-title`} />}
      data-enabled={enabled || undefined}
      className={cn(
        "rounded-xl border bg-card text-card-foreground shadow-xs transition-colors",
        "data-enabled:border-primary/35 data-enabled:shadow-[inset_3px_0_0_var(--primary)]",
      )}
    >
      <div className="flex min-h-10 items-center gap-2 pr-3 pl-1.5">
        <Collapsible.Trigger
          className={cn(
            "group/trigger flex min-w-0 flex-1 items-center gap-1.5 self-stretch rounded-lg px-1.5 text-left text-sm font-medium outline-none",
            "focus-visible:ring-3 focus-visible:ring-ring/50",
          )}
        >
          <ChevronRight
            aria-hidden
            className="size-4 shrink-0 text-muted-foreground transition-transform duration-200 group-data-[panel-open]/trigger:rotate-90"
          />
          <span id={`${id}-title`} className={cn("truncate", enabled && "text-foreground")}>
            {title}
          </span>
        </Collapsible.Trigger>
        {modified ? (
          <ModifiedDot
            label={title}
            onReset={() => resetParams({ store, tab: formId }, owner, names)}
          />
        ) : null}
        {node.enable ? <EnableSwitch node={node.enable} title={title} onEnabled={() => setOpen(true)} /> : null}
      </div>
      <Collapsible.Panel
        className={cn(
          "h-(--collapsible-panel-height) overflow-hidden transition-[height] duration-200 ease-out-soft",
          "data-ending-style:h-0 data-starting-style:h-0",
        )}
      >
        <div className="flex flex-col gap-3 px-3 pt-1 pb-3">
          <RenderChildren nodes={node.children} />
        </div>
      </Collapsible.Panel>
    </Collapsible.Root>
  )
}
