import { Collapsible } from "@base-ui/react/collapsible"
import { ChevronRight } from "lucide-react"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { useForm, useFormState, useOwner } from "../hooks"
import { groupKey } from "../form-index"
import { ContainerTitleContext } from "../fallback-name"
import { RenderChildren } from "../Renderer"
import type { GroupNode } from "../schema"

/** Visual group; with a title it is a lightweight collapsible (no card). */
export function Group({ node }: { node: GroupNode }) {
  const { t } = useTranslation()
  const { store } = useForm()
  const owner = useOwner()
  const key = groupKey(owner, node)
  const open = useFormState((s) => s.open[key] ?? node.open ?? true)

  if (!node.title) {
    return (
      <div className="flex flex-col gap-3 rounded-lg border border-border/70 bg-muted/25 p-3">
        <RenderChildren nodes={node.children} />
      </div>
    )
  }

  return (
    <Collapsible.Root open={open} onOpenChange={(next) => store.getState().setOpen(key, next)} className="flex flex-col">
      <Collapsible.Trigger
        className={cn(
          "group/trigger -mx-1 flex items-center gap-1 rounded-md px-1 py-1 text-left text-xs font-semibold tracking-wide text-muted-foreground uppercase outline-none",
          "hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50",
        )}
      >
        <ChevronRight aria-hidden className="size-3.5 transition-transform duration-200 group-data-[panel-open]/trigger:rotate-90" />
        {t(node.title)}
      </Collapsible.Trigger>
      <Collapsible.Panel className="h-(--collapsible-panel-height) overflow-hidden transition-[height] duration-200 ease-out-soft data-ending-style:h-0 data-starting-style:h-0">
        <div className="flex flex-col gap-3 pt-2">
          <ContainerTitleContext value={t(node.title)}>
            <RenderChildren nodes={node.children} />
          </ContainerTitleContext>
        </div>
      </Collapsible.Panel>
    </Collapsible.Root>
  )
}
