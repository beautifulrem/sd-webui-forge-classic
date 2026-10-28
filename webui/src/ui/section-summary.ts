/**
 * Section header summaries contributed by custom widgets: a widget folder
 * with a `summary.ts` exporting `sectionSummary` sums up its values in the
 * header of the section holding it, where it shows while the section is
 * collapsed (e.g. ControlNet's "2x Unit"). Discovered at build time like the
 * widgets, but imported eagerly: keep those modules small.
 */
import type { InputNode, WidgetNode } from "./schema"

/** `(the owner's values, the section's inputs of the widget) → translated text, or null`. */
export type SectionSummary = (values: Record<string, unknown> | undefined, nodes: WidgetNode[]) => string | null

const modules = import.meta.glob<{ sectionSummary: SectionSummary }>("../widgets/*/summary.ts", { eager: true })

const summaries = new Map<string, SectionSummary>(
  Object.entries(modules).flatMap(([path, module]) => (typeof module.sectionSummary === "function" ? [[path.split("/").at(-2) ?? path, module.sectionSummary]] : [])),
)

/** The header text of a section whose inputs are `inputs` (null: none). */
export function summarizeSection(values: Record<string, unknown> | undefined, inputs: readonly InputNode[]): string | null {
  if (summaries.size === 0) return null
  const byWidget = new Map<string, WidgetNode[]>()
  for (const input of inputs) {
    if (input.type !== "custom" || !summaries.has(input.widget)) continue
    byWidget.set(input.widget, [...(byWidget.get(input.widget) ?? []), input])
  }
  const parts: string[] = []
  for (const [widget, nodes] of byWidget) {
    const text = summaries.get(widget)?.(values, nodes)
    if (text) parts.push(text)
  }
  return parts.length > 0 ? parts.join(", ") : null
}
