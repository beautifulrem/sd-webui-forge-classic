/** The ControlNet section's header summary: how many units are enabled (the old "2x Unit" badge). */
import { i18n } from "@/i18n"
import type { WidgetNode } from "@/ui/schema"
import { NS } from "./i18n"

export function sectionSummary(values: Record<string, unknown> | undefined, nodes: WidgetNode[]): string | null {
  const count = nodes.filter((node) => (values?.[node.name] as { enabled?: unknown } | undefined)?.enabled === true).length
  return count > 0 ? i18n.t("{{count}}x Unit", { ns: NS, count }) : null
}
