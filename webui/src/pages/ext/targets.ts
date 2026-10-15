/**
 * Extension pages as "Send to" targets: a page declares what it accepts
 * (`Page(send_to=[SendTarget("image", "JoyCaption")])`, schema-format.md §8
 * `send_to`); views that send images offer "Send to <label>" (`PageSendButtons`).
 */
import { useQueries, useQuery } from "@tanstack/react-query"
import { toast } from "sonner"
import { bootstrapQuery, pageSchemaQuery } from "@/api/queries"
import { t } from "@/i18n"
import type { PageSchema } from "@/ui/schema"
import { sendToPage } from "./deliver"

export type PageSendTarget = { kind: "image"; param: string; label: string }
export type PageTarget = { pageId: string; target: PageSendTarget }

/** The send targets a page schema declares (well-formed entries only). */
export function sendTargetsOf(schema: PageSchema | undefined): PageSendTarget[] {
  const raw = (schema as { send_to?: unknown } | undefined)?.send_to
  if (!Array.isArray(raw)) return []
  return raw.filter(
    (item): item is PageSendTarget =>
      typeof item === "object" && item !== null && item.kind === "image" && typeof item.param === "string" && typeof item.label === "string",
  )
}

/** Every page target accepting `kind`, in navigation order (schema pages only; their schemas are cached). */
export function usePageSendTargets(kind: PageSendTarget["kind"] = "image"): PageTarget[] {
  const bootstrap = useQuery(bootstrapQuery)
  const pages = (bootstrap.data?.pages ?? []).filter((page) => !page.widget)
  const schemas = useQueries({ queries: pages.map((page) => ({ ...pageSchemaQuery(page.id), retry: false })) })
  return pages.flatMap((page, index) =>
    sendTargetsOf(schemas[index]?.data)
      .filter((target) => target.kind === kind)
      .map((target) => ({ pageId: page.id, target })),
  )
}

/** Sends an image (upload or output file id) to a page target, with toasts. */
export function sendImageToPage({ pageId, target }: PageTarget, fileId: string): Promise<void> {
  const where = t(target.label)
  const id = toast.loading(t("Sending to {{target}}…", { target: where }))
  return sendToPage(pageId, { [target.param]: fileId })
    .then((count) => {
      if (count > 0) toast.success(t("Sent to {{target}}", { target: where }), { id })
      else toast.error(t("Could not send to {{target}}", { target: where }), { id })
    })
    .catch((error: unknown) => {
      toast.error(t("Could not send to {{target}}", { target: where }), { id, description: error instanceof Error ? error.message : String(error) })
    })
}
