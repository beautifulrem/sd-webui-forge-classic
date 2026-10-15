import { Suspense, useId } from "react"
import { useTranslation } from "react-i18next"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { Skeleton } from "@/components/ui/skeleton"
import { useParam } from "../hooks"
import { Field } from "../Field"
import type { CustomNode, Json } from "../schema"
import { getWidget } from "../widgets"
import { JsonInput } from "./Json"

export function Custom({ node }: { node: CustomNode }) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<Json>(node)
  const Widget = getWidget(node.widget)
  const label = t(param.label)

  if (!Widget) {
    // Unknown widget: still editable, so the value is never lost.
    if (import.meta.env.DEV) console.warn(`[ui] no widget named "${node.widget}" in src/widgets`)
    return <JsonInput node={node} />
  }

  return (
    <Field
      id={id}
      label={label}
      info={param.info}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={param.reset}
      disabled={!param.enabled}
    >
      <ErrorBoundary inline>
        <Suspense fallback={<Skeleton className="h-16 w-full" />}>
          {/* oxlint-disable-next-line react/static-components -- a stable lazy component from the widget registry */}
          <Widget
            id={id}
            label={label}
            value={param.value}
            onChange={param.setValue}
            props={node.props ?? {}}
            node={node}
            disabled={!param.enabled}
          />
        </Suspense>
      </ErrorBoundary>
    </Field>
  )
}
