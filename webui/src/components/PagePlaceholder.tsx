import type { LucideIcon } from "lucide-react"
import type { ReactNode } from "react"
import { useTranslation } from "react-i18next"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"

/** Empty state for pages whose full view lands in a later wave. */
export function PagePlaceholder(props: { icon: LucideIcon; title: string; description: string; children?: ReactNode }) {
  const { t } = useTranslation()
  const Icon = props.icon
  return (
    <div className="flex h-full items-center justify-center overflow-y-auto p-6">
      <Empty className="max-w-md border">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <Icon aria-hidden />
          </EmptyMedia>
          <EmptyTitle>{t(props.title)}</EmptyTitle>
          <EmptyDescription>{t(props.description)}</EmptyDescription>
        </EmptyHeader>
        {props.children ? <EmptyContent>{props.children}</EmptyContent> : null}
      </Empty>
    </div>
  )
}
