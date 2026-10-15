/**
 * Every setting in one virtualized scroll (category and section headings
 * between them); the nav scrolls to a section, and reports the section in
 * view back. Search terms are highlighted in the rendered text.
 */
import { useVirtualizer } from "@tanstack/react-virtual"
import { SearchX } from "lucide-react"
import { type Ref, useEffect, useImperativeHandle, useRef } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { cn } from "@/lib/utils"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { useSearchHighlight } from "./highlight"
import { itemKey, sectionOfRow, sectionRowIndex, type SettingsRowItem } from "./settings-model"
import { SETTINGS_FORM, SETTINGS_OWNER, settingsForm } from "./settings-store"
import { SettingItemView } from "./SettingRow"

export type SettingsListHandle = { scrollToSection: (id: string) => void; scrollToTop: () => void }

type Props = {
  rows: SettingsRowItem[]
  terms: string[]
  onSectionInView: (id: string | null) => void
  onClearFilters: () => void
  filtering: boolean
  ref?: Ref<SettingsListHandle>
}

function estimate(row: SettingsRowItem | undefined): number {
  if (!row) return 80
  if (row.kind === "category") return 64
  if (row.kind === "section") return 44
  if (row.item.kind === "row") return 120
  if (row.item.kind === "separator") return 20
  return 96
}

function RowView({ row, first }: { row: SettingsRowItem; first: boolean }) {
  const { t } = useTranslation()
  if (row.kind === "category") {
    return (
      <h2 className={cn("pb-1 font-heading text-xs font-semibold tracking-wider text-muted-foreground uppercase", first ? "pt-4" : "pt-8")}>
        {t(row.label)}
      </h2>
    )
  }
  if (row.kind === "section") {
    return (
      <h3 id={`settings-section-${row.id}`} className="flex items-baseline gap-2 border-b pt-3 pb-2 text-base font-semibold tracking-tight">
        {t(row.label)}
      </h3>
    )
  }
  return <SettingItemView item={row.item} />
}

export function SettingsList({ rows, terms, onSectionInView, onClearFilters, filtering, ref }: Props) {
  // TanStack Virtual's virtualizer is a mutable object the React Compiler must not memoise.
  "use no memo"
  const { t } = useTranslation()
  const scroller = useRef<HTMLDivElement>(null)
  const content = useRef<HTMLDivElement>(null)
  // oxlint-disable-next-line react/incompatible-library -- opted out of memoisation above
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scroller.current,
    estimateSize: (index) => estimate(rows[index]),
    overscan: 6,
    getItemKey: (index) => {
      const row = rows[index]
      if (!row) return index
      return row.kind === "item" ? `${row.sectionId}/${itemKey(row.item, index)}` : `${row.kind}:${row.id}`
    },
  })
  useSearchHighlight(content, terms)

  useImperativeHandle(ref, () => ({
    scrollToSection: (id) => {
      const index = sectionRowIndex(rows, id)
      if (index >= 0) virtualizer.scrollToIndex(index, { align: "start" })
    },
    scrollToTop: () => scroller.current?.scrollTo({ top: 0 }),
  }))

  const start = virtualizer.range?.startIndex ?? 0
  const inView = rows.length > 0 ? sectionOfRow(rows, start) : null
  useEffect(() => onSectionInView(inView), [inView, onSectionInView])

  if (rows.length === 0) {
    return (
      <div className="grid h-full place-items-center p-6">
        <Empty className="max-w-sm">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <SearchX aria-hidden />
            </EmptyMedia>
            <EmptyTitle>{t("No settings found")}</EmptyTitle>
            <EmptyDescription>{t("Try other words, or search for a setting's key.")}</EmptyDescription>
          </EmptyHeader>
          {filtering ? (
            <EmptyContent>
              <Button variant="outline" size="sm" onClick={onClearFilters}>
                {t("Clear filters")}
              </Button>
            </EmptyContent>
          ) : null}
        </Empty>
      </div>
    )
  }

  return (
    <div ref={scroller} className="h-full overflow-y-auto overscroll-contain [scrollbar-gutter:stable]" tabIndex={-1}>
      <FormProvider formId={SETTINGS_FORM} store={settingsForm}>
        <OwnerProvider owner={SETTINGS_OWNER}>
          <div ref={content} className="mx-auto w-full max-w-4xl px-4 pb-24 sm:px-6">
            <div className="relative" style={{ height: virtualizer.getTotalSize() }}>
              {virtualizer.getVirtualItems().map((item) => {
                const row = rows[item.index]
                if (!row) return null
                return (
                  <div
                    key={item.key}
                    ref={virtualizer.measureElement}
                    data-index={item.index}
                    className="absolute inset-x-0 top-0"
                    style={{ transform: `translateY(${item.start}px)` }}
                  >
                    <RowView row={row} first={item.index === 0} />
                  </div>
                )
              })}
            </div>
          </div>
        </OwnerProvider>
      </FormProvider>
    </div>
  )
}
