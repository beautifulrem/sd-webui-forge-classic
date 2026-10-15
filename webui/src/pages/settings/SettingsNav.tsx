/** Settings section navigation: categories → sections (match counts, unsaved dots), then the system views. */
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import type { FilteredSection } from "./settings-model"
import { type SettingsView, SYSTEM_VIEWS } from "./views"

type Props = {
  sections: FilteredSection[]
  view: SettingsView
  activeSection: string | null
  filtering: boolean
  /** section id → unsaved edits in it */
  unsaved: ReadonlyMap<string, number>
  onSection: (id: string) => void
  onView: (view: SettingsView) => void
}

const itemClass = (active: boolean) =>
  cn(
    "flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1.5 text-left text-[0.82rem] outline-none transition-colors",
    "text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-2 focus-visible:ring-ring/60",
    active && "bg-sidebar-accent font-medium text-sidebar-accent-foreground",
  )

export function SettingsNav({ sections, view, activeSection, filtering, unsaved, onSection, onView }: Props) {
  const { t } = useTranslation()
  const categories: { id: string; label: string; sections: FilteredSection[] }[] = []
  for (const section of sections) {
    const last = categories.at(-1)
    if (last && last.id === section.categoryId) last.sections.push(section)
    else categories.push({ id: section.categoryId, label: section.categoryLabel, sections: [section] })
  }
  return (
    <nav aria-label={t("Settings sections")} className="flex flex-col gap-4 p-3">
      {categories.map((category) => (
        <div key={category.id} className="flex flex-col gap-0.5">
          <p className="px-2 pb-1 text-[0.7rem] font-semibold tracking-wider text-muted-foreground uppercase">{t(category.label)}</p>
          <ul className="flex flex-col gap-px">
            {category.sections.map((section) => {
              const active = view === "settings" && activeSection === section.id
              const edits = unsaved.get(section.id) ?? 0
              return (
                <li key={section.id}>
                  <button type="button" className={itemClass(active)} aria-current={active ? "location" : undefined} onClick={() => onSection(section.id)}>
                    <span className="min-w-0 flex-1 truncate">{t(section.label)}</span>
                    {edits > 0 ? (
                      <>
                        <span aria-hidden className="size-1.5 shrink-0 rounded-full bg-primary" />
                        <span className="sr-only">{t("{{count}} unsaved", { count: edits })}</span>
                      </>
                    ) : null}
                    {filtering ? <span className="shrink-0 text-xs tabular-nums text-muted-foreground">{section.count}</span> : null}
                  </button>
                </li>
              )
            })}
          </ul>
        </div>
      ))}
      {categories.length === 0 ? <p className="px-2 text-sm text-muted-foreground">{t("No matching sections")}</p> : null}
      <div className="flex flex-col gap-0.5 border-t pt-3">
        <p className="px-2 pb-1 text-[0.7rem] font-semibold tracking-wider text-muted-foreground uppercase">{t("System")}</p>
        <ul className="flex flex-col gap-px">
          {SYSTEM_VIEWS.map(({ id, label, icon: Icon }) => (
            <li key={id}>
              <button type="button" className={itemClass(view === id)} aria-current={view === id ? "page" : undefined} onClick={() => onView(id)}>
                <Icon className="size-3.5 shrink-0" aria-hidden />
                <span className="truncate">{t(label)}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  )
}
