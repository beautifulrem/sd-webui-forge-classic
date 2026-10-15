/**
 * Chrome strings of a widget, kept with the widget: `widgets/<name>/locales/<locale>.json`
 * (English text → translation) in an i18next namespace of their own, so a
 * widget never edits the shared `src/locales` catalogs. Not a widget itself
 * (no `index.tsx`).
 *
 *   const NS = registerWidgetCatalogs("xyz-values", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))
 *   const { t: tw } = useTranslation(NS)   // tw("…"): the widget catalog, then the server catalog, then the key
 *
 * Widget sources call the translator `tw`, and each widget's tests check its
 * keys against its own catalog (`widgetKeys`).
 */
import { i18n } from "@/i18n"

/** Adds the catalogs (`./locales/<locale>.json` modules) under `widget:<name>`; returns that namespace. */
export function registerWidgetCatalogs(name: string, files: Record<string, Record<string, string>>): string {
  const ns = `widget:${name}`
  for (const [path, catalog] of Object.entries(files)) {
    const locale = path.split("/").at(-1)?.replace(/\.json$/, "")
    if (locale) i18n.addResourceBundle(locale, ns, catalog, true, true)
  }
  return ns
}

/** The literal keys a widget source passes to `tw("…")` (tests). */
export function widgetKeys(source: string): string[] {
  return [...source.matchAll(/\btw\(\s*"((?:[^"\\]|\\.)+)"/g)].map((m) => m[1]!)
}
