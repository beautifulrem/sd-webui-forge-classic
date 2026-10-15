/**
 * "Download localization template" (the former Settings → Actions button):
 * every English string the UI shows, as `{"English": "translation"}` JSON for
 * a new `localizations/<locale>.json`. Keys come from the SPA's own catalog,
 * the server's catalog (schema labels, extension strings) and the settings
 * schema; values are the active language's translations, empty when there
 * is none (or the UI is in English).
 */
import { api } from "@/api/endpoints"
import { DEFAULT_LOCALE, i18n } from "@/i18n"
import type { Node, SettingsSchema } from "@/ui/schema"
import { htmlText } from "./settings-model"

/** The locale whose catalogs list the most strings (their keys are the English texts). */
const REFERENCE_LOCALE = "zh-CN"

const chromeCatalogs = import.meta.glob<Record<string, string>>("../../locales/*.json", { import: "default" })

/** Labels, help, notes, links and choice labels of a settings schema. */
export function settingsStrings(schema: SettingsSchema): string[] {
  const out: string[] = []
  const add = (text: unknown) => {
    if (typeof text === "string" && text.trim()) out.push(text)
  }
  const visit = (node: Node) => {
    const item = node as Node & { label?: unknown; info?: unknown; notes?: unknown[]; links?: { label: string }[]; choices?: { label: string }[]; children?: Node[]; text?: unknown }
    add(item.label)
    add(item.info)
    item.notes?.forEach(add)
    item.links?.forEach((link) => add(link.label))
    item.choices?.forEach((choice) => add(choice.label))
    if (node.type === "html" && typeof item.text === "string") add(htmlText(item.text))
    item.children?.forEach(visit)
  }
  for (const category of schema.categories) {
    add(category.label)
    for (const section of category.sections) {
      add(section.label)
      for (const node of section.items as Node[]) visit(node)
    }
  }
  return out
}

/** `{key: translation or ""}` for every key, sorted. */
export function buildTemplate(keys: Iterable<string>, translate: (key: string) => string | undefined): Record<string, string> {
  const unique = [...new Set(keys)].filter((key) => key.trim()).sort((a, b) => a.localeCompare(b, "en"))
  return Object.fromEntries(unique.map((key) => [key, translate(key) ?? ""]))
}

async function referenceKeys(): Promise<string[]> {
  const loader = chromeCatalogs[`../../locales/${REFERENCE_LOCALE}.json`]
  const [chrome, server] = await Promise.all([loader ? loader() : Promise.resolve({}), api.i18n(REFERENCE_LOCALE.replace("-", "_")).catch(() => ({}))])
  return [...Object.keys(chrome), ...Object.keys(server)]
}

/** Builds the template for the active language and downloads it. */
export async function downloadLocalizationTemplate(schema: SettingsSchema | null): Promise<number> {
  const language = i18n.language
  const keys = [...(await referenceKeys()), ...(schema ? settingsStrings(schema) : [])]
  const translate = (key: string): string | undefined => {
    if (language === DEFAULT_LOCALE) return undefined
    const value = i18n.getResource(language, "chrome", key) ?? i18n.getResource(language, "server", key)
    return typeof value === "string" ? value : undefined
  }
  const template = buildTemplate(keys, translate)
  const blob = new Blob([`${JSON.stringify(template, null, 2)}\n`], { type: "application/json" })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = `localization-${language === DEFAULT_LOCALE ? "template" : language}.json`
  document.body.append(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
  return Object.keys(template).length
}
