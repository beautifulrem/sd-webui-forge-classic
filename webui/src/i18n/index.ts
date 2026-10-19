/**
 * i18next with natural-language keys: every key is the English text itself.
 *
 * Two namespaces per locale:
 *  - `chrome`: the SPA's own strings, bundled from src/locales/<locale>.json
 *    (lazy, one chunk per locale);
 *  - `server`: the merged backend catalog from GET /api/v2/i18n/{locale},
 *    which translates schema labels, info texts, choices and extension strings.
 * `t()` looks in `chrome` first and falls back to `server`, then to the key.
 */
import i18next from "i18next"
import { initReactI18next } from "react-i18next"
import { create } from "zustand"
import { persist } from "zustand/middleware"
import { api } from "@/api/endpoints"

export const DEFAULT_LOCALE = "en"

const chromeCatalogs = import.meta.glob<Record<string, string>>("../locales/*.json", { import: "default" })

/** "zh_CN" (server/file style) → "zh-CN" (BCP 47, i18next and <html lang>). */
export function toBcp47(locale: string): string {
  return locale.replace(/_/g, "-")
}

/** "zh-CN" → "zh_CN", the form used by the server's localizations/ directory. */
export function toServerLocale(locale: string): string {
  return locale.replace(/-/g, "_")
}

export type TextDirection = "ltr" | "rtl"

/** Languages written right to left, for engines without `Intl.Locale` text info. */
const RTL_LANGUAGES = new Set(["ar", "arc", "ckb", "dv", "fa", "he", "iw", "ks", "ps", "sd", "syr", "ug", "ur", "yi"])

/**
 * The writing direction of a locale ("ar", "he-IL", "fa_IR" → rtl), from
 * CLDR through `Intl.Locale` (its script, so "pa-Arab" is rtl and "pa" is not). This
 * replaces the Gradio UI's `"rtl": true` flag in a localization file: a
 * catalog is named after its language, which already says the direction.
 */
export function textDirection(locale: string): TextDirection {
  try {
    const info = new Intl.Locale(toBcp47(locale)) as Intl.Locale & { getTextInfo?: () => { direction?: string }; textInfo?: { direction?: string } }
    const direction = info.getTextInfo?.().direction ?? info.textInfo?.direction
    if (direction === "rtl" || direction === "ltr") return direction
    return RTL_LANGUAGES.has(info.language) ? "rtl" : "ltr"
  } catch {
    return "ltr"
  }
}

type LocaleState = {
  /** The user's explicit choice; null follows the server default, then the browser. */
  chosen: string | null
  /** The locale currently active in i18next. */
  locale: string
  /** Its writing direction (also `<html dir>`). */
  direction: TextDirection
  setLocale: (locale: string) => void
}

export const useLocale = create<LocaleState>()(
  persist(
    (set) => ({
      chosen: null,
      locale: DEFAULT_LOCALE,
      direction: "ltr",
      setLocale: (locale) => set({ chosen: locale }),
    }),
    { name: "remi-locale", version: 1, partialize: ({ chosen }) => ({ chosen }) },
  ),
)

/** The locale to use before the server has told us its default. */
export function initialLocale(): string {
  return useLocale.getState().chosen ?? browserLocale()
}

export function browserLocale(): string {
  const nav = typeof navigator === "undefined" ? "" : navigator.language
  return nav.toLowerCase().startsWith("zh") ? "zh-CN" : DEFAULT_LOCALE
}

export const i18n = i18next.createInstance()

void i18n.use(initReactI18next).init({
  lng: DEFAULT_LOCALE,
  fallbackLng: false,
  ns: ["chrome", "server"],
  defaultNS: "chrome",
  fallbackNS: "server",
  keySeparator: false,
  nsSeparator: false,
  returnEmptyString: false,
  interpolation: { escapeValue: false },
  resources: {},
  react: { bindI18nStore: "added" },
})

const loaded = new Set<string>()

async function loadChrome(locale: string): Promise<Record<string, string>> {
  const loader = chromeCatalogs[`../locales/${locale}.json`]
  return loader ? loader() : {}
}

async function loadServer(locale: string): Promise<Record<string, string>> {
  try {
    return await api.i18n(toServerLocale(locale))
  } catch (error) {
    console.warn(`[i18n] server catalog for ${locale} unavailable`, error)
    return {}
  }
}

/** Loads both catalogs for `locale` (once) and switches to it. */
export async function activateLocale(locale: string): Promise<void> {
  const lng = toBcp47(locale)
  if (lng !== DEFAULT_LOCALE && !loaded.has(lng)) {
    const [chrome, server] = await Promise.all([loadChrome(lng), loadServer(lng)])
    i18n.addResourceBundle(lng, "chrome", chrome, true, true)
    i18n.addResourceBundle(lng, "server", server, true, true)
    loaded.add(lng)
  }
  await i18n.changeLanguage(lng)
  const direction = textDirection(lng)
  document.documentElement.lang = lng
  document.documentElement.dir = direction
  useLocale.setState({ locale: lng, direction })
}

/** Re-fetches the server catalog (e.g. after extensions changed). */
export async function reloadServerCatalog(): Promise<void> {
  const lng = i18n.language
  if (lng === DEFAULT_LOCALE) return
  i18n.addResourceBundle(lng, "server", await loadServer(lng), true, true)
}

/** Translate outside React (toasts, command labels built in callbacks). */
export const t = i18n.t.bind(i18n)
