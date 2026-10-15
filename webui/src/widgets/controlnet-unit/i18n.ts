/** The widget's chrome catalog (`./locales/<locale>.json`), namespace `widget:controlnet-unit`. */
import { registerWidgetCatalogs } from "../_lib/i18n"

export const NS = registerWidgetCatalogs("controlnet-unit", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))
