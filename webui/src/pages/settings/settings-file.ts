/** Settings export as a downloaded JSON file. */
import { exportSettings } from "./settings-model"
import { useSettingsEditor } from "./settings-store"

/** Downloads the stored settings (only those differing from their defaults unless `all`). */
export function downloadSettings(all: boolean) {
  const { saved, schema } = useSettingsEditor.getState()
  const data = exportSettings(saved, schema?.defaults, { all })
  const blob = new Blob([`${JSON.stringify(data, null, 2)}\n`], { type: "application/json" })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = `remi-settings-${new Date().toISOString().slice(0, 10)}.json`
  document.body.append(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
