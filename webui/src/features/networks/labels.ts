/** Display labels of the browser. */
import { useTranslation } from "react-i18next"
import { presetLabel } from "@/features/model-bar/model-options"

/** "SDXL", "Anima", … for an architecture id. */
export function useArchLabel() {
  const { t } = useTranslation()
  return (arch: string | null | undefined) => (!arch || arch === "unknown" ? t("Unknown") : presetLabel(arch))
}
