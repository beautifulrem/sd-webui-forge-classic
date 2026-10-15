/** Where prompt cards go: the positive or negative prompt (and, on the Networks page, which form). */
import { useTranslation } from "react-i18next"
import { Segmented } from "@/components/Segmented"
import type { GenerateTab } from "@/features/progress/run-controller"
import { usePromptTarget } from "./prompt-target"
import { type PromptTarget, useNetworksUi } from "./store"

export function PromptTargetSwitch({ className }: { className?: string }) {
  const { t } = useTranslation()
  const target = usePromptTarget()
  const setTarget = useNetworksUi((s) => s.setTarget)
  return (
    <Segmented<PromptTarget>
      size="sm"
      aria-label={t("Insert into")}
      value={target}
      onValueChange={setTarget}
      options={[
        { value: "positive", label: t("Prompt") },
        { value: "negative", label: t("Negative") },
      ]}
      className={className}
    />
  )
}

export function FormTargetSwitch({ className }: { className?: string }) {
  const { t } = useTranslation()
  const form = useNetworksUi((s) => s.pageForm)
  const setForm = useNetworksUi((s) => s.setPageForm)
  return (
    <Segmented<GenerateTab>
      size="sm"
      aria-label={t("Form")}
      value={form}
      onValueChange={setForm}
      options={[
        { value: "txt2img", label: t("txt2img") },
        { value: "img2img", label: t("img2img") },
      ]}
      className={className}
    />
  )
}
