/** Confirms "Reset all parameters" when it would discard prompt text or source images (see `requestReset`). */
import { useTranslation } from "react-i18next"
import { ConfirmDialog } from "@/components/ConfirmDialog"
import { useResetRequest } from "./reset-request"
import { resetForm } from "./run-helpers"

export function ResetFormDialog() {
  const { t } = useTranslation()
  const tab = useResetRequest((s) => s.tab)
  const close = () => useResetRequest.setState({ tab: null })
  return (
    <ConfirmDialog
      open={tab !== null}
      onOpenChange={(open) => !open && close()}
      title={t("Reset all parameters?")}
      description={t("Every parameter returns to its default, including the prompts and the source images.")}
      confirmLabel={t("Reset")}
      destructive
      onConfirm={() => {
        if (tab) resetForm(tab)
        close()
      }}
    />
  )
}
