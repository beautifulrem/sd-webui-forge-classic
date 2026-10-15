/** Confirms "Reset all parameters" when it would discard prompt text or source images (see `requestReset`). */
import { useTranslation } from "react-i18next"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { useResetRequest } from "./reset-request"
import { resetForm } from "./run-helpers"

export function ResetFormDialog() {
  const { t } = useTranslation()
  const tab = useResetRequest((s) => s.tab)
  const close = () => useResetRequest.setState({ tab: null })
  return (
    <AlertDialog open={tab !== null} onOpenChange={(open) => !open && close()}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("Reset all parameters?")}</AlertDialogTitle>
          <AlertDialogDescription>{t("Every parameter returns to its default, including the prompts and the source images.")}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            onClick={() => {
              if (tab) resetForm(tab)
              close()
            }}
          >
            {t("Reset")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
