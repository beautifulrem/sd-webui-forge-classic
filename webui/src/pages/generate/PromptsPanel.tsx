/**
 * Prompt, negative prompt and styles of the form, with "paste parameters"
 * (from the prompt text, or the last generation when it is empty), "clear"
 * (confirmed) and image drops that read the image's parameters.
 */
import { ClipboardPaste, Eraser } from "lucide-react"
import { useState } from "react"
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
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { PromptField } from "@/features/prompt-editor/PromptField"
import { StylesField } from "@/features/prompt-editor/styles/StylesField"
import { findByInfotext } from "@/features/send-to/result-actions"
import { useForm, useFormState } from "@/ui/hooks"
import type { MultiChoiceNode, TextNode } from "@/ui/schema"
import { usePasteParameters } from "./paste-parameters"

function IconAction({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger render={<Button variant="ghost" size="icon-xs" aria-label={label} onClick={onClick} />}>{children}</TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

export function PromptsPanel({ prompt, negative, styles }: { prompt: TextNode | null; negative: TextNode | null; styles: MultiChoiceNode | null }) {
  const { t } = useTranslation()
  const { store } = useForm()
  const [confirmClear, setConfirmClear] = useState(false)
  const paste = usePasteParameters()
  const index = useFormState((s) => s.index)
  const stepsRef = findByInfotext(index, "Steps")
  const steps = useFormState((s) => (stepsRef ? Number(s.values[stepsRef[0]]?.[stepsRef[1]] ?? 0) || undefined : undefined))
  const selectedStyles = useFormState((s) => s.values.core?.styles)
  const styleList = Array.isArray(selectedStyles) ? (selectedStyles as string[]) : []

  if (!prompt && !negative) return null

  return (
    <section aria-label={t("Prompts")} className="flex flex-col gap-3">
      {prompt ? (
        <PromptField
          node={prompt}
          kind="positive"
          steps={steps}
          styles={styleList}
          minLines={4}
          onDropImage={(file) => void paste.fromImage(file)}
          actions={
            <>
              <IconAction
                label={t("Read parameters from the prompt, or from the last generation when it is empty")}
                onClick={() => void paste.fromText(String(store.getState().values.core?.[prompt.name] ?? ""))}
              >
                <ClipboardPaste />
              </IconAction>
              <IconAction label={t("Clear prompts")} onClick={() => setConfirmClear(true)}>
                <Eraser />
              </IconAction>
            </>
          }
        />
      ) : null}
      {negative ? <PromptField node={negative} kind="negative" steps={steps} styles={styleList} onDropImage={(file) => void paste.fromImage(file)} /> : null}
      {styles ? <StylesField node={styles} prompt={prompt?.name} negative={negative?.name} /> : null}

      <AlertDialog open={confirmClear} onOpenChange={setConfirmClear}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("Clear both prompts?")}</AlertDialogTitle>
            <AlertDialogDescription>{t("The prompt and the negative prompt are emptied. Undo with Ctrl+Z in each editor.")}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmClear(false)
                const cleared: Record<string, string> = {}
                if (prompt) cleared[prompt.name] = ""
                if (negative) cleared[negative.name] = ""
                store.getState().setValues("core", cleared)
              }}
            >
              {t("Clear")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  )
}
