/**
 * Style editor (former styles popup): pick or name a style, edit its prompt
 * and negative prompt, copy them from the form's prompts, save or delete.
 * Saving refreshes the style list and the forms' `styles` choices.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { ClipboardCopy, Save, Trash2 } from "lucide-react"
import { Suspense, useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { keys, queryClient, stylesQuery } from "@/api/queries"
import type { PromptStyle } from "@/api/types"
import { SearchSelect } from "@/components/SearchSelect"
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
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { LazyPromptEditor } from "../index"

type StyleEditorDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** The form's current prompts, for "Copy from prompts". */
  currentPrompts: () => { prompt: string; negative: string }
  /** Style to show first. */
  initial?: string | null
}

function refreshStyles() {
  void queryClient.invalidateQueries({ queryKey: keys.styles })
  // The forms' `styles` choices are resolved into the schema.
  void queryClient.invalidateQueries({ queryKey: ["schema"] })
}

export function StyleEditorDialog({ open, onOpenChange, currentPrompts, initial }: StyleEditorDialogProps) {
  const { t } = useTranslation()
  const id = useId()
  const styles = useQuery({ ...stylesQuery, enabled: open })
  const [draft, setDraft] = useState<PromptStyle>({ name: initial ?? "", prompt: "", negative_prompt: "" })
  const [confirmDelete, setConfirmDelete] = useState(false)
  const existing = styles.data?.find((style) => style.name === draft.name)
  const dirty = !existing || existing.prompt !== draft.prompt || existing.negative_prompt !== draft.negative_prompt

  const select = (name: string) => {
    const style = styles.data?.find((s) => s.name === name)
    setDraft(style ? { ...style } : { name, prompt: draft.prompt, negative_prompt: draft.negative_prompt })
  }

  const save = useMutation({
    mutationFn: () => api.styles.save({ ...draft, name: draft.name.trim() }),
    onSuccess: (saved) => {
      refreshStyles()
      toast.success(t("Style “{{name}}” saved", { name: saved?.name ?? draft.name }))
    },
    onError: (error) => toast.error(t("Could not save the style"), { description: error.message }),
  })
  const remove = useMutation({
    mutationFn: () => api.styles.remove(draft.name),
    onSuccess: () => {
      refreshStyles()
      toast.success(t("Style “{{name}}” deleted", { name: draft.name }))
      setDraft({ name: "", prompt: "", negative_prompt: "" })
    },
    onError: (error) => toast.error(t("Could not delete the style"), { description: error.message }),
  })

  const options = (styles.data ?? []).map((style) => ({ value: style.name, label: style.name }))

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="flex max-h-[min(44rem,calc(100dvh-2rem))] flex-col gap-4 sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{t("Styles")}</DialogTitle>
            <DialogDescription>
              {t("Use {prompt} in a style to place your prompt there; otherwise the style is appended to the prompt.")}
            </DialogDescription>
          </DialogHeader>
          <div className="flex min-h-0 flex-col gap-3 overflow-y-auto">
            <div className="flex flex-col gap-1.5">
              <label id={`${id}-name`} className="text-[0.8rem] font-medium">
                {t("Style")}
              </label>
              <SearchSelect
                aria-labelledby={`${id}-name`}
                options={options}
                value={draft.name || null}
                onValueChange={select}
                allowCustom
                placeholder={t("Choose a style or type a new name")}
              />
            </div>
            {(["prompt", "negative_prompt"] as const).map((field) => (
              <div key={field} className="flex flex-col gap-1.5">
                <span id={`${id}-${field}`} className="text-[0.8rem] font-medium">
                  {field === "prompt" ? t("Prompt") : t("Negative prompt")}
                </span>
                <Suspense fallback={<div aria-hidden className="h-20 rounded-lg border border-input" />}>
                  <LazyPromptEditor
                    value={draft[field]}
                    onChange={(value) => setDraft((d) => ({ ...d, [field]: value }))}
                    ariaLabel={field === "prompt" ? t("Style prompt") : t("Style negative prompt")}
                    kind={field === "prompt" ? "positive" : "negative"}
                    minLines={3}
                    maxLines={10}
                  />
                </Suspense>
              </div>
            ))}
          </div>
          <DialogFooter className="flex-wrap gap-2 sm:justify-between">
            <Button
              variant="ghost"
              onClick={() => {
                const { prompt, negative } = currentPrompts()
                setDraft((d) => ({ ...d, prompt, negative_prompt: negative }))
              }}
            >
              <ClipboardCopy />
              {t("Copy from prompts")}
            </Button>
            <div className="flex gap-2">
              <Button variant="outline" disabled={!existing || remove.isPending} onClick={() => setConfirmDelete(true)}>
                <Trash2 />
                {t("Delete")}
              </Button>
              <Button disabled={!draft.name.trim() || !dirty || save.isPending} onClick={() => save.mutate()}>
                <Save />
                {t("Save")}
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      <AlertDialog open={confirmDelete} onOpenChange={setConfirmDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("Delete style “{{name}}”?", { name: draft.name })}</AlertDialogTitle>
            <AlertDialogDescription>{t("The style is removed from styles.csv.")}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmDelete(false)
                remove.mutate()
              }}
            >
              {t("Delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
