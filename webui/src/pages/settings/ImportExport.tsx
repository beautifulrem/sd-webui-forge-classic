/**
 * Settings file import / export. Export writes the stored values (changed
 * ones by default) as JSON; import reads such a file (or an A1111
 * `config.json`), validates the differences on the server without applying
 * them, and stages the valid ones as unsaved edits to review and apply.
 */
import { AlertTriangle, Check, FileUp, Loader2 } from "@/components/icons"
import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import { validationErrors } from "@/api/http"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { type ImportPlan, planImport } from "./settings-model"
import { editorValues, stageValues, useSettingsEditor } from "./settings-store"

type Checked = { plan: ImportPlan; rejected: Record<string, string> }

export function ImportDialog({ open, onOpenChange, onStaged }: { open: boolean; onOpenChange: (open: boolean) => void; onStaged: (count: number) => void }) {
  const { t } = useTranslation()
  const inputId = useId()
  const [checked, setChecked] = useState<Checked | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const reset = () => {
    setChecked(null)
    setError(null)
  }

  const read = async (file: File) => {
    reset()
    setBusy(true)
    try {
      const { inputs } = useSettingsEditor.getState()
      const editable = new Set([...inputs].filter(([, node]) => node.enabled_if !== false && !node.locked).map(([key]) => key))
      const plan = planImport(await file.text(), editorValues(), editable)
      const rejected: Record<string, string> = {}
      if (Object.keys(plan.changes).length > 0) {
        try {
          await api.validateOptions(plan.changes)
        } catch (failure) {
          const errors = validationErrors(failure)
          if (!errors) throw failure
          for (const item of errors) rejected[item.param] = item.message
        }
      }
      setChecked({ plan, rejected })
    } catch (failure) {
      setError(failure instanceof Error ? t(failure.message) : String(failure))
    } finally {
      setBusy(false)
    }
  }

  const valid = checked ? Object.entries(checked.plan.changes).filter(([key]) => !Object.hasOwn(checked.rejected, key)) : []
  const rejected = checked ? Object.entries(checked.rejected) : []

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) reset()
        onOpenChange(next)
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("Import settings")}</DialogTitle>
          <DialogDescription>{t("Read a settings file. Its changes become unsaved edits you can review before applying them.")}</DialogDescription>
        </DialogHeader>
        <label
          htmlFor={inputId}
          className="flex cursor-pointer flex-col items-center gap-2 rounded-md border border-dashed px-4 py-6 text-center text-sm text-muted-foreground transition-colors hover:border-primary/50 hover:bg-muted/40 focus-within:ring-3 focus-within:ring-ring/50"
        >
          {busy ? <Loader2 className="size-5 animate-spin" aria-hidden /> : <FileUp className="size-5" aria-hidden />}
          <span>{t("Choose a .json settings file")}</span>
          <input
            id={inputId}
            type="file"
            accept="application/json,.json"
            className="sr-only"
            onChange={(event) => {
              const file = event.target.files?.[0]
              event.target.value = ""
              if (file) void read(file)
            }}
          />
        </label>
        {error ? (
          <p role="alert" className="flex items-center gap-2 text-sm text-destructive">
            <AlertTriangle className="size-4" aria-hidden />
            {error}
          </p>
        ) : null}
        {checked ? (
          <div className="flex flex-col gap-2 text-sm" aria-live="polite">
            <p className="flex items-center gap-2">
              <Check className="size-4 text-success" aria-hidden />
              {t("{{count}} settings will change", { count: valid.length })}
            </p>
            <p className="text-muted-foreground">
              {t("{{same}} already match, {{unknown}} unknown here (ignored)", { same: checked.plan.same.length, unknown: checked.plan.unknown.length })}
            </p>
            {rejected.length > 0 ? (
              <div className="rounded-lg bg-destructive/10 px-3 py-2">
                <p className="font-medium text-destructive">{t("{{count}} values are not valid here and are skipped:", { count: rejected.length })}</p>
                <ul className="mt-1 max-h-32 overflow-y-auto text-xs">
                  {rejected.map(([key, message]) => (
                    <li key={key}>
                      <code className="font-mono">{key}</code>: {message}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        ) : null}
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("Cancel")}
          </Button>
          <Button
            disabled={valid.length === 0}
            onClick={() => {
              stageValues(Object.fromEntries(valid))
              onStaged(valid.length)
              reset()
              onOpenChange(false)
            }}
          >
            {t("Stage changes")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
