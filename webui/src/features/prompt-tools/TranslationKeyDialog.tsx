/**
 * Enters the translation provider's API key. The key is write-only: the
 * server stores it apart from the settings and only says whether one is
 * set, so the field never shows it again.
 */
import { useQueryClient } from "@tanstack/react-query"
import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { promptToolsApi } from "./api"
import { toolKeys } from "./queries"

type TranslationKeyDialogProps = {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** A key is stored already. */
  configured: boolean
}

export function TranslationKeyDialog({ open, onOpenChange, configured }: TranslationKeyDialogProps) {
  const { t } = useTranslation()
  const client = useQueryClient()
  const inputId = useId()
  const hintId = useId()
  const [key, setKey] = useState("")
  const [busy, setBusy] = useState(false)

  const save = async (value: string) => {
    setBusy(true)
    try {
      const { configured: stored } = await promptToolsApi.setTranslationKey(value)
      await client.invalidateQueries({ queryKey: toolKeys.config })
      toast.success(stored ? t("API key saved") : t("API key removed"))
      setKey("")
      onOpenChange(false)
    } catch (error) {
      toast.error(t("Something went wrong"), { description: error instanceof Error ? error.message : String(error) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) setKey("")
        onOpenChange(next)
      }}
    >
      <DialogContent className="sm:max-w-md">
        <form
          className="contents"
          onSubmit={(event) => {
            event.preventDefault()
            if (key.trim()) void save(key)
          }}
        >
          <DialogHeader>
            <DialogTitle>{t("Translation API key")}</DialogTitle>
            <DialogDescription>{t("Stored on the server, apart from the settings. It is never shown again; enter a new one to replace it.")}</DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-1.5">
            <label htmlFor={inputId} className="text-sm font-medium">
              {t("API key")}
            </label>
            <Input
              id={inputId}
              type="password"
              autoComplete="off"
              spellCheck={false}
              value={key}
              placeholder={configured ? t("A key is set") : t("Paste the key")}
              aria-describedby={hintId}
              onChange={(event) => setKey(event.target.value)}
            />
            <p id={hintId} className="text-xs text-muted-foreground">
              {configured ? t("A key is set.") : t("No key is set.")}
            </p>
          </div>
          <DialogFooter>
            {configured ? (
              <Button type="button" variant="ghost" className="mr-auto text-destructive hover:bg-destructive/10 hover:text-destructive" disabled={busy} onClick={() => void save("")}>
                {t("Remove the key")}
              </Button>
            ) : null}
            <Button type="button" variant="outline" disabled={busy} onClick={() => onOpenChange(false)}>
              {t("Cancel")}
            </Button>
            <Button type="submit" disabled={busy || !key.trim()}>
              {t("Save")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
