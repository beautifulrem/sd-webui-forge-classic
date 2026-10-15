/** Install an extension from a git URL, optionally a branch and a folder name. */
import { useMutation } from "@tanstack/react-query"
import { CheckCircle2, Download, Loader2 } from "lucide-react"
import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { requestServerCommand } from "@/features/server-control/server-control"
import { FieldError } from "@/ui/Field"
import { adminErrorMessage, dirnameFromUrl, isGitUrl, isValidDirname } from "./extensions-model"
import { extensionKeys } from "./queries"

export function InstallUrlView({ access }: { access: boolean }) {
  const { t } = useTranslation()
  const id = useId()
  const [url, setUrl] = useState("")
  const [branch, setBranch] = useState("")
  const [dirname, setDirname] = useState("")
  const [touched, setTouched] = useState(false)
  const install = useMutation({
    mutationFn: () => api.extensions.install({ url: url.trim(), branch: branch.trim() || null, dirname: dirname.trim() || null }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: extensionKeys.installed })
      void queryClient.invalidateQueries({ queryKey: ["extensions", "index"] })
    },
  })
  const folder = dirname.trim() || dirnameFromUrl(url)
  const urlError = touched && url.trim() && !isGitUrl(url) ? t("Enter a git URL (https://, ssh://, git:// or git@host:path).") : null
  const folderError = folder && !isValidDirname(folder) ? t("Use letters, digits, dots, dashes and underscores.") : null
  const valid = isGitUrl(url) && Boolean(folder) && !folderError

  return (
    <form
      className="flex max-w-xl flex-col gap-4"
      onSubmit={(event) => {
        event.preventDefault()
        setTouched(true)
        if (valid) install.mutate()
      }}
    >
      <div className="flex flex-col gap-1.5">
        <label htmlFor={`${id}-url`} className="text-sm font-medium">
          {t("Repository URL")}
        </label>
        <Input
          id={`${id}-url`}
          value={url}
          onChange={(event) => {
            setUrl(event.target.value)
            install.reset()
          }}
          onBlur={() => setTouched(true)}
          placeholder="https://github.com/user/sd-webui-extension"
          className="font-mono text-xs"
          required
          disabled={!access}
          aria-invalid={urlError ? true : undefined}
          aria-describedby={urlError ? `${id}-url-error` : undefined}
          autoComplete="off"
          spellCheck={false}
        />
        {urlError ? <FieldError id={`${id}-url`} message={urlError} /> : null}
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="flex flex-col gap-1.5">
          <label htmlFor={`${id}-branch`} className="text-sm font-medium">
            {t("Branch")}
          </label>
          <Input id={`${id}-branch`} value={branch} onChange={(event) => setBranch(event.target.value)} placeholder={t("Default branch")} disabled={!access} spellCheck={false} />
        </div>
        <div className="flex flex-col gap-1.5">
          <label htmlFor={`${id}-dir`} className="text-sm font-medium">
            {t("Folder name")}
          </label>
          <Input
            id={`${id}-dir`}
            value={dirname}
            onChange={(event) => setDirname(event.target.value)}
            placeholder={dirnameFromUrl(url) || t("From the URL")}
            disabled={!access}
            spellCheck={false}
            aria-invalid={folderError ? true : undefined}
            aria-describedby={folderError ? `${id}-dir-error` : undefined}
          />
          {folderError ? <FieldError id={`${id}-dir`} message={folderError} /> : null}
        </div>
      </div>
      <p className="text-xs text-muted-foreground">{t("The repository is cloned into extensions/, its install script runs, and it loads with the next Reload UI.")}</p>
      <div className="flex items-center gap-3">
        <Button type="submit" disabled={!access || install.isPending || (touched && !valid)}>
          {install.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Download aria-hidden />}
          {install.isPending ? t("Installing…") : t("Install")}
        </Button>
        <div aria-live="polite" className="min-w-0 text-sm">
          {install.isSuccess ? (
            <span className="flex flex-wrap items-center gap-2">
              <CheckCircle2 className="size-4 text-success" aria-hidden />
              {t("{{name}} is installed.", { name: install.data.name })}
              <Button type="button" size="sm" variant="outline" onClick={() => requestServerCommand("reload")}>
                {t("Reload UI")}
              </Button>
            </span>
          ) : install.isError ? (
            <span className="text-destructive">{adminErrorMessage(install.error, t)}</span>
          ) : null}
        </div>
      </div>
    </form>
  )
}
