/** Settings → System info: the sysinfo report (summary + collapsible sections), download, copy, and a file check. */
import { useMutation, useQuery } from "@tanstack/react-query"
import { Check, ChevronRight, Copy, Download, FileCheck2, Loader2, X } from "@/components/icons"
import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { Badge } from "@/components/ui/badge"
import { Button, buttonVariants } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"

/** Short scalar entries shown as a summary; everything else is a collapsible section. */
const SUMMARY_KEYS = ["Platform", "Python", "Version", "Git status", "Script path", "Data path", "Extensions dir"]

function formatValue(value: unknown): string {
  if (typeof value === "string") return value
  return JSON.stringify(value, null, 2) ?? String(value)
}

function sizeHint(value: unknown): string | null {
  if (Array.isArray(value)) return String(value.length)
  if (value && typeof value === "object") return String(Object.keys(value).length)
  return null
}

function Section({ name, value }: { name: string; value: unknown }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const hint = sizeHint(value)
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-lg border bg-card">
      <CollapsibleTrigger className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm font-medium outline-none hover:bg-muted/50 focus-visible:ring-2 focus-visible:ring-ring/60">
        <ChevronRight className={cn("size-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-90")} aria-hidden />
        <span className="flex-1">{t(name)}</span>
        {hint ? <span className="text-xs text-muted-foreground tabular-nums">{hint}</span> : null}
      </CollapsibleTrigger>
      <CollapsibleContent>
        {open ? (
          <pre className="max-h-96 overflow-auto border-t px-3 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap break-all">{formatValue(value)}</pre>
        ) : null}
      </CollapsibleContent>
    </Collapsible>
  )
}

function CheckFile() {
  const { t } = useTranslation()
  const id = useId()
  const check = useMutation({ mutationFn: async (file: File) => api.system.checkSysinfo(await file.text()) })
  return (
    <div className="flex flex-wrap items-center gap-2">
      <label htmlFor={id} className={cn(buttonVariants({ variant: "outline", size: "sm" }), "cursor-pointer focus-within:ring-3 focus-within:ring-ring/50")}>
        {check.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <FileCheck2 aria-hidden />}
        {t("Check a sysinfo file")}
        <input
          id={id}
          type="file"
          accept=".json,.txt,application/json,text/plain"
          className="sr-only"
          onChange={(event) => {
            const file = event.target.files?.[0]
            event.target.value = ""
            if (file) check.mutate(file)
          }}
        />
      </label>
      <span aria-live="polite">
        {check.data ? (
          check.data.valid ? (
            <Badge variant="secondary" className="gap-1">
              <Check aria-hidden />
              {t("Valid: the file was not modified")}
            </Badge>
          ) : (
            <Badge variant="destructive" className="gap-1">
              <X aria-hidden />
              {t("Invalid: the file was modified or is not a sysinfo file")}
            </Badge>
          )
        ) : check.error ? (
          <span className="text-sm text-destructive">{check.error.message}</span>
        ) : null}
      </span>
    </div>
  )
}

export function SysinfoView() {
  const { t } = useTranslation()
  const info = useQuery({ queryKey: ["sysinfo"], queryFn: ({ signal }) => api.system.sysinfo(signal), staleTime: 60_000 })
  const data = info.data
  const summary = data ? SUMMARY_KEYS.filter((key) => key in data && typeof data[key] !== "object") : []
  const sections = data ? Object.keys(data).filter((key) => !summary.includes(key) && key !== "Checksum") : []

  const copy = async () => {
    if (!data) return
    try {
      await navigator.clipboard.writeText(JSON.stringify(data, null, 4))
      toast.success(t("Copied to the clipboard"))
    } catch {
      toast.error(t("Could not copy to the clipboard"))
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-6 sm:px-6">
      <header className="flex flex-wrap items-end gap-3">
        <div className="min-w-0 flex-1">
          <h1 className="text-lg font-semibold tracking-tight">{t("System info")}</h1>
          <p className="text-sm text-muted-foreground">{t("Attach the downloaded file when you report a problem.")}</p>
        </div>
        <a href={api.system.sysinfoDownloadUrl()} download className={buttonVariants({ size: "sm" })}>
          <Download aria-hidden />
          {t("Download")}
        </a>
        <Button variant="outline" size="sm" onClick={() => void copy()} disabled={!data}>
          <Copy aria-hidden />
          {t("Copy")}
        </Button>
      </header>
      <CheckFile />
      {info.isPending ? (
        <div className="flex flex-col gap-2" aria-busy="true">
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} className="h-9 w-full" />
          ))}
        </div>
      ) : info.isError ? (
        <p role="alert" className="text-sm text-destructive">
          {info.error.message}
        </p>
      ) : (
        <>
          <dl className="grid gap-x-4 gap-y-2 rounded-md bg-card p-4 text-sm sm:grid-cols-[max-content_1fr]">
            {summary.map((key) => (
              <div key={key} className="contents">
                <dt className="text-muted-foreground">{t(key)}</dt>
                <dd className="min-w-0 font-mono text-xs break-all sm:pt-0.5">{formatValue(data?.[key])}</dd>
              </div>
            ))}
          </dl>
          <div className="flex flex-col gap-2">
            {sections.map((key) => (
              <Section key={key} name={key} value={data?.[key]} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
