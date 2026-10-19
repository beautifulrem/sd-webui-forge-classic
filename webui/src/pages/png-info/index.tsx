/**
 * PNG Info: drop, paste or choose an image (or open one from the gallery,
 * `#/png-info?file=<id>`) to read everything it carries: the generation
 * parameters, every metadata entry (PNG text, EXIF, …) and the PNG chunks.
 * Send its parameters to txt2img / img2img, or the image itself to img2img,
 * inpaint or Extras.
 */
import { useQuery } from "@tanstack/react-query"
import { ArrowRight, ClipboardCopy, FileSearch, ImageUp, Images, ListTree, Paintbrush, Rows3, SlidersHorizontal, Upload, Wand, X } from "lucide-react"
import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { navigate, type Route } from "@/app/router"
import { HeaderSlot } from "@/app/header-slot"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { errorText } from "@/features/errors/error-text"
import { copyText } from "@/features/gallery/files"
import { InfotextPanel } from "@/features/infotext/InfotextPanel"
import { isImageFile } from "@/features/infotext/read-image-info"
import { ChunksTable, EntriesTable, ParametersTable } from "@/features/pnginfo/MetadataTables"
import { applyParameters, sendWithFeedback } from "@/features/send-to/result-actions"
import { PageSendButtons } from "@/pages/ext/send-targets"
import { DropZone } from "@/ui/inputs/DropZone"

const ACCEPT = "image/png,image/jpeg,image/webp,image/gif,image/avif,image/bmp,image/tiff"

function useUpload(onDone: (id: string) => void) {
  const { t } = useTranslation()
  const [busy, setBusy] = useState(false)
  const upload = async (file: File) => {
    setBusy(true)
    try {
      // checked here (its first bytes), so a wrong file never reaches the server (a 415 in the console)
      if ((!file.type.startsWith("image/") && file.type !== "") || !(await isImageFile(file))) {
        toast.error(t("Not an image: {{name}}", { name: file.name }))
        return
      }
      const uploaded = await api.upload(file)
      onDone(uploaded.id)
    } catch (error) {
      toast.error(t("Could not upload the image"), { description: errorText(error) })
    } finally {
      setBusy(false)
    }
  }
  return { busy, upload }
}

function Actions({ fileId, infotext }: { fileId: string; infotext: string | null }) {
  const { t } = useTranslation()
  const params = infotext ?? ""
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <h3 className="text-[0.7rem] font-semibold tracking-wide text-muted-foreground uppercase">{t("Send parameters")}</h3>
        <div className="flex flex-wrap gap-1.5">
          <Button size="sm" disabled={!infotext} onClick={() => applyParameters("txt2img", params)}>
            <Wand aria-hidden />
            {t("To txt2img")}
          </Button>
          <Button size="sm" variant="secondary" disabled={!infotext} onClick={() => applyParameters("img2img", params)}>
            <Images aria-hidden />
            {t("To img2img")}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={!infotext}
            onClick={() => void copyText(params).then((ok) => (ok ? toast.success(t("Copied")) : toast.error(t("Could not copy"))))}
          >
            <ClipboardCopy aria-hidden />
            {t("Copy")}
          </Button>
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        <h3 className="text-[0.7rem] font-semibold tracking-wide text-muted-foreground uppercase">{t("Send image")}</h3>
        <div className="flex flex-wrap gap-1.5">
          <Button size="sm" variant="outline" onClick={() => sendWithFeedback("img2img", { fileId, infotext })}>
            <Images aria-hidden />
            {t("img2img")}
          </Button>
          <Button size="sm" variant="outline" onClick={() => sendWithFeedback("inpaint", { fileId, infotext })}>
            <Paintbrush aria-hidden />
            {t("Inpaint")}
          </Button>
          <Button size="sm" variant="outline" onClick={() => sendWithFeedback("extras", { fileId })}>
            <ArrowRight aria-hidden />
            {t("Extras")}
          </Button>
          {/* Extension pages that accept images ("Send to JoyCaption") */}
          <PageSendButtons fileId={fileId} />
        </div>
        <p className="text-xs text-muted-foreground">{t("img2img and inpaint also take the parameters, following the send seed / size settings.")}</p>
      </div>
    </div>
  )
}

function Details({ fileId }: { fileId: string }) {
  const { t } = useTranslation()
  const info = useQuery({ queryKey: ["pnginfo", fileId], queryFn: ({ signal }) => api.pngInfo(fileId, signal), staleTime: Infinity })
  if (info.isPending) return <Skeleton className="h-64 w-full" />
  if (info.isError)
    return (
      <p role="alert" className="rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive">
        {errorText(info.error, { 404: t("The image is no longer there."), 415: t("This file cannot be read as an image (or it is too large to read).") })}
      </p>
    )
  const data = info.data
  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-xl border bg-card p-4">
        <div className="mb-3 flex items-baseline justify-between gap-2">
          <h2 className="truncate text-sm font-semibold" title={data.name}>
            {data.name}
          </h2>
          <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
            {data.format ?? "?"} · {data.width}×{data.height}
          </span>
        </div>
        {data.infotext ? (
          <InfotextPanel infotext={data.infotext} />
        ) : (
          <p className="text-sm text-muted-foreground">{t("This image has no generation parameters. Its other metadata is below.")}</p>
        )}
        <div className="mt-4 border-t pt-4">
          <Actions fileId={fileId} infotext={data.infotext ?? null} />
        </div>
      </div>
      <Tabs defaultValue="params" className="rounded-xl border bg-card p-3">
        <TabsList variant="line" className="w-full justify-start border-b pb-px">
          <TabsTrigger value="params" className="flex-none px-2.5">
            <SlidersHorizontal aria-hidden />
            {t("Parameters")}
          </TabsTrigger>
          <TabsTrigger value="entries" className="flex-none px-2.5">
            <ListTree aria-hidden />
            {t("All metadata")}
            <span className="text-xs text-muted-foreground tabular-nums">{data.entries.length}</span>
          </TabsTrigger>
          <TabsTrigger value="chunks" className="flex-none px-2.5" disabled={data.chunks.length === 0}>
            <Rows3 aria-hidden />
            {t("PNG chunks")}
          </TabsTrigger>
        </TabsList>
        <TabsContent value="params" className="pt-2">
          <ParametersTable params={data.params} />
        </TabsContent>
        <TabsContent value="entries" className="pt-2">
          <EntriesTable entries={data.entries} />
        </TabsContent>
        <TabsContent value="chunks" className="pt-2">
          <ChunksTable chunks={data.chunks} />
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default function PngInfoPage({ route }: { route: Route }) {
  const { t } = useTranslation()
  const fileId = route.query?.get("file") ?? null
  const show = (id: string | null) => navigate({ name: "png-info", query: id ? new URLSearchParams({ file: id }) : undefined }, { replace: fileId !== null })
  const { busy, upload } = useUpload((id) => show(id))
  const onFiles = (files: File[]) => files[0] && void upload(files[0])

  // Paste anywhere on the page (not into a text field) reads the pasted image.
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      if (event.defaultPrevented || busy) return
      const target = event.target instanceof Element ? event.target : null
      if (target?.closest("input, textarea, [contenteditable]")) return
      const file = Array.from(event.clipboardData?.files ?? []).find((item) => item.type.startsWith("image/"))
      if (!file) return
      event.preventDefault()
      void upload(file)
    }
    window.addEventListener("paste", onPaste)
    return () => window.removeEventListener("paste", onPaste)
  })

  if (!fileId) {
    return (
      <div className="flex h-full items-center justify-center overflow-y-auto p-4">
        <DropZone onFiles={onFiles} accept={ACCEPT} disabled={busy} aria-label={t("Choose an image to read")} className="flex w-full max-w-xl flex-col items-center gap-3 px-6 py-16 text-center">
          <span className="grid size-14 place-items-center rounded-2xl bg-primary/10 text-primary">{busy ? <Upload className="size-6 animate-pulse" aria-hidden /> : <FileSearch className="size-6" aria-hidden />}</span>
          <span className="text-base font-semibold">{busy ? t("Reading…") : t("Drop an image here")}</span>
          <span className="max-w-sm text-sm text-muted-foreground">
            {t("Or paste it, or click to choose a file. PNG, JPEG, WebP and other formats: the generation parameters, metadata and PNG chunks are shown.")}
          </span>
        </DropZone>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto">
      <HeaderSlot>
        <Button variant="ghost" size="sm" onClick={() => show(null)}>
          <X aria-hidden />
          <span className="hidden sm:inline">{t("Clear")}</span>
        </Button>
      </HeaderSlot>
      <div className="mx-auto grid w-full max-w-6xl gap-4 p-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] lg:items-start">
        <div className="flex flex-col gap-2 lg:sticky lg:top-4">
          <DropZone onFiles={onFiles} accept={ACCEPT} disabled={busy} aria-label={t("Replace the image")} className="overflow-hidden border-solid">
            <img src={api.fileUrl(fileId)} alt={t("The image being read")} className="max-h-[70dvh] w-full bg-checker object-contain" />
            <span className="absolute right-2 bottom-2 inline-flex items-center gap-1 rounded-md bg-background/85 px-2 py-1 text-xs font-medium shadow-sm backdrop-blur">
              <ImageUp className="size-3.5" aria-hidden />
              {busy ? t("Reading…") : t("Drop, paste or click to replace")}
            </span>
          </DropZone>
        </div>
        <Details fileId={fileId} />
      </div>
    </div>
  )
}
