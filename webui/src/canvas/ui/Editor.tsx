/**
 * The image editor (lazy chunk with Konva): binds a document to its values —
 * the background upload id and each layer's upload id — in both directions,
 * uploads new backgrounds (pick, drop, paste, blank canvas), and shows the
 * editor inline or maximised.
 */
import { Maximize2 } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { useClientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog"
import { showToast } from "@/lib/toast"
import { cn } from "@/lib/utils"
import { imageFiles, previewUrlOf, uploadFile, useUploadInfo } from "@/ui/uploads"
import type { LayerKind } from "../core/modes"
import { getDocument, loadImage } from "../engine/registry"
import type { ImageEditorProps } from "../types"
import { EditorBody } from "./EditorBody"
import { useDocValue } from "./use-doc"

const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error))

export default function Editor(props: ImageEditorProps) {
  const { t } = useTranslation()
  const doc = getDocument(props.docKey, props.pendingKey)
  const screenBrush = useClientOption("forge_canvas_consistent_brush")
  const plain = useClientOption("forge_canvas_plain")
  const plainColor = useClientOption("forge_canvas_plain_color")
  const toolbarAlways = useClientOption("forge_canvas_toolbar_always")
  const canvasHeight = useClientOption("forge_canvas_height")
  const [uploading, setUploading] = useState(0)
  const [maximized, setMaximized] = useState(false)
  /** A background that could not be loaded (no endless spinner): the editor shows an error with Retry. */
  const [failed, setFailed] = useState<string | null>(null)
  /** Bumped by Retry: loads the background again. */
  const [attempt, setAttempt] = useState(0)
  const imageUrl = useUploadInfo(props.image)?.url ?? null
  /** The background id the document holds; layers load only over their own background, and painting waits for it. */
  const loadedId = useDocValue(doc, (d) => d.imageId)
  const ready = loadedId === props.image
  const loadFailed = props.image !== null && failed === props.image
  const busy = uploading > 0 || (props.image !== null && !ready && !loadFailed)

  // Committed layer uploads go to whoever renders the editor now.
  const handlers = useRef({ onLayerChange: props.onLayerChange, onImageChange: props.onImageChange })
  useEffect(() => {
    handlers.current = { onLayerChange: props.onLayerChange, onImageChange: props.onImageChange }
  })
  useEffect(() => {
    doc.bindHandlers({
      onCommit: (key, id) => handlers.current.onLayerChange?.(key, id),
      onSyncError: (error) => showToast("error", t("Could not upload the canvas layer"), errorText(error)),
    })
  }, [doc, t])

  // Background: follow the value (also when it is set from outside, e.g. "Send to inpaint").
  useEffect(() => {
    const id = props.image
    if (id === doc.imageId) return
    if (!id) {
      doc.setBackground(null, { width: 0, height: 0 }, null)
      return
    }
    if (!imageUrl) return
    let cancelled = false
    loadImage(imageUrl)
      .then((image) => {
        if (cancelled) return
        setFailed(null)
        doc.setBackground(image, { width: image.naturalWidth, height: image.naturalHeight }, id)
      })
      .catch((error) => {
        if (cancelled) return
        // Never leave the old picture (and its layers) editable under the new value.
        doc.setBackground(null, { width: 0, height: 0 }, null)
        setFailed(id)
        showToast("error", t("Could not load the image"), errorText(error))
      })
    return () => {
      cancelled = true
    }
  }, [doc, props.image, imageUrl, attempt, t])

  // Layers: follow their values (cleared, or set from outside) once the background is in.
  const layerValues = JSON.stringify(Object.entries(props.layers ?? {}).map(([key, layer]) => [key, layer.kind, layer.value]))
  useEffect(() => {
    if (!ready || !doc.hasImage) return
    let cancelled = false
    for (const [key, kind, value] of JSON.parse(layerValues) as [string, LayerKind, string | null][]) {
      const current = doc.layer(key)
      if (value === (current?.sync.value ?? null)) continue
      if (value === null) {
        if (current) doc.resetLayer(key)
        continue
      }
      loadImage(previewUrlOf(value) ?? api.fileUrl(value))
        .then((image) => {
          if (!cancelled && doc.layer(key)?.sync.value !== value) doc.loadLayer(key, kind, image, value)
        })
        .catch((error) => {
          if (!cancelled) showToast("error", t("Could not load the image"), errorText(error))
        })
    }
    return () => {
      cancelled = true
    }
  }, [doc, ready, layerValues, t])

  const uploadBackground = async (file: File) => {
    setUploading((n) => n + 1)
    try {
      const info = await uploadFile(file)
      const image = await loadImage(info.url ?? api.fileUrl(info.id))
      const size = { width: image.naturalWidth, height: image.naturalHeight }
      // The layers' last edits reach their values first, so the owner can offer to undo the replacement.
      await doc.flush().catch(() => {})
      doc.setBackground(image, size, info.id)
      handlers.current.onImageChange?.({ id: info.id, ...size })
    } catch (error) {
      showToast("error", t("Upload failed"), errorText(error))
    } finally {
      setUploading((n) => n - 1)
    }
  }

  const onFiles = (files: File[]) => {
    const [first] = imageFiles(files)
    if (first) void uploadBackground(first)
  }

  const onBlank = () => {
    const size = props.blankSize
    if (!size) return
    const canvas = document.createElement("canvas")
    canvas.width = size.width
    canvas.height = size.height
    const ctx = canvas.getContext("2d")
    if (!ctx) return
    ctx.fillStyle = "#ffffff"
    ctx.fillRect(0, 0, size.width, size.height)
    canvas.toBlob((blob) => {
      canvas.width = canvas.height = 0
      if (blob) void uploadBackground(new File([blob], "blank.png", { type: "image/png" }))
    }, "image/png")
  }

  const onRemove = async () => {
    await doc.flush().catch(() => {})
    doc.setBackground(null, { width: 0, height: 0 }, null)
    handlers.current.onImageChange?.(null)
  }

  const onRetry = () => {
    setFailed(null)
    setAttempt((n) => n + 1)
  }

  const body = (inDialog: boolean) => (
    <EditorBody
      key={props.docKey}
      doc={doc}
      editor={props}
      screenBrush={screenBrush}
      plainColor={plain ? plainColor : null}
      toolbarAlways={toolbarAlways || inDialog}
      maximized={inDialog}
      onMaximize={setMaximized}
      busy={busy}
      ready={ready}
      loadFailed={loadFailed}
      onRetry={onRetry}
      onFiles={onFiles}
      onBlank={onBlank}
      onRemove={() => void onRemove()}
    />
  )

  return (
    <div
      role="group"
      aria-label={props.label ?? t("Image editor")}
      data-fill={props.fill ? "" : undefined}
      className={cn("relative min-h-0 overflow-hidden rounded-xl border bg-canvas", props.error && "border-destructive/60", props.className)}
      style={props.fill ? undefined : { height: props.height ?? canvasHeight }}
    >
      {maximized ? (
        <div className="flex size-full flex-col items-center justify-center gap-2 p-4 text-center text-sm text-muted-foreground">
          <Maximize2 className="size-5 opacity-60" aria-hidden />
          <p>{t("Editing in full screen")}</p>
          <Button variant="outline" size="sm" onClick={() => setMaximized(false)}>
            {t("Exit full screen")}
          </Button>
        </div>
      ) : (
        body(false)
      )}
      <Dialog open={maximized} onOpenChange={setMaximized}>
        <DialogContent
          showCloseButton={false}
          className="top-0 left-0 flex h-dvh w-screen max-w-none translate-x-0 translate-y-0 gap-0 rounded-none bg-canvas p-0 ring-0 sm:max-w-none"
        >
          <DialogTitle className="sr-only">{props.label ?? t("Image editor")}</DialogTitle>
          <DialogDescription className="sr-only">{t("Press Escape or F to exit full screen.")}</DialogDescription>
          {maximized ? body(true) : null}
        </DialogContent>
      </Dialog>
    </div>
  )
}
