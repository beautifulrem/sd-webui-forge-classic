/** Pasting generation parameters into the form on screen, with feedback toasts. */
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import type { FormTab, InfotextParsed } from "@/api/types"
import { readImageParameters } from "@/features/infotext/read-image-info"
import { applyPaste, pastePlan } from "@/features/infotext/paste"
import { useForm } from "@/ui/hooks"

/**
 * Parameters into the form: `text` is parsed by the server for this form
 * (empty text = the last generation's); `image` is read in the browser, or
 * uploaded and read by the server (`/infotext/image`) when the browser finds
 * nothing (formats it does not parse).
 */
export function usePasteParameters() {
  const { t } = useTranslation()
  const { store, formId } = useForm()
  const tab = formId as FormTab

  const apply = (parsed: InfotextParsed, id: string | number) => {
    const count = applyPaste({ store, tab }, pastePlan(store.getState().index, parsed).entries)
    if (count === 0) toast.info(t("No parameters to apply"), { id })
    else toast.success(t("Parameters applied: {{count}}", { count }), { id })
    const problems = parsed.problems ?? []
    if (problems.length > 0) {
      toast.warning(t("Some values were skipped"), { description: problems.slice(0, 4).map((p) => `${p.param ?? p.owner ?? ""}: ${p.message}`).join("\n") })
    }
  }
  const fail = (id: string | number, error: unknown) =>
    toast.error(t("Could not read the parameters"), { id, description: error instanceof Error ? error.message : String(error) })

  const fromText = async (text: string) => {
    const id = toast.loading(t("Reading parameters…"))
    try {
      apply(await api.parseInfotext(text, tab), id)
    } catch (error) {
      fail(id, error)
    }
  }

  const fromImage = async (file: File) => {
    const id = toast.loading(t("Reading parameters…"))
    try {
      // A file the browser cannot parse (corrupt EXIF, …) is still worth the server's look.
      const text = await readImageParameters(file).catch(() => null)
      if (text) return apply(await api.parseInfotext(text, tab), id)
      const upload = await api.upload(file)
      const info = await api.imageInfo(upload.id, tab)
      if (info.parsed) apply(info.parsed, id)
      else toast.info(t("This image has no generation parameters"), { id })
    } catch (error) {
      fail(id, error)
    }
  }

  return { fromText, fromImage }
}
