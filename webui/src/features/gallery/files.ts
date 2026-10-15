/** Output file helpers: download, open in a new tab, copy text. */
import { api } from "@/api/endpoints"

/** Saves an output file through the browser (the server names it). */
export function downloadFile(fileId: string) {
  const link = document.createElement("a")
  link.href = api.downloadUrl(fileId)
  // No value: the browser keeps the server's Content-Disposition file name.
  link.setAttribute("download", "")
  link.rel = "noopener"
  document.body.append(link)
  link.click()
  link.remove()
}

export function openInNewTab(fileId: string) {
  window.open(api.fileUrl(fileId), "_blank", "noopener,noreferrer")
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}
