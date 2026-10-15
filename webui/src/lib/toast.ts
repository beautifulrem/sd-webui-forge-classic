import { toast } from "sonner"

export type ToastLevel = "info" | "success" | "warning" | "error"

/** Shows a toast for a server-provided level (`undefined` = info). */
export function showToast(level: ToastLevel | undefined, message: string, description?: string) {
  const fn = level && level !== "info" ? toast[level] : toast.info
  fn(message, description ? { description } : undefined)
}
