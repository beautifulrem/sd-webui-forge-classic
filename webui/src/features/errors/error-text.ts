/**
 * A failed request as text for the user, in the UI language. The server's
 * error details are English (they are for logs and API clients), so the
 * pages show a sentence chosen by status code instead; callers pass their
 * own (already translated) sentence for the statuses that mean something
 * specific to them.
 */
import { toast } from "sonner"
import { ApiError } from "@/api/http"
import { t } from "@/i18n"
import type { ValidationError } from "@/ui/schema"

export type StatusTexts = Partial<Record<number, string>>

export function errorText(error: unknown, texts: StatusTexts = {}): string {
  if (!(error instanceof ApiError)) return t("The server cannot be reached.")
  const own = texts[error.status]
  if (own) return own
  switch (error.status) {
    case 401:
      return t("Sign in again to continue.")
    case 403:
      return t("The server does not allow this.")
    case 404:
      return t("It is no longer there.")
    case 409:
      return t("It changed meanwhile; refresh and try again.")
    case 413:
      return t("The file is too large.")
    case 415:
      return t("This file cannot be read as an image.")
    case 422:
      return t("The server rejected the request.")
    case 503:
      return t("The server is busy; try again shortly.")
    default:
      return error.status >= 500 ? t("The server ran into an error ({{status}}).", { status: error.status }) : t("The request failed ({{status}}).", { status: error.status })
  }
}

/**
 * Rejected values in the UI language: each distinct message translated with
 * the same key as the inline error under its field (the server's messages
 * are catalog keys, see `core.required`).
 */
export function validationText(errors: readonly ValidationError[]): string {
  return [...new Set(errors.map((error) => t(error.message)))].join(" · ")
}

/** The "Some parameters are invalid" toast for rejected or missing values (the fields show them inline). */
export function toastInvalid(errors: readonly ValidationError[]): void {
  toast.error(t("Some parameters are invalid"), { description: validationText(errors) })
}
