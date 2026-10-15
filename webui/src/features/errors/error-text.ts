/**
 * A failed request as text for the user, in the UI language. The server's
 * error details are English (they are for logs and API clients), so the
 * pages show a sentence chosen by status code instead; callers pass their
 * own (already translated) sentence for the statuses that mean something
 * specific to them.
 */
import { ApiError } from "@/api/http"
import { t } from "@/i18n"

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
