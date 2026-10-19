/**
 * `user.css` from the data folder, applied over the app's own styles as the
 * Gradio UI did (it linked it after every other stylesheet). The server lists
 * it in the bootstrap's `user_assets` when the file exists and serves it from
 * `/user-assets/user.css`.
 *
 * The `<link>` goes at the end of `<body>`: every stylesheet the app loads
 * (the entry CSS and the CSS of lazily loaded pages, which Vite appends to
 * `<head>`) comes earlier in document order, so a user rule wins over an app
 * rule of the same specificity.
 */
import { useEffect } from "react"
import { api } from "@/api/endpoints"

export const USER_CSS = "user.css"
export const USER_CSS_ID = "remi-user-css"

/** Adds, keeps or removes the user stylesheet's `<link>` for the listed assets. */
export function applyUserStylesheet(assets: readonly string[] | undefined, doc: Document = document): void {
  const existing = doc.getElementById(USER_CSS_ID)
  if (!assets?.includes(USER_CSS)) {
    existing?.remove()
    return
  }
  if (existing) return
  const link = doc.createElement("link")
  link.id = USER_CSS_ID
  link.rel = "stylesheet"
  link.href = api.userAssetUrl(USER_CSS)
  doc.body.append(link)
}

/** Keeps the user stylesheet in step with the bootstrap's `user_assets`. */
export function useUserStylesheet(assets: readonly string[] | undefined): void {
  useEffect(() => applyUserStylesheet(assets), [assets])
}
