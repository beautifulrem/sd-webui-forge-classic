import { QueryClientProvider } from "@tanstack/react-query"
import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { I18nextProvider } from "react-i18next"
import { queryClient } from "@/api/queries"
import { App } from "@/app/App"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { activateLocale, i18n, initialLocale } from "@/i18n"
import "@/features/prompt-tools/register"
import "./index.css"

// A tab opened before the WebUI was updated asks for chunk names the new
// build no longer has: load the new build (at most once a minute, so a
// server that really lacks a chunk shows the error instead of looping).
window.addEventListener("vite:preloadError", (event) => {
  const last = Number(sessionStorage.getItem("remi-chunk-reload") ?? 0)
  if (Date.now() - last < 60_000) return
  sessionStorage.setItem("remi-chunk-reload", String(Date.now()))
  event.preventDefault()
  location.reload()
})

const root = createRoot(document.getElementById("root")!)

// Load the stored/browser locale first so the shell never flashes English;
// a failure only means English, so render either way.
await activateLocale(initialLocale()).catch(() => {})

root.render(
  <StrictMode>
    <I18nextProvider i18n={i18n}>
      <QueryClientProvider client={queryClient}>
        {/* Last line of defence: a shell error shows a message, never a white page. */}
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </QueryClientProvider>
    </I18nextProvider>
  </StrictMode>,
)
