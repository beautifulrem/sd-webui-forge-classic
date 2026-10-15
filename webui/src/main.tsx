import { QueryClientProvider } from "@tanstack/react-query"
import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { I18nextProvider } from "react-i18next"
import { queryClient } from "@/api/queries"
import { App } from "@/app/App"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { activateLocale, i18n, initialLocale } from "@/i18n"
import "./index.css"

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
