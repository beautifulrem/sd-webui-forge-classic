/** The application shell: rail + header + lazily loaded page, plus global services. */
import { useQuery } from "@tanstack/react-query"
import { Keyboard, Languages, Monitor, Moon, Rows3, Rows4, Sun } from "lucide-react"
import { lazy, Suspense, useEffect, useMemo } from "react"
import { useTranslation } from "react-i18next"
import { bootstrapQuery } from "@/api/queries"
import { ErrorBoundary } from "@/components/ErrorBoundary"
import { PageSkeleton } from "@/components/PageSkeleton"
import { Toaster } from "@/components/ui/sonner"
import { ServerControlHost } from "@/features/server-control/ServerControlHost"
import { TooltipProvider } from "@/components/ui/tooltip"
import { activateLocale, browserLocale, useLocale } from "@/i18n"
import { useAppearance, useApplyAppearance } from "@/lib/theme"
import { type Command, useCommands, useRegisterCommands } from "./commands"
import { Header } from "./Header"
import { availableLocales } from "./locales"
import { BottomBar, NavRail } from "./NavRail"
import { buildNav, pageFor, type NavItem } from "./routes"
import { onShortcutsKey, useShortcutsHelp } from "./shortcuts-help"
import { navigate, type Route, routeKey, useRoute } from "./router"
import { useWsBridge } from "./ws-bridge"

const CommandPalette = lazy(() => import("./CommandPalette"))
const ShortcutsDialog = lazy(() => import("./ShortcutsDialog"))

function useGlobalShortcuts() {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === "k") {
        event.preventDefault()
        const { palette, setPalette } = useCommands.getState()
        setPalette(!palette)
      } else onShortcutsKey(event)
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])
}

function useShellCommands(nav: NavItem[], locales: ReturnType<typeof availableLocales>) {
  const setTheme = useAppearance((s) => s.setTheme)
  const setDensity = useAppearance((s) => s.setDensity)
  const setLocale = useLocale((s) => s.setLocale)
  const commands = useMemo<Command[]>(() => {
    return [
      ...nav.map((item) => ({
        id: `nav:${item.id}`,
        label: item.label,
        group: "Go to",
        icon: item.icon,
        run: () => navigate(item.route),
      })),
      { id: "help:shortcuts", label: "Keyboard shortcuts", group: "Help", icon: Keyboard, shortcut: "?", keywords: ["keys", "hotkeys"], run: () => useShortcutsHelp.getState().setOpen(true) },
      { id: "theme:light", label: "Light theme", group: "Appearance", icon: Sun, run: () => setTheme("light") },
      { id: "theme:dark", label: "Dark theme", group: "Appearance", icon: Moon, run: () => setTheme("dark") },
      { id: "theme:system", label: "System theme", group: "Appearance", icon: Monitor, run: () => setTheme("system") },
      { id: "density:comfortable", label: "Comfortable density", group: "Appearance", icon: Rows3, run: () => setDensity("comfortable") },
      { id: "density:compact", label: "Compact density", group: "Appearance", icon: Rows4, run: () => setDensity("compact") },
      ...locales.map((locale) => ({
        id: `locale:${locale.id}`,
        label: locale.name,
        group: "Language",
        icon: Languages,
        keywords: [locale.id, "language"],
        run: () => setLocale(locale.id),
      })),
    ]
  }, [nav, locales, setTheme, setDensity, setLocale])
  useRegisterCommands("shell", commands)
}

function routeTitle(route: Route, nav: NavItem[]): string {
  if (route.name === "generate") return "Generate"
  if (route.name === "not-found") return "Not found"
  return nav.find((item) => routeKey(item.route) === routeKey(route))?.label ?? route.name
}

export function App() {
  const { t } = useTranslation()
  const route = useRoute()
  const bootstrap = useQuery(bootstrapQuery)
  const chosen = useLocale((s) => s.chosen)
  const paletteLoaded = useCommands((s) => s.paletteLoaded)
  const shortcutsLoaded = useShortcutsHelp((s) => s.loaded)

  useApplyAppearance()
  useWsBridge()
  useGlobalShortcuts()

  const serverLocale = bootstrap.data?.locale
  useEffect(() => {
    void activateLocale(chosen ?? serverLocale ?? browserLocale())
  }, [chosen, serverLocale])

  const nav = useMemo(() => buildNav(bootstrap.data?.tabs, bootstrap.data?.pages), [bootstrap.data])
  const locales = useMemo(() => availableLocales(bootstrap.data?.locales), [bootstrap.data])
  useShellCommands(nav, locales)

  const title = t(routeTitle(route, nav))
  useEffect(() => {
    document.title = `${title} · Remi`
  }, [title])

  const Page = pageFor(route)

  return (
    <TooltipProvider delay={400}>
      <a
        href="#main"
        onClick={(event) => {
          event.preventDefault()
          document.getElementById("main")?.focus()
        }}
        className="sr-only z-50 rounded-md bg-primary px-3 py-2 text-primary-foreground focus:not-sr-only focus:fixed focus:top-2 focus:left-2"
      >
        {t("Skip to content")}
      </a>
      <div className="flex h-dvh overflow-hidden">
        <NavRail items={nav} route={route} />
        <div className="flex min-w-0 flex-1 flex-col">
          <Header title={title} locales={locales} />
          <main id="main" tabIndex={-1} className="relative min-h-0 flex-1 overflow-clip outline-none">
            <ErrorBoundary resetKey={routeKey(route)}>
              <Suspense fallback={<PageSkeleton />}>
                {/* oxlint-disable-next-line react/static-components -- a stable lazy component from the route table */}
                <Page route={route} />
              </Suspense>
            </ErrorBoundary>
          </main>
          <BottomBar items={nav} route={route} />
        </div>
      </div>
      {paletteLoaded ? (
        <Suspense fallback={null}>
          <CommandPalette />
        </Suspense>
      ) : null}
      {shortcutsLoaded ? (
        <Suspense fallback={null}>
          <ShortcutsDialog />
        </Suspense>
      ) : null}
      <Toaster position="bottom-right" closeButton richColors={false} />
      <ServerControlHost />
    </TooltipProvider>
  )
}
