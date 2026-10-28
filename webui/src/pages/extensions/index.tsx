/**
 * Extensions: installed (enable, update, uninstall), available (the
 * extension index), install from a URL, and backup / restore of the commits.
 * Route: `#/extensions?tab=installed|available|url|backup`.
 */
import { useQuery } from "@tanstack/react-query"
import { Archive, Download, Link2, Package, ShieldAlert } from "@/components/icons"
import { lazy, Suspense } from "react"
import { useTranslation } from "react-i18next"
import { navigate, type Route } from "@/app/router"
import { FitTabsList } from "@/components/FitTabsList"
import { PAGE_CONTAINER } from "@/components/page-container"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsTrigger } from "@/components/ui/tabs"
import { InstalledView } from "./InstalledView"
import { installedQuery } from "./queries"

const AvailableView = lazy(() => import("./AvailableView").then((m) => ({ default: m.AvailableView })))
const InstallUrlView = lazy(() => import("./InstallUrlView").then((m) => ({ default: m.InstallUrlView })))
const BackupView = lazy(() => import("./BackupView").then((m) => ({ default: m.BackupView })))

const TABS = [
  { id: "installed", label: "Installed", icon: Package },
  { id: "available", label: "Available", icon: Download },
  { id: "url", label: "Install from URL", icon: Link2 },
  { id: "backup", label: "Backup and restore", icon: Archive },
] as const

type TabId = (typeof TABS)[number]["id"]

function parseTab(value: string | null | undefined): TabId {
  return TABS.some((tab) => tab.id === value) ? (value as TabId) : "installed"
}

function openTab(value: TabId) {
  navigate({ name: "extensions", query: value === "installed" ? undefined : new URLSearchParams({ tab: value }) }, { replace: true })
}

export default function ExtensionsPage({ route }: { route: Route }) {
  const { t } = useTranslation()
  const tab = parseTab(route.query?.get("tab"))
  const list = useQuery(installedQuery)
  const access = list.data?.access ?? true

  return (
    <div data-scroll-root className="h-full overflow-y-auto">
      <div className={PAGE_CONTAINER}>
        <Tabs value={tab} onValueChange={(value) => openTab(value as TabId)}>
          {/* on a phone the four tabs do not fit: they become a select */}
          <FitTabsList
            aria-label={t("Extensions")}
            variant="line"
            className="border-b pb-px"
            options={TABS.map(({ id, label, icon: Icon }) => ({ value: id, label: t(label), icon: <Icon aria-hidden /> }))}
            value={tab}
            onValueChange={openTab}
          >
            {TABS.map(({ id, label, icon: Icon }) => (
              <TabsTrigger key={id} value={id} className="flex-none px-2.5">
                <Icon aria-hidden />
                {t(label)}
              </TabsTrigger>
            ))}
          </FitTabsList>

          {list.data && !list.data.access ? (
            <p role="status" className="mt-2 flex items-start gap-2 rounded-lg bg-muted px-3 py-2 text-sm text-muted-foreground">
              <ShieldAlert className="mt-0.5 size-4 shrink-0" aria-hidden />
              {t("Extension management is disabled: this server is reachable from other machines. Start it with --enable-insecure-extension-access to install, update or switch extensions here.")}
            </p>
          ) : null}

          <Suspense fallback={<Skeleton className="mt-2 h-64 w-full" />}>
            <TabsContent value="installed" className="pt-2">
              <InstalledView />
            </TabsContent>
            <TabsContent value="available" className="pt-2">
              {tab === "available" ? <AvailableView /> : null}
            </TabsContent>
            <TabsContent value="url" className="pt-2">
              {tab === "url" ? <InstallUrlView access={access} /> : null}
            </TabsContent>
            <TabsContent value="backup" className="pt-2">
              {tab === "backup" ? <BackupView access={access} /> : null}
            </TabsContent>
          </Suspense>
        </Tabs>
      </div>
    </div>
  )
}
