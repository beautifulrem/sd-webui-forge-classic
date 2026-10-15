/**
 * Extensions: installed (enable, update, uninstall), available (the
 * extension index), install from a URL, and backup / restore of the commits.
 * Route: `#/extensions?tab=installed|available|url|backup`.
 */
import { useQuery } from "@tanstack/react-query"
import { Archive, Download, Link2, Package, ShieldAlert } from "lucide-react"
import { lazy, Suspense } from "react"
import { useTranslation } from "react-i18next"
import { navigate, type Route } from "@/app/router"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
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

export default function ExtensionsPage({ route }: { route: Route }) {
  const { t } = useTranslation()
  const tab = parseTab(route.query?.get("tab"))
  const list = useQuery(installedQuery)
  const access = list.data?.access ?? true

  return (
    <div data-scroll-root className="h-full overflow-y-auto">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-4 px-4 py-5 sm:px-6">
        <Tabs
          value={tab}
          onValueChange={(value) => navigate({ name: "extensions", query: value === "installed" ? undefined : new URLSearchParams({ tab: String(value) }) }, { replace: true })}
        >
          <TabsList variant="line" className="w-full justify-start overflow-x-auto border-b pb-px">
            {TABS.map(({ id, label, icon: Icon }) => (
              <TabsTrigger key={id} value={id} className="flex-none px-2.5">
                <Icon aria-hidden />
                {t(label)}
              </TabsTrigger>
            ))}
          </TabsList>

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
