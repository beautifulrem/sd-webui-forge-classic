/** Navigation entries and the lazily loaded page for each route. */
import { lazy, type ComponentType, type LazyExoticComponent } from "react"
import {
  FileSearch,
  GalleryVerticalEnd,
  ImageUpscale,
  Images,
  Layers,
  ListOrdered,
  type IconComponent,
  Puzzle,
  Settings,
  WandSparkles,
} from "@/components/icons"
import type { PageInfo } from "@/api/types"
import { pageIcon } from "./icons"
import { type Route, routeKey } from "./router"

type PageModule = { default: ComponentType<{ route: never }> }

const loaders = {
  generate: () => import("@/pages/generate"),
  extras: () => import("@/pages/extras"),
  "png-info": () => import("@/pages/png-info"),
  gallery: () => import("@/pages/gallery"),
  networks: () => import("@/pages/networks"),
  queue: () => import("@/pages/queue"),
  settings: () => import("@/pages/settings"),
  extensions: () => import("@/pages/extensions"),
  page: () => import("@/pages/ext"),
  "not-found": () => import("@/pages/not-found"),
} satisfies Record<Route["name"], () => Promise<PageModule>>

const pages = Object.fromEntries(
  Object.entries(loaders).map(([name, load]) => [name, lazy<ComponentType<{ route: never }>>(load)]),
) as Record<Route["name"], LazyExoticComponent<ComponentType<{ route: never }>>>

/** The page component for a route; every page receives its own route variant. */
export function pageFor<R extends Route>(route: R): ComponentType<{ route: R }> {
  return pages[route.name] as unknown as ComponentType<{ route: R }>
}

/** Starts downloading a page chunk early (nav hover / focus); module loads are cached. */
export function preloadRoute(route: Route) {
  void loaders[route.name]().catch(() => {})
}

export type NavGroup = "generate" | "library" | "pages" | "system"

export type NavItem = {
  /** Built-in tab id (matches `Bootstrap.tabs`) or `page:<id>`. */
  id: string
  /** English label; translated at render time. */
  label: string
  icon: IconComponent
  route: Route
  group: NavGroup
  /** Shown in the mobile bottom bar. */
  primary?: boolean
}

export const BUILTIN_NAV: NavItem[] = [
  { id: "txt2img", label: "txt2img", icon: WandSparkles, route: { name: "generate", tab: "txt2img" }, group: "generate", primary: true },
  { id: "img2img", label: "img2img", icon: Images, route: { name: "generate", tab: "img2img" }, group: "generate", primary: true },
  { id: "extras", label: "Extras", icon: ImageUpscale, route: { name: "extras" }, group: "generate" },
  { id: "png-info", label: "PNG Info", icon: FileSearch, route: { name: "png-info" }, group: "library" },
  { id: "gallery", label: "Gallery", icon: GalleryVerticalEnd, route: { name: "gallery" }, group: "library", primary: true },
  { id: "networks", label: "Networks", icon: Layers, route: { name: "networks" }, group: "library" },
  { id: "queue", label: "Queue", icon: ListOrdered, route: { name: "queue" }, group: "library", primary: true },
  { id: "extensions", label: "Extensions", icon: Puzzle, route: { name: "extensions" }, group: "system" },
  { id: "settings", label: "Settings", icon: Settings, route: { name: "settings" }, group: "system" },
]

/** Tabs that are always shown, whatever `Bootstrap.tabs` says. */
const ALWAYS_SHOWN = new Set(["queue", "settings", "extensions"])

/**
 * Built-in entries filtered and ordered by the server's tab list (when given),
 * followed by extension pages.
 */
export function buildNav(tabs: readonly string[] | undefined, extPages: readonly PageInfo[] | undefined): NavItem[] {
  let builtin = BUILTIN_NAV
  if (tabs && tabs.length > 0) {
    const order = new Map(tabs.map((id, index) => [id, index]))
    const rank = (item: NavItem) => order.get(item.id) ?? Number.MAX_SAFE_INTEGER
    builtin = BUILTIN_NAV.filter((item) => order.has(item.id) || ALWAYS_SHOWN.has(item.id)).sort(
      (a, b) => rank(a) - rank(b),
    )
  }
  const extensions: NavItem[] = (extPages ?? []).map((page) => ({
    id: `page:${page.id}`,
    label: page.title,
    icon: pageIcon(page.icon),
    route: { name: "page", id: page.id },
    group: "pages",
  }))
  const system = builtin.filter((item) => item.group === "system")
  return [...builtin.filter((item) => item.group !== "system"), ...extensions, ...system]
}

export function isActive(item: NavItem, current: Route) {
  return routeKey(item.route) === routeKey(current)
}
