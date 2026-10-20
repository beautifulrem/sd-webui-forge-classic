/** Slim left navigation rail (desktop) and bottom bar (mobile). */
import { Ellipsis } from "lucide-react"
import { lazy, Suspense, useState } from "react"
import { useTranslation } from "react-i18next"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { useProgress } from "@/api/progress"
import { useQueuedCount } from "@/features/queue/status"
import { isActive, preloadRoute, type NavItem } from "./routes"
import { type Route, routeHref } from "./router"

const MoreSheet = lazy(() => import("./MoreSheet"))

function Logo() {
  return (
    <a
      href={routeHref({ name: "generate", tab: "txt2img" })}
      aria-label="Remi WebUI"
      className="grid size-9 place-items-center rounded-xl bg-gradient-to-br from-primary to-[oklch(from_var(--primary)_calc(l-0.12)_c_calc(h+40))] font-heading text-base font-semibold text-primary-foreground shadow-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/60"
    >
      R
    </a>
  )
}

function RailLink({ item, active }: { item: NavItem; active: boolean }) {
  const { t } = useTranslation()
  const busy = useProgress((s) => item.id === "queue" && s.running !== null)
  const queued = useQueuedCount(item.id === "queue")
  const Icon = item.icon
  const label = t(item.label)
  const described = queued > 0 ? t("{{label}}, {{count}} waiting", { label, count: queued }) : label
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <a
            href={routeHref(item.route)}
            aria-label={described}
            aria-current={active ? "page" : undefined}
            onPointerEnter={() => preloadRoute(item.route)}
            onFocus={() => preloadRoute(item.route)}
            className={cn(
              "group/rail relative grid size-10 place-items-center rounded-lg text-muted-foreground outline-none transition-colors",
              "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground focus-visible:ring-3 focus-visible:ring-ring/60",
              active && "bg-sidebar-accent text-sidebar-accent-foreground",
            )}
          />
        }
      >
        <span
          aria-hidden
          className={cn(
            "absolute top-1/2 -left-2 h-5 w-[3px] -translate-y-1/2 rounded-r-full bg-primary transition-[opacity,height] duration-200",
            active ? "opacity-100" : "h-2 opacity-0",
          )}
        />
        <Icon className="size-[1.15rem]" strokeWidth={active ? 2.2 : 1.8} aria-hidden />
        {queued > 0 ? (
          <span
            aria-hidden
            className={cn(
              "absolute top-0.5 right-0.5 grid h-4 min-w-4 place-items-center rounded-full bg-primary px-1 text-[0.6rem] leading-none font-semibold text-primary-foreground tabular-nums ring-2 ring-sidebar",
            )}
          >
            {queued > 99 ? "99+" : queued}
          </span>
        ) : busy ? (
          // Static on purpose: an endless animation makes the compositor redraw at display rate for the
          // whole job, which costs sampling speed on machines whose GPU the browser shares.
          <span aria-hidden className="absolute top-2 right-2 size-1.5 rounded-full bg-primary" />
        ) : null}
      </TooltipTrigger>
      <TooltipContent side="right" sideOffset={8}>
        {described}
      </TooltipContent>
    </Tooltip>
  )
}

export function NavRail({ items, route }: { items: NavItem[]; route: Route }) {
  const { t } = useTranslation()
  const main = items.filter((item) => item.group !== "system")
  const system = items.filter((item) => item.group === "system")

  return (
    <nav
      aria-label={t("Main navigation")}
      className="hidden w-14 shrink-0 flex-col items-center gap-1 border-r bg-sidebar py-2.5 md:flex"
    >
      <Logo />
      <div className="no-scrollbar mt-2 flex min-h-0 flex-1 flex-col items-center gap-1 overflow-y-auto px-2 py-1">
        {main.map((item, index) => {
          const separator = index > 0 && main[index - 1]?.group !== item.group
          return (
            <div key={item.id} className="contents">
              {separator ? <div aria-hidden className="my-1.5 h-px w-6 shrink-0 bg-border" /> : null}
              <RailLink item={item} active={isActive(item, route)} />
            </div>
          )
        })}
      </div>
      <div className="flex flex-col items-center gap-1 px-2">
        {system.map((item) => (
          <RailLink key={item.id} item={item} active={isActive(item, route)} />
        ))}
      </div>
    </nav>
  )
}

/** Mobile: primary destinations in a bottom bar, the rest in a sheet. */
export function BottomBar({ items, route }: { items: NavItem[]; route: Route }) {
  const { t } = useTranslation()
  const [more, setMore] = useState(false)
  const primary = items.filter((item) => item.primary)
  const rest = items.filter((item) => !item.primary)
  const restActive = rest.some((item) => isActive(item, route))

  return (
    <nav
      aria-label={t("Main navigation")}
      data-toast-obstacle
      className="flex shrink-0 items-stretch gap-1 border-t bg-sidebar px-2 pt-1 pb-[max(0.25rem,env(safe-area-inset-bottom))] md:hidden"
    >
      {primary.map((item) => (
        <BottomTab key={item.id} item={item} active={isActive(item, route)} />
      ))}
      <button
        type="button"
        onClick={() => setMore(true)}
        aria-haspopup="dialog"
        className={cn(
          "flex flex-1 flex-col items-center gap-0.5 rounded-lg px-1 py-1.5 text-[0.68rem] text-muted-foreground outline-none focus-visible:ring-3 focus-visible:ring-ring/60",
          restActive && "text-foreground",
        )}
      >
        <Ellipsis className={cn("size-5", restActive && "text-primary")} aria-hidden />
        <span>{t("More")}</span>
      </button>
      {more ? (
        <Suspense fallback={null}>
          <MoreSheet items={rest} route={route} onClose={() => setMore(false)} />
        </Suspense>
      ) : null}
    </nav>
  )
}

export function BottomTab({ item, active, onClick }: { item: NavItem; active: boolean; onClick?: () => void }) {
  const { t } = useTranslation()
  const Icon = item.icon
  return (
    <a
      href={routeHref(item.route)}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={cn(
        "flex min-w-0 flex-1 flex-col items-center gap-0.5 rounded-lg px-1 py-1.5 text-[0.68rem] text-muted-foreground outline-none",
        "focus-visible:ring-3 focus-visible:ring-ring/60",
        active && "text-foreground",
      )}
    >
      <Icon className={cn("size-5", active && "text-primary")} aria-hidden />
      <span className="max-w-full truncate">{t(item.label)}</span>
    </a>
  )
}
