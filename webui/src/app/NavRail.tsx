/** Slim left navigation rail (desktop) and bottom bar (mobile). */
import { Ellipsis } from "@/components/icons"
import * as m from "motion/react-m"
import { lazy, Suspense, useState } from "react"
import { useTranslation } from "react-i18next"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { useProgress } from "@/api/progress"
import { useQueuedCount } from "@/features/queue/status"
import { isActive, preloadRoute, type NavItem } from "./routes"
import { type Route, routeHref } from "./router"

const MoreSheet = lazy(() => import("./MoreSheet"))

/**
 * A sheet of animation paper (動画用紙), punched for the peg bar (タップ): a round hole between two
 * flat ones. Every drawing and cel of a scene is registered on those three holes.
 */
function Logo() {
  return (
    <a
      href={routeHref({ name: "generate", tab: "txt2img" })}
      aria-label="Remi WebUI"
      className="grid size-10 place-items-center rounded-md text-rail-foreground outline-none focus-visible:ring-3 focus-visible:ring-rail-foreground/60"
    >
      <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
        <mask id="remi-logo-holes">
          <rect width="32" height="32" fill="white" />
          <rect x="6" y="5" width="6" height="3" rx="1.5" fill="black" />
          <circle cx="16" cy="6.5" r="2" fill="black" />
          <rect x="20" y="5" width="6" height="3" rx="1.5" fill="black" />
        </mask>
        <rect x="3" y="2" width="26" height="28" rx="2" fill="currentColor" mask="url(#remi-logo-holes)" />
        <path d="M9 22c3-5 7-7 14-8" fill="none" className="stroke-rail" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
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
              "group/rail relative grid size-10 place-items-center rounded-md text-rail-muted outline-none",
              "hover:text-rail-foreground focus-visible:ring-3 focus-visible:ring-rail-foreground/60",
              active && "text-rail-foreground",
            )}
          />
        }
      >
        {/* registered on the bar: a flat peg notch on the rail's edge */}
        {/* the current page's tile moves to the page picked */}
        {active ? <m.span layoutId="rail-current" aria-hidden className="absolute inset-0 rounded-md bg-rail-active" /> : null}
        <Icon className="relative size-5" weight={active ? "fill" : "regular"} aria-hidden />
        {queued > 0 ? (
          <span
            aria-hidden
            className={cn(
              "absolute top-0.5 right-0.5 grid h-4 min-w-4 place-items-center rounded-full bg-rail-foreground px-1 text-[0.6rem] leading-none font-semibold text-rail tabular-nums ring-2 ring-rail",
            )}
          >
            {queued > 99 ? "99+" : queued}
          </span>
        ) : busy ? (
          // Static on purpose: an endless animation makes the compositor redraw at display rate for the
          // whole job, which costs sampling speed on machines whose GPU the browser shares.
          <span aria-hidden className="absolute top-2 right-2 size-1.5 rounded-full bg-rail-foreground" />
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
      className="hidden w-14 shrink-0 flex-col items-center gap-1 bg-rail py-2 md:flex"
    >
      <Logo />
      <div className="no-scrollbar mt-2 flex min-h-0 flex-1 flex-col items-center gap-1 overflow-y-auto px-2 py-1">
        {main.map((item, index) => {
          const separator = index > 0 && main[index - 1]?.group !== item.group
          return (
            <div key={item.id} className="contents">
              {separator ? <div aria-hidden className="h-3 shrink-0" /> : null}
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
      className="flex shrink-0 items-stretch gap-1 bg-rail px-2 pt-1 pb-[max(0.25rem,env(safe-area-inset-bottom))] md:hidden"
    >
      {primary.map((item) => (
        <BottomTab key={item.id} item={item} active={isActive(item, route)} onRail />
      ))}
      <button
        type="button"
        onClick={() => setMore(true)}
        aria-haspopup="dialog"
        className={cn(
          "flex flex-1 flex-col items-center gap-0.5 rounded-md px-1 py-1.5 text-[0.68rem] text-rail-muted outline-none focus-visible:ring-3 focus-visible:ring-rail-foreground/60",
          restActive && "text-rail-foreground",
        )}
      >
        <Ellipsis className="size-5" weight={restActive ? "bold" : "regular"} aria-hidden />
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

/** `onRail`: on the ink-blue bar (otherwise on a sheet's own surface). */
export function BottomTab({ item, active, onClick, onRail = false }: { item: NavItem; active: boolean; onClick?: () => void; onRail?: boolean }) {
  const { t } = useTranslation()
  const Icon = item.icon
  return (
    <a
      href={routeHref(item.route)}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={cn(
        "flex min-w-0 flex-1 flex-col items-center gap-0.5 rounded-md px-1 py-1.5 text-[0.68rem] outline-none focus-visible:ring-3",
        onRail ? "text-rail-muted focus-visible:ring-rail-foreground/60" : "text-muted-foreground focus-visible:ring-ring/60",
        active && (onRail ? "text-rail-foreground" : "text-foreground"),
      )}
    >
      <Icon className={cn("size-5", active && !onRail && "text-primary")} weight={active ? "fill" : "regular"} aria-hidden />
      <span className="max-w-full truncate">{t(item.label)}</span>
    </a>
  )
}
