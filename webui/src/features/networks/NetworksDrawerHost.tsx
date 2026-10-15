/**
 * Generate's networks panel: a header button (Alt+N, also in the command
 * palette) toggling the drawer, whose chunk loads on first open, and the
 * prompt-focus tracking that decides where cards go.
 */
import { Layers } from "lucide-react"
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { type Command, formatShortcut, useRegisterCommands } from "@/app/commands"
import { HeaderSlot } from "@/app/header-slot"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import type { GenerateTab } from "@/features/progress/run-controller"
import { useTrackPromptFocus } from "./prompt-target"
import { DRAWER_ID, isNetworksShortcut, NETWORKS_SHORTCUT } from "./shortcut"
import { useNetworksUi } from "./store"

const NetworksDrawer = lazy(() => import("./NetworksDrawer"))

export function NetworksDrawerHost({ tab }: { tab: GenerateTab }) {
  const { t } = useTranslation()
  useTrackPromptFocus()
  const open = useNetworksUi((s) => s.drawerOpen)
  const setDrawer = useNetworksUi((s) => s.setDrawer)
  const trigger = useRef<HTMLButtonElement>(null)
  // Opened by the user (focus the search) rather than restored after a reload.
  const [userOpened, setUserOpened] = useState(false)
  const toggle = (next: boolean) => {
    setUserOpened(next)
    setDrawer(next)
  }

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!isNetworksShortcut(event) || event.repeat || event.isComposing) return
      event.preventDefault()
      const next = !useNetworksUi.getState().drawerOpen
      setUserOpened(next)
      useNetworksUi.getState().setDrawer(next)
      if (!next) trigger.current?.focus()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])

  const commands = useMemo<Command[]>(
    () => [
      {
        id: "networks.toggle",
        label: "Toggle the networks panel",
        group: "Generate",
        icon: Layers,
        keywords: ["lora", "embedding", "checkpoint", "extra networks"],
        shortcut: NETWORKS_SHORTCUT,
        run: () => {
          const next = !useNetworksUi.getState().drawerOpen
          setUserOpened(next)
          useNetworksUi.getState().setDrawer(next)
        },
      },
    ],
    [],
  )
  useRegisterCommands("networks-drawer", commands)

  const label = open ? t("Hide networks") : t("Show networks")
  return (
    <>
      <HeaderSlot>
        <Button
          ref={trigger}
          variant={open ? "secondary" : "ghost"}
          size="sm"
          aria-pressed={open}
          aria-controls={open ? DRAWER_ID : undefined}
          aria-keyshortcuts="Alt+N"
          title={`${label} (${formatShortcut(NETWORKS_SHORTCUT)})`}
          onClick={() => toggle(!open)}
          className="gap-1.5"
        >
          <Layers aria-hidden />
          <span className="hidden lg:inline">{t("Networks")}</span>
          <Kbd className="hidden xl:inline-flex">{formatShortcut(NETWORKS_SHORTCUT)}</Kbd>
        </Button>
      </HeaderSlot>
      {open ? (
        <Suspense fallback={null}>
          <NetworksDrawer tab={tab} autoFocus={userOpened} onClose={() => toggle(false)} returnFocus={() => trigger.current?.focus()} />
        </Suspense>
      ) : null}
    </>
  )
}
