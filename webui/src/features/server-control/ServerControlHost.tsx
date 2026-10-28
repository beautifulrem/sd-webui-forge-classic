/**
 * Mounted once by the shell: the unsaved-changes unload guard, the palette
 * commands for "Reload UI" / "Restart server", and (lazily, only while a
 * command is being confirmed or followed) the confirm dialog and the
 * reconnecting overlay.
 */
import { RefreshCw, Power } from "@/components/icons"
import { lazy, Suspense, useEffect, useMemo } from "react"
import { type Command, useRegisterCommands } from "@/app/commands"
import { requestServerCommand, useServerControl } from "./server-control"
import { installBeforeUnloadGuard } from "./unsaved"

const ServerOverlay = lazy(() => import("./ServerOverlay"))

export function ServerControlHost() {
  const active = useServerControl((s) => s.pending !== null || s.phase !== "idle")
  useEffect(installBeforeUnloadGuard, [])
  const commands = useMemo<Command[]>(
    () => [
      { id: "server:reload", label: "Reload UI", group: "System", icon: RefreshCw, keywords: ["restart", "reload", "refresh"], run: () => requestServerCommand("reload") },
      { id: "server:restart", label: "Restart server", group: "System", icon: Power, keywords: ["restart", "reboot"], run: () => requestServerCommand("restart") },
    ],
    [],
  )
  useRegisterCommands("server-control", commands)
  return active ? (
    <Suspense fallback={null}>
      <ServerOverlay />
    </Suspense>
  ) : null
}
