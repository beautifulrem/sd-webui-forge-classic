/** The settings page's views: the settings list and the system pages. */
import { FileText, Info, type IconComponent, Server } from "@/components/icons"

export type SettingsView = "settings" | "server" | "sysinfo" | "licenses"

export const SYSTEM_VIEWS: { id: Exclude<SettingsView, "settings">; label: string; icon: IconComponent }[] = [
  { id: "server", label: "Server and models", icon: Server },
  { id: "sysinfo", label: "System info", icon: Info },
  { id: "licenses", label: "Licenses", icon: FileText },
]
