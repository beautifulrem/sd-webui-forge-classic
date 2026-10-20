/** Signing in and out outside the form: error texts, the sign-out action and its palette command. */
import { type TFunction } from "i18next"
import { LogOut } from "lucide-react"
import { useMemo } from "react"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import { useSession } from "@/api/session"
import { type Command, useRegisterCommands } from "@/app/commands"

/** What a failed sign-in shows. */
export function signInError(error: unknown, t: TFunction): string {
  if (error instanceof ApiError && error.status === 429) return t("Too many attempts, wait a minute.")
  if (error instanceof ApiError && error.status === 401) return t("Wrong username or password.")
  return t("The server cannot be reached.")
}

/** Ends the session (server-side too) and shows the sign-in form. */
export async function signOut(): Promise<void> {
  await api.auth.logout().catch(() => {})
  location.reload()
}

/** "Sign out" in the command palette while signed in with a login. */
export function useSignOutCommand() {
  const user = useSession((s) => s.user)
  const commands = useMemo<Command[]>(
    () => (user ? [{ id: "auth:sign-out", label: "Sign out", group: "System", icon: LogOut, keywords: ["log out", "logout", user], run: () => void signOut() }] : []),
    [user],
  )
  useRegisterCommands("auth", commands)
}
