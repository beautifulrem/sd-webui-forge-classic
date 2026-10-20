/**
 * The sign-in form (`--auth` / `--auth-file`). The server renders no login
 * page of its own: without a session a page load gets the SPA shell and
 * `AuthGate` shows this form, alone while signed out, over the app (which
 * keeps its state) once its session ended.
 */
import { LogIn } from "lucide-react"
import { type FormEvent, type ReactNode, useEffect, useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import { useSession } from "@/api/session"
import { useWs } from "@/api/ws"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Spinner } from "@/components/ui/spinner"
import { signInError } from "./session-actions"

function SignInForm({ expired, onSignedIn }: { expired: boolean; onSignedIn: (user: string) => void }) {
  const { t } = useTranslation()
  const id = useId()
  const [username, setUsername] = useState(() => useSession.getState().user ?? "")
  const [password, setPassword] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await api.auth.login(username, password)
      onSignedIn(username)
    } catch (failure) {
      setError(signInError(failure, t))
      setPassword("")
      setBusy(false)
    }
  }

  return (
    <form
      onSubmit={(event) => void submit(event)}
      aria-labelledby={`${id}-title`}
      aria-describedby={`${id}-description`}
      className="grid w-full max-w-sm gap-4 rounded-xl bg-card p-6 text-card-foreground shadow-lg ring-1 ring-foreground/10"
    >
      <div className="grid gap-1">
        <h1 id={`${id}-title`} className="font-heading text-lg font-semibold tracking-tight">
          {t("Sign in to Remi WebUI")}
        </h1>
        <p id={`${id}-description`} className="text-sm text-muted-foreground">
          {expired ? t("Your session has ended. Sign in again to continue where you left off.") : t("This server needs a login.")}
        </p>
      </div>
      <div className="grid gap-2">
        <Label htmlFor={`${id}-user`}>{t("Username")}</Label>
        <Input
          id={`${id}-user`}
          name="username"
          autoComplete="username"
          autoCapitalize="none"
          spellCheck={false}
          required
          autoFocus={!username}
          value={username}
          onChange={(event) => setUsername(event.target.value)}
        />
      </div>
      <div className="grid gap-2">
        <Label htmlFor={`${id}-password`}>{t("Password")}</Label>
        <Input
          id={`${id}-password`}
          name="password"
          type="password"
          autoComplete="current-password"
          required
          autoFocus={Boolean(username)}
          aria-invalid={error ? true : undefined}
          aria-errormessage={error ? `${id}-error` : undefined}
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
      </div>
      <p id={`${id}-error`} role="alert" className="min-h-5 text-sm text-destructive">
        {error}
      </p>
      <Button type="submit" disabled={busy}>
        {busy ? <Spinner /> : <LogIn aria-hidden />}
        {t("Sign in")}
      </Button>
    </form>
  )
}

/** Signed out: only the form (the app is not mounted, so it requests nothing). */
export function SignInScreen() {
  const { t } = useTranslation()
  useEffect(() => {
    document.title = `${t("Sign in")} · Remi`
  }, [t])
  return (
    <main className="grid min-h-dvh place-items-center bg-background p-4">
      {/* reload: the page that was asked for (the app, or the API docs) now loads with the session */}
      <SignInForm expired={false} onSignedIn={() => location.reload()} />
    </main>
  )
}

/** The session ended while the app was open: the form over it, then everything is refetched. */
export function SignInDialog() {
  const { t } = useTranslation()
  return (
    <div role="dialog" aria-modal="true" aria-label={t("Sign in")} className="fixed inset-0 z-[100] grid place-items-center bg-background/80 p-4 backdrop-blur-sm">
      <SignInForm
        expired
        onSignedIn={(user) => {
          useSession.setState({ state: "signed-in", user })
          void queryClient.invalidateQueries()
          useWs.getState().connect()
        }}
      />
    </div>
  )
}

/** Renders the app when signed in (or no login is needed), else the sign-in form. */
export function AuthGate({ children }: { children: ReactNode }) {
  const state = useSession((s) => s.state)
  if (state === "signed-out") return <SignInScreen />
  return (
    <>
      {/* inert: nothing behind the form takes focus or clicks while it is open */}
      <div className="contents" inert={state === "expired"}>
        {children}
      </div>
      {state === "expired" ? <SignInDialog /> : null}
    </>
  )
}
