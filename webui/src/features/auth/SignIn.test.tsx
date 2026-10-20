/**
 * The SPA's sign-in form (`--auth`): the server renders no login page, so
 * the app shows it while signed out, and over the app when a request is
 * refused with 401 (the app keeps its state).
 */
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { useSession } from "@/api/session"
import { recordFetch } from "@/test/fetch-recorder"
import { AuthGate } from "./SignIn"

afterEach(() => {
  useSession.setState({ state: "signed-in", user: null })
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

function fill(user: string, password: string) {
  fireEvent.change(screen.getByLabelText("Username"), { target: { value: user } })
  fireEvent.change(screen.getByLabelText("Password"), { target: { value: password } })
  fireEvent.click(screen.getByRole("button", { name: "Sign in" }))
}

describe("sign-in", () => {
  it("signed out: only the form renders, and a wrong password says so", async () => {
    useSession.setState({ state: "signed-out" })
    recordFetch((request) => (request.path === "/api/v2/auth/login" ? { status: 401, body: { detail: "wrong username or password" } } : undefined))
    render(
      <AuthGate>
        <p>the app</p>
      </AuthGate>,
    )
    expect(screen.queryByText("the app")).toBeNull()
    expect(screen.getByRole("heading", { name: "Sign in to Remi WebUI" })).toBeInTheDocument()
    fill("ann", "nope")
    expect(await screen.findByRole("alert")).toHaveTextContent("Wrong username or password.")
    expect(screen.getByLabelText("Password")).toHaveValue("")
  })

  it("says to wait when the server throttles the attempts", async () => {
    useSession.setState({ state: "signed-out" })
    recordFetch(() => ({ status: 429, body: { detail: "too many failed logins; wait a minute" } }))
    render(<AuthGate>{null}</AuthGate>)
    fill("ann", "x")
    expect(await screen.findByRole("alert")).toHaveTextContent("Too many attempts, wait a minute.")
  })

  it("signing in reloads the page that was asked for", async () => {
    useSession.setState({ state: "signed-out" })
    const reload = vi.fn()
    vi.stubGlobal("location", { ...window.location, reload, origin: window.location.origin, href: window.location.href })
    const { requests } = recordFetch()
    render(<AuthGate>{null}</AuthGate>)
    fill("ann", "s3cret")
    await waitFor(() => expect(reload).toHaveBeenCalled())
    expect(requests.at(-1)).toMatchObject({ method: "POST", path: "/api/v2/auth/login", body: { username: "ann", password: "s3cret" } })
  })

  it("a 401 while the app is open shows the form over it; signing in keeps the app", async () => {
    useSession.setState({ state: "signed-in", user: "ann" })
    let signedIn = false
    recordFetch((request) => (request.path === "/api/v2/auth/login" ? ((signedIn = true), undefined) : signedIn ? undefined : { status: 401, body: { detail: "Not authenticated" } }))
    render(
      <AuthGate>
        <p>the app</p>
      </AuthGate>,
    )
    await act(async () => {
      await api.queue.status().catch(() => {})
    })
    const dialog = await screen.findByRole("dialog", { name: "Sign in" })
    expect(dialog).toHaveTextContent("Your session has ended.")
    expect(screen.getByText("the app")).toBeInTheDocument() // still mounted, inert behind the form
    expect(screen.getByText("the app").closest("[inert]")).not.toBeNull()
    expect(screen.getByLabelText("Username")).toHaveValue("ann")
    fill("ann", "s3cret")
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    expect(useSession.getState()).toMatchObject({ state: "signed-in", user: "ann" })
    expect(screen.getByText("the app").closest("[inert]")).toBeNull()
  })

  it("a refused sign-in request is not a lost session", async () => {
    recordFetch(() => ({ status: 401, body: { detail: "wrong username or password" } }))
    await api.auth.login("ann", "x").catch(() => {})
    expect(useSession.getState().state).toBe("signed-in")
  })
})
