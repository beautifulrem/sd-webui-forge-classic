/**
 * Signing in with `--auth`, against the mock login (`api/mock/auth.ts`,
 * account demo / demo): the server renders no login page, the SPA shows its
 * own form (translated like the rest of the app), over the app when the
 * session ends while it is open.
 */
import AxeBuilder from "@axe-core/playwright"
import { expect, type Page, test } from "@playwright/test"
import { openGenerate, presetStorage, type Theme, watchConsole } from "./helpers"

async function requireLogin(page: Page) {
  await page.addInitScript(() => sessionStorage.setItem("remi-mock-login", "1"))
}

async function signIn(page: Page, user: string, password: string) {
  await page.getByLabel("Username").fill(user)
  await page.getByLabel("Password").fill(password)
  await page.getByRole("button", { name: "Sign in" }).click()
}

test.describe("sign-in", () => {
  test("signed out only the form shows; signing in opens the app; signing out ends it", async ({ page }) => {
    const console = watchConsole(page)
    await requireLogin(page)
    await page.goto("/#/txt2img")
    await expect(page.getByRole("heading", { name: "Sign in to Remi WebUI" })).toBeVisible()
    await expect(page).toHaveTitle("Sign in · Remi")
    await expect(page.getByRole("navigation")).toHaveCount(0)
    await expect(page.getByLabel("Username")).toBeFocused()

    await signIn(page, "demo", "wrong")
    await expect(page.getByRole("alert")).toHaveText("Wrong username or password.")
    await expect(page.getByLabel("Password")).toHaveValue("")

    await signIn(page, "demo", "demo")
    await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible()
    await page.getByRole("button", { name: "Appearance" }).click()
    await page.getByRole("menuitem", { name: "Sign out (demo)" }).click()
    await expect(page.getByRole("heading", { name: "Sign in to Remi WebUI" })).toBeVisible()
    console.assertClean()
  })

  test("the form follows the language", async ({ page }) => {
    await presetStorage(page, { locale: "zh-CN" })
    await requireLogin(page)
    await page.goto("/")
    await expect(page.getByRole("heading", { name: "登录 Remi WebUI" })).toBeVisible()
    await expect(page.getByLabel("用户名")).toBeVisible()
    await expect(page.getByLabel("密码")).toBeVisible()
    await expect(page.locator("html")).toHaveAttribute("lang", "zh-CN")
  })

  test("a session that ends while the app is open asks again and keeps the form's state", async ({ page }) => {
    await requireLogin(page)
    await page.addInitScript(() => {
      if (!sessionStorage.getItem("remi-e2e-signed")) {
        sessionStorage.setItem("remi-e2e-signed", "1")
        sessionStorage.setItem("remi-mock-session", "demo")
      }
    })
    await openGenerate(page)
    const prompt = page.getByRole("textbox", { name: "Prompt", exact: true })
    await prompt.click()
    await page.keyboard.type("a lighthouse at dusk")

    // the session ends (e.g. revoked elsewhere); the next request is refused
    await page.evaluate(() => sessionStorage.removeItem("remi-mock-session"))
    await page.evaluate(() => (location.hash = "#/gallery"))
    const dialog = page.getByRole("dialog", { name: "Sign in" })
    await expect(dialog).toBeVisible()
    await expect(dialog).toContainText("Your session has ended.")
    await expect(page.getByLabel("Username")).toHaveValue("demo")

    await signIn(page, "demo", "demo")
    await expect(dialog).toHaveCount(0)
    await page.evaluate(() => (location.hash = "#/txt2img"))
    await expect(prompt).toContainText("a lighthouse at dusk")
  })

  for (const theme of ["light", "dark"] satisfies Theme[]) {
    test(`the form has no serious or critical axe violations (${theme})`, async ({ page }) => {
      await presetStorage(page, { theme })
      await requireLogin(page)
      await page.goto("/")
      await signIn(page, "demo", "wrong")
      await expect(page.getByRole("alert")).toHaveText("Wrong username or password.")
      const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa", "best-practice"]).analyze()
      const serious = results.violations.filter((violation) => violation.impact === "serious" || violation.impact === "critical").map((violation) => `${violation.id}: ${violation.nodes.map((node) => node.target.join(" ")).join(", ")}`)
      expect(serious).toEqual([])
    })
  }
})
