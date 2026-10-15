/** Shared helpers for the mock-mode end-to-end tests. */
import { expect, type Page } from "@playwright/test"

export const mod = process.platform === "darwin" ? "Meta" : "Control"

export type Theme = "light" | "dark"

/** Persists an appearance (and optionally a locale) before the app boots, as the user's stored choice. */
export async function presetStorage(page: Page, options: { theme?: Theme; locale?: string } = {}) {
  await page.addInitScript(({ theme, locale }) => {
    if (sessionStorage.getItem("remi-e2e-preset")) return // only on the first load: later reloads keep what the test changed
    sessionStorage.setItem("remi-e2e-preset", "1")
    if (theme) localStorage.setItem("remi-appearance", JSON.stringify({ state: { theme, density: "comfortable", accent: "iris" }, version: 1 }))
    if (locale) localStorage.setItem("remi-locale", JSON.stringify({ state: { chosen: locale }, version: 1 }))
  }, options)
}

/** Opens a route and waits until its page chunk rendered (the main landmark has content and nothing is loading). */
export async function openRoute(page: Page, hash: string) {
  await page.goto(`/#/${hash}`)
  const main = page.getByRole("main")
  await expect(main).toBeVisible()
  await expect(main.locator("[aria-busy=true]")).toHaveCount(0)
  await expect(main.locator(":scope *").first()).toBeVisible()
}

/** Opens a generate tab and waits for its prompt editor. */
export async function openGenerate(page: Page, tab: "txt2img" | "img2img" = "txt2img") {
  await page.goto(`/#/${tab}`)
  await expect(page.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible()
}

/** Replaces the prompt editor's text. */
export async function setPrompt(page: Page, text: string) {
  const prompt = page.getByRole("textbox", { name: "Prompt", exact: true })
  await prompt.click()
  await page.keyboard.press(`${mod}+A`)
  await page.keyboard.type(text)
  return prompt
}

/** Collects console errors and uncaught page errors; call `assertClean()` at the end of a test. */
export function watchConsole(page: Page) {
  const problems: string[] = []
  page.on("console", (message) => {
    if (message.type() === "error") problems.push(`console.error: ${message.text()}`)
  })
  page.on("pageerror", (error) => problems.push(`pageerror: ${error.message}`))
  return {
    problems,
    assertClean() {
      expect(problems, problems.join("\n")).toEqual([])
    },
  }
}
