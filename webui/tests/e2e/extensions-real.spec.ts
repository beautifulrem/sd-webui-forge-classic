/**
 * Extension management and the browser features that need no model, against
 * a real server with a local git host and extension index:
 * `tools/smoke_extensions.py --browser` starts the fixture and the server
 * (an empty models folder, `WEBUI_EXTENSIONS_INDEX` at the fixture) and runs
 * these tests (`@fixture`; they never run against the mock, whose own flows
 * are in `extensions.spec.ts`). They run in order and share the server: the
 * first installs what the later ones use.
 */
import { expect, type Page, test } from "@playwright/test"
import { overrideBootstrap, REAL_SERVER, watchConsole as watchAllConsole } from "./helpers"

type Fixture = { git: string; repos: Record<string, string> }
const FIXTURE: Fixture | null = process.env.REMI_E2E_EXT_FIXTURE ? (JSON.parse(process.env.REMI_E2E_EXT_FIXTURE) as Fixture) : null
const EXT = "remi-smoke-ext"
const INDEX_EXT = "remi-smoke-index-ext"
const INDEX_NAME = "Remi smoke index extension"
/** Git over loopback, an in-process Reload UI: generous, still far below a model load. */
const SLOW = 120_000

type ExtensionInfo = { name: string; commit_hash: string | null; can_update: boolean }

test.describe.configure({ mode: "serial" })
test.skip(!FIXTURE || !REAL_SERVER, "needs the git fixture of tools/smoke_extensions.py")

/**
 * Console errors and page errors, without the browser's own network log of
 * the requests refused while the server reloads (the app retries them; a
 * reload over a real server always has that gap, the mock's has none).
 */
function watchConsole(page: Page) {
  const watched = watchAllConsole(page)
  return {
    assertClean() {
      const refused = /^console\.error: (Failed to load resource: net::ERR_(CONNECTION_REFUSED|CONNECTION_RESET|EMPTY_RESPONSE)|WebSocket connection to '.*' failed: .*net::ERR_(CONNECTION_REFUSED|CONNECTION_RESET))/
      const problems = watched.problems.filter((problem) => !refused.test(problem))
      expect(problems, problems.join("\n")).toEqual([])
    },
  }
}

async function installed(page: Page, name: string): Promise<ExtensionInfo | undefined> {
  const response = await page.request.get("/api/v2/extensions")
  const list = (await response.json()) as { items: ExtensionInfo[] }
  return list.items.find((item) => item.name === name)
}

/** Waits for the reload a server command started: the reconnect overlay, then a fresh page. */
async function waitForReload(page: Page, trigger: () => Promise<void>) {
  await page.evaluate(() => Object.assign(window, { __beforeReload: true }))
  await trigger()
  await expect(page.getByRole("alertdialog", { name: "Reloading the UI…" })).toBeVisible()
  await expect.poll(() => page.evaluate(() => "__beforeReload" in window).catch(() => true), { timeout: SLOW }).toBe(false)
  await expect(page.getByRole("main")).toBeVisible({ timeout: SLOW })
}

test.describe("extension management on a real server", () => {
  test("installs from a URL and from the index, then loads them with Reload UI @fixture", async ({ page }) => {
    const console = watchConsole(page)
    await page.goto("/#/extensions?tab=url")
    const main = page.getByRole("main")
    await main.getByRole("textbox", { name: "Repository URL" }).fill(FIXTURE!.repos[EXT]!)
    await expect(main.getByRole("textbox", { name: "Folder name" })).toHaveAttribute("placeholder", EXT)
    await main.getByRole("button", { name: "Install" }).click()
    await expect(main.getByText(`${EXT} is installed.`)).toBeVisible({ timeout: SLOW })

    await page.getByRole("tab", { name: "Available" }).click()
    await expect(main.getByRole("textbox", { name: "Extension index URL" })).toHaveValue(`${FIXTURE!.git}/index.json`)
    await main.getByRole("button", { name: /^(Load|Reload) index$/ }).click()
    const list = main.getByRole("list", { name: "Available extensions" })
    // The installed one is hidden by the default tag filter; the other is offered.
    await list.getByRole("button", { name: `Install ${INDEX_NAME}`, exact: true }).click()
    await expect(page.getByText(`${INDEX_EXT} is installed`, { exact: true })).toBeVisible({ timeout: SLOW }) // its folder name
    await expect(list.getByRole("button", { name: `Install ${INDEX_NAME}`, exact: true })).toHaveCount(0)
    expect(await installed(page, INDEX_EXT)).toBeTruthy()

    await page.getByRole("button", { name: "Reload UI" }).first().click()
    const confirm = page.getByRole("alertdialog", { name: "Reload UI" })
    await waitForReload(page, () => confirm.getByRole("button", { name: "Reload UI" }).click())
    // Loaded: the right-to-left catalog the extension ships is a language now.
    const app = (await (await page.request.get("/api/v2/app")).json()) as { locales: { id: string }[] }
    expect(app.locales.map((locale) => locale.id)).toContain("ar")
    console.assertClean()
  })

  test("a fresh backup matches now; Check for updates shows its progress; the update and Restore move the commit @fixture", async ({ page }) => {
    const console = watchConsole(page)
    const first = (await installed(page, EXT))!.commit_hash!
    await page.goto("/#/extensions?tab=backup")
    const main = page.getByRole("main")
    const name = `smoke-ui-${Date.now()}`
    await main.getByRole("textbox", { name: "Backup name" }).fill(name)
    await main.getByRole("button", { name: "Back up" }).click()
    const backup = main.getByRole("listbox", { name: "Backups" }).getByRole("option", { name: new RegExp(name) })
    await backup.click()
    await expect(main.getByText("0 extensions differ from now")).toBeVisible()

    // The remote moves on.
    const advanced = (await (await page.request.post(`${FIXTURE!.git}/__advance/${EXT}`)).json()) as { commit: string }
    // Slow each check down a little so the progress can be seen (the checks still run on the server).
    await page.route(/\/api\/v2\/extensions\/[^/]+\/check$/, async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1_500))
      await route.continue()
    })
    await page.getByRole("tab", { name: "Installed" }).click()
    await main.getByRole("button", { name: "Check for updates" }).click()
    await expect(main.getByRole("progressbar", { name: "Checking for updates" })).toBeVisible()
    await expect(main.getByText(/^Checked \d+ of 2/)).toBeVisible()
    await expect(page.getByText("1 updates available")).toBeVisible({ timeout: SLOW })
    await expect(main.getByText("Update available")).toHaveCount(1)

    await main.getByRole("checkbox", { name: "Update" }).check()
    const pending = page.getByRole("region", { name: "Pending extension changes" })
    await expect(pending).toContainText("0 switched, 1 to update")
    await waitForReload(page, () => pending.getByRole("button", { name: "Apply and reload UI" }).click())
    expect((await installed(page, EXT))?.commit_hash).toBe(advanced.commit)

    // The backup now shows the difference, and restores the backed-up commit.
    await page.goto("/#/extensions?tab=backup")
    await main.getByRole("listbox", { name: "Backups" }).getByRole("option", { name: new RegExp(name) }).click()
    await expect(main.getByText(/^1 extensions? differs? from now$/)).toBeVisible()
    await main.getByRole("button", { name: "Restore extensions" }).click()
    const confirm = page.getByRole("alertdialog", { name: "Restore the extensions of this backup?" })
    await waitForReload(page, () => confirm.getByRole("button", { name: "Restore extensions" }).click())
    await expect.poll(async () => (await installed(page, EXT))?.commit_hash, { timeout: SLOW }).toBe(first)

    await page.goto("/#/extensions?tab=backup")
    await main.getByRole("listbox", { name: "Backups" }).getByRole("option", { name: new RegExp(name) }).click()
    await expect(main.getByText("0 extensions differ from now")).toBeVisible()
    await main.getByRole("button", { name: "Delete this backup" }).click()
    await page.getByRole("alertdialog", { name: "Delete this backup?" }).getByRole("button", { name: "Delete" }).click()
    await expect(main.getByRole("option", { name: new RegExp(name) })).toHaveCount(0)
    console.assertClean()
  })

  test("lays the app out right to left in the language an extension ships @fixture", async ({ page }) => {
    await page.goto("/#/txt2img")
    await expect(page.getByRole("main")).toBeVisible()
    await page.getByRole("button", { name: "Appearance" }).click()
    await page.getByRole("menuitemradio", { name: "العربية" }).click()
    const html = page.locator("html")
    await expect(html).toHaveAttribute("dir", "rtl")
    await expect(html).toHaveAttribute("lang", "ar")
    const nav = page.getByRole("navigation", { name: "Main navigation" })
    // The extension's catalog translates the shell, and the navigation rail moves to the right.
    await expect(nav.getByRole("link", { name: "الإعدادات" })).toBeVisible()
    const rail = await nav.boundingBox()
    const main = await page.getByRole("main").boundingBox()
    expect(rail!.x).toBeGreaterThan(main!.x)
    // Back to English (the choice is the browser's, not the server's).
    // The language items keep the menu open; reopen it only if it closed.
    const english = page.getByRole("menuitemradio", { name: "English" })
    if (!(await english.isVisible())) await page.getByRole("button", { name: "Appearance" }).click() // not in the extension's small catalog
    await english.click()
    await expect(html).toHaveAttribute("dir", "ltr")
  })
})

test.describe("browser features on a real server, no model", () => {
  test("steps through the gallery in the viewer with a sideways wheel and a gamepad @fixture", async ({ page }) => {
    await page.addInitScript(() => {
      const state = { pad: null as { axes: number[]; buttons: { pressed: boolean }[] } | null, polls: 0 }
      Object.assign(window, { __pad: state })
      Object.defineProperty(navigator, "getGamepads", {
        configurable: true,
        value: () => {
          if (!state.pad || state.polls <= 0) return [null]
          state.polls--
          return [{ ...state.pad, connected: true, mapping: "standard", index: 0, id: "probe" }]
        },
      })
    })
    await overrideBootstrap(page, { options: { js_modal_lightbox_gamepad: true, js_modal_lightbox_gamepad_repeat: 250 } })
    await page.goto("/#/gallery")
    const grid = page.getByRole("list", { name: "Images" })
    await grid.getByRole("button", { name: /^smoke image 1/ }).first().click({ timeout: SLOW })
    const viewer = page.getByRole("dialog", { name: "Image viewer" })
    const position = viewer.getByText(/^\d+ of 3$/)
    const start = Number((await position.textContent())!.split(" ")[0])
    const box = (await viewer.getByRole("img").first().boundingBox())!
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2)
    // A sideways wheel (tilt wheel, horizontal swipe) turns the page; a vertical one zooms.
    const step = start < 3 ? 1 : -1
    await page.mouse.wheel(120 * step, 0)
    await expect(position).toHaveText(`${start + step} of 3`)
    // A gamepad's left stick, the other way, for one poll.
    await page.evaluate((x) => Object.assign((window as unknown as { __pad: object }).__pad, { pad: { axes: [x, 0], buttons: [] }, polls: 1 }), -step)
    await expect(position).toHaveText(`${start} of 3`)
    await page.keyboard.press("Escape")
    await expect(viewer).toBeHidden()
  })

  test("asks before leaving with unsaved settings, and always with “confirm_leave” @fixture", async ({ page, context }) => {
    await page.goto("/#/settings")
    const main = page.getByRole("main")
    await main.getByRole("switch").first().click({ timeout: SLOW })
    await expect(page.getByRole("region", { name: "Unsaved changes" })).toContainText("Unsaved changes: 1")
    const asked = page.waitForEvent("dialog")
    void page.close({ runBeforeUnload: true })
    const dialog = await asked
    expect(dialog.type()).toBe("beforeunload")
    await dialog.accept() // leave: nothing was applied, the server keeps its settings
    await expect.poll(() => page.isClosed()).toBe(true)

    // Without unsaved edits, the setting alone makes the browser ask.
    const second = await context.newPage()
    await overrideBootstrap(second, { options: { confirm_leave: true } })
    await second.goto("/#/txt2img")
    await expect(second.getByRole("textbox", { name: "Prompt", exact: true })).toBeVisible({ timeout: SLOW })
    await second.getByRole("main").click({ position: { x: 5, y: 5 } }) // a user gesture: browsers only ask after one
    const again = second.waitForEvent("dialog")
    void second.close({ runBeforeUnload: true })
    const dialog2 = await again
    expect(dialog2.type()).toBe("beforeunload")
    await dialog2.accept()
  })
})
