/**
 * Features that go through browser APIs a test cannot see on a real screen:
 * the screen wake lock while generating, the notification permission prompt
 * and the notification and sound when images are ready in the background,
 * gamepad navigation in the viewer, the live preview in the viewer, and the
 * feedback of the extensions' "Check for updates". The APIs are replaced by
 * probes (`navigator.wakeLock`, `Notification`, `Audio`,
 * `navigator.getGamepads`) that record what the app does with them.
 *
 * Tagged `@real`: they run against the mock backend with the other tests and,
 * with `REMI_E2E_BASE_URL`, against a real server that generates
 * (`tools/smoke_real.py --browser`, docs/remi-webui/testing.md).
 */
import { expect, type Page, test } from "@playwright/test"
import { mod, openGenerate, overrideBootstrap, REAL_SERVER } from "./helpers"

type Probe = {
  wake: { requests: number; held: number }
  permissionAsked: number
  notes: { title: string; body: string | undefined }[]
  sounds: { src: string; volume: number }[]
  /** The pad `getGamepads()` reports for the next `padPolls` polls. */
  pad: { axes: number[]; buttons: { pressed: boolean }[] } | null
  padPolls: number
  /** The page is in the background (hidden, not focused). */
  hidden: boolean
}

type ProbeWindow = Window & { __probe: Probe }

/** Replaces the browser APIs with recording probes before the app loads. */
async function installProbes(page: Page) {
  await page.addInitScript(() => {
    const probe: Probe = { wake: { requests: 0, held: 0 }, permissionAsked: 0, notes: [], sounds: [], pad: null, padPolls: 0, hidden: false }
    ;(window as unknown as ProbeWindow).__probe = probe

    Object.defineProperty(navigator, "wakeLock", {
      configurable: true,
      value: {
        request: async () => {
          probe.wake.requests++
          probe.wake.held++
          const sentinel = {
            released: false,
            release: async () => {
              if (sentinel.released) return
              sentinel.released = true
              probe.wake.held--
            },
          }
          return sentinel
        },
      },
    })

    class ProbeNotification {
      static permission: NotificationPermission = "default"
      static async requestPermission(): Promise<NotificationPermission> {
        probe.permissionAsked++
        ProbeNotification.permission = "granted"
        return "granted"
      }
      onclick: (() => void) | null = null
      constructor(title: string, options?: NotificationOptions) {
        probe.notes.push({ title, body: options?.body })
      }
      close() {}
    }
    Object.defineProperty(window, "Notification", { configurable: true, writable: true, value: ProbeNotification })

    class ProbeAudio {
      volume = 1
      currentTime = 0
      readonly src: string
      constructor(src: string) {
        this.src = src
      }
      play() {
        probe.sounds.push({ src: this.src, volume: this.volume })
        return Promise.resolve()
      }
    }
    Object.defineProperty(window, "Audio", { configurable: true, writable: true, value: ProbeAudio })

    Object.defineProperty(navigator, "getGamepads", {
      configurable: true,
      value: () => {
        if (probe.pad && probe.padPolls > 0) {
          probe.padPolls--
          return [probe.pad]
        }
        return [null]
      },
    })

    Object.defineProperty(document, "visibilityState", { configurable: true, get: () => (probe.hidden ? "hidden" : "visible") })
    const hasFocus = document.hasFocus.bind(document)
    document.hasFocus = () => !probe.hidden && hasFocus()
  })
}

const probe = (page: Page) => page.evaluate(() => (window as unknown as ProbeWindow).__probe)

/** Types a value into a numeric field of the form and commits it. */
async function setNumber(page: Page, label: string, value: number) {
  const box = page.getByRole("textbox", { name: label, exact: true }).first()
  await box.fill(String(value))
  await box.press("Tab")
  await expect(box).toHaveValue(String(value))
}

/** Small, short runs (a real model generates in seconds). */
async function quickRuns(page: Page, steps = REAL_SERVER ? 6 : 3) {
  await setNumber(page, "Sampling steps", steps)
  if (REAL_SERVER) {
    await setNumber(page, "Width", 512)
    await setNumber(page, "Height", 512)
  }
}

/** A run takes this long at most (a real server may load the model first). */
const RUN_TIMEOUT = REAL_SERVER ? 240_000 : 15_000

const history = (page: Page) => page.getByRole("listbox", { name: "History" }).getByRole("option")
const generateButton = (page: Page) => page.getByRole("button", { name: /^Generate\b(?! several)/ }).first()

async function generate(page: Page, results: number) {
  await generateButton(page).click()
  await expect(history(page)).toHaveCount(results, { timeout: RUN_TIMEOUT })
  await expect(generateButton(page)).toBeVisible({ timeout: RUN_TIMEOUT })
}

test.describe("browser APIs", () => {
  test("keeps the screen awake while generating, then lets it sleep @real", async ({ page }) => {
    await installProbes(page)
    await overrideBootstrap(page, { options: { prevent_screen_sleep_during_generation: true } })
    await openGenerate(page)
    await quickRuns(page)
    await page.keyboard.press(`${mod}+Enter`)
    await expect.poll(async () => (await probe(page)).wake.requests, { timeout: RUN_TIMEOUT }).toBeGreaterThan(0)
    await expect(history(page)).toHaveCount(1, { timeout: RUN_TIMEOUT })
    await expect.poll(async () => (await probe(page)).wake.held).toBe(0)
  })

  test("asks for the notification permission, then notifies with a sound when images are ready in the background @real", async ({ page }) => {
    await installProbes(page)
    // A notification.mp3 in the WebUI folder (the setting's requirement); this page only plays it through the probe.
    await overrideBootstrap(page, { options: { notification_audio: true, notification_volume: 40 }, add_user_assets: ["notification.mp3"] })
    await openGenerate(page)
    await quickRuns(page)
    await page.getByRole("button", { name: "More generate options" }).click()
    await page.getByRole("menuitem", { name: "Notify me when images are ready" }).click()
    await expect(page.getByText("Notifications are on")).toBeVisible()
    expect((await probe(page)).permissionAsked).toBe(1)

    // In the foreground: neither a notification nor a sound.
    await generate(page, 1)
    expect((await probe(page)).notes).toEqual([])
    expect((await probe(page)).sounds).toEqual([])

    // In the background: both, once per run.
    await page.evaluate(() => {
      ;(window as unknown as ProbeWindow).__probe.hidden = true
      document.dispatchEvent(new Event("visibilitychange"))
    })
    await generate(page, 2)
    await expect.poll(async () => (await probe(page)).notes, { timeout: RUN_TIMEOUT }).toEqual([{ title: "Remi", body: "Images generated: 1" }])
    const { sounds } = await probe(page)
    expect(sounds).toHaveLength(1)
    expect(sounds[0]?.src).toMatch(/\/user-assets\/notification\.mp3$/)
    expect(sounds[0]?.volume).toBeCloseTo(0.4)
  })

  test("steps through the results in the viewer with a gamepad @real", async ({ page }) => {
    await installProbes(page)
    await overrideBootstrap(page, { options: { js_modal_lightbox: true, js_modal_lightbox_gamepad: true, js_modal_lightbox_gamepad_repeat: 250 } })
    await openGenerate(page)
    await quickRuns(page)
    await generate(page, 1)
    await generate(page, 2)
    await page.getByRole("button", { name: "Open image viewer" }).first().click()
    const viewer = page.getByRole("dialog", { name: "Image viewer" })
    const position = viewer.getByText(/^\d+ of \d+$/)
    await expect(position).toHaveText("1 of 2")

    // Left stick to the right for one poll: the next image.
    await page.evaluate(() => Object.assign((window as unknown as ProbeWindow).__probe, { pad: { axes: [1, 0], buttons: [] }, padPolls: 1 }))
    await expect(position).toHaveText("2 of 2")
    // D-pad left (standard mapping, button 14): back.
    await page.evaluate(() => {
      const buttons = Array.from({ length: 16 }, (_, i) => ({ pressed: i === 14 }))
      Object.assign((window as unknown as ProbeWindow).__probe, { pad: { axes: [0, 0], buttons }, padPolls: 1 })
    })
    await expect(position).toHaveText("1 of 2")
    await page.keyboard.press("Escape")
    await expect(viewer).toBeHidden()
  })

  test("follows the live preview in the viewer while generating @real", async ({ page }) => {
    await overrideBootstrap(page, { options: { js_modal_lightbox: true, js_live_preview_in_modal_lightbox: true } })
    await openGenerate(page)
    // Long enough a run to open the viewer during it.
    await quickRuns(page, REAL_SERVER ? 12 : 30)
    await generate(page, 1)
    await generateButton(page).click()
    await expect(page.getByRole("button", { name: "Interrupt" }).first()).toBeVisible({ timeout: RUN_TIMEOUT })
    await page.getByRole("button", { name: "Open image viewer" }).first().click()
    const viewer = page.getByRole("dialog", { name: "Image viewer" })
    await expect(viewer.getByRole("img", { name: "Live preview" })).toBeVisible({ timeout: RUN_TIMEOUT })
    await expect(viewer.getByText("Live preview", { exact: true })).toBeVisible()
    // The run ends: the viewer shows its result.
    await expect(viewer.getByRole("img", { name: "Generated image" })).toBeVisible({ timeout: RUN_TIMEOUT })
    await expect(viewer.getByText(/^\d+ of 2$/)).toBeVisible()
  })

  test("Check for updates always answers @real", async ({ page }) => {
    await page.goto("/#/extensions")
    const check = page.getByRole("button", { name: "Check for updates" })
    await expect(check.or(page.getByText(/Extension management is disabled/))).toBeVisible({ timeout: 30_000 })
    if (!(await check.isVisible()) || (await check.isDisabled())) {
      await expect(page.getByText(/Extension management is disabled/)).toBeVisible()
      return
    }
    await check.click()
    // Nothing to check (no enabled user extension from a repository): a toast says so; otherwise the progress shows.
    await expect(page.getByText("No user extensions to check").or(page.getByRole("progressbar", { name: "Checking for updates" }))).toBeVisible()
  })
})
