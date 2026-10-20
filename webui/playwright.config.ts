/**
 * End-to-end tests against the mock backend (`VITE_MOCK=1`): no Python
 * server, jobs and previews are simulated in the browser. Run with
 * `pnpm test:e2e` (starts Vite on port 5199 unless one is running there).
 *
 * With `REMI_E2E_BASE_URL` (a running WebUI with a model, e.g. the one
 * `tools/smoke_real.py --browser` starts) only the tests tagged `@real` run,
 * against that server and its built SPA: the real-server browser smoke
 * (docs/remi-webui/testing.md).
 */
import { defineConfig, devices } from "@playwright/test"

const PORT = Number(process.env.REMI_E2E_PORT ?? 5199)
const CI = Boolean(process.env.CI)
const REAL = process.env.REMI_E2E_BASE_URL?.replace(/\/+$/, "") || null

// The dev server is on loopback: never reach it through an HTTP(S) proxy from the environment.
for (const key of ["NO_PROXY", "no_proxy"]) {
  process.env[key] = [process.env[key], "127.0.0.1", "localhost"].filter(Boolean).join(",")
}

export default defineConfig({
  testDir: "tests/e2e",
  // Warms Vite's transforms; a real server serves the built SPA.
  globalSetup: REAL ? undefined : "./tests/e2e/global-setup.ts",
  outputDir: "node_modules/.e2e-results",
  fullyParallel: false,
  // The mock worker and Vite's on-demand transforms are CPU bound: a few workers keep timings stable.
  // A real server runs one job at a time: its tests run one after another.
  workers: REAL ? 1 : process.env.REMI_E2E_WORKERS ? Number(process.env.REMI_E2E_WORKERS) : CI ? 2 : 3,
  forbidOnly: CI,
  retries: CI ? 1 : 0,
  reporter: CI ? [["line"], ["html", { open: "never", outputFolder: "node_modules/.e2e-report" }]] : "list",
  // Real generation (and a first model load) takes longer than the mock's.
  timeout: REAL ? 300_000 : 45_000,
  globalTimeout: 30 * 60_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: REAL ?? `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    locale: "en-US",
    timezoneId: "UTC",
  },
  webServer: REAL
    ? undefined
    : {
        // Vite itself, not `pnpm exec vite`: pnpm does not pass the stop signal on, the orphaned
        // server keeps the pipe open and Playwright never exits.
        command: `node node_modules/vite/bin/vite.js --port ${PORT} --strictPort --host 127.0.0.1`,
        env: { VITE_MOCK: "1" },
        url: `http://127.0.0.1:${PORT}`,
        reuseExistingServer: !CI,
        timeout: 90_000,
      },
  projects: REAL
    ? [{ name: "real", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } }, grep: /@real/ }]
    : [
        { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } }, grepInvert: /@phone/ },
        { name: "phone", use: { ...devices["Pixel 7"] }, grep: /@phone/ },
      ],
})
