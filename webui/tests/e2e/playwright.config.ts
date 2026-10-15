/**
 * End-to-end tests against the mock backend (`pnpm dev:mock`): no Python
 * server, jobs and previews are simulated in the browser. Run with
 * `pnpm test:e2e` (starts the dev server on port 5199 unless one is running).
 */
import { defineConfig, devices } from "@playwright/test"

const PORT = 5199

// The dev server is on loopback: never reach it through an HTTP(S) proxy from the environment.
for (const key of ["NO_PROXY", "no_proxy"]) {
  process.env[key] = [process.env[key], "127.0.0.1", "localhost"].filter(Boolean).join(",")
}

export default defineConfig({
  testDir: ".",
  outputDir: "../../node_modules/.e2e-results",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "line" : "list",
  timeout: 30_000,
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    locale: "en-US",
  },
  webServer: {
    command: `pnpm exec vite --port ${PORT} --strictPort --host 127.0.0.1`,
    cwd: "../..",
    env: { VITE_MOCK: "1" },
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } }, grepInvert: /@phone/ },
    { name: "phone", use: { ...devices["Pixel 7"] }, grep: /@phone/ },
  ],
})
