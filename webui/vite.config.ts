/// <reference types="vitest/config" />
import path from "node:path"
import babel from "@rolldown/plugin-babel"
import tailwindcss from "@tailwindcss/vite"
import react, { reactCompilerPreset } from "@vitejs/plugin-react"
import { defineConfig, loadEnv, type Plugin } from "vite"

// The WebUI server (python launch.py) serves the built app from webui/dist;
// `pnpm dev` proxies its API and WebSocket to that server.
const backend = process.env.REMI_BACKEND ?? "http://127.0.0.1:7860"

/**
 * Mock mode (VITE_MOCK=1): the API is answered in the browser (src/api/mock);
 * this plugin only serves placeholder artwork for /files and /thumbs ids,
 * which <img> tags request directly.
 */
function mockFiles(): Plugin {
  return {
    name: "remi-mock-files",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const match = /^\/api\/v2\/(?:files|thumbs)\/([^?]+)/.exec(req.url ?? "")
        if (!match) return next()
        const id = decodeURIComponent(match[1])
        const hue = Number(id.split("-").at(-1)) || [...id].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 360, 7)
        const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="1024" viewBox="0 0 100 100">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="hsl(${hue} 70% 62%)"/><stop offset="1" stop-color="hsl(${(hue + 80) % 360} 60% 28%)"/>
</linearGradient></defs><rect width="100" height="100" fill="url(#g)"/>
<circle cx="64" cy="38" r="16" fill="hsl(${(hue + 40) % 360} 90% 85% / .55)"/>
<text x="6" y="94" font-family="system-ui" font-size="5" fill="white" opacity=".8">${id.replace(/[<&>]/g, "")}</text></svg>`
        res.setHeader("Content-Type", "image/svg+xml")
        res.setHeader("Cache-Control", "no-store")
        res.end(svg)
      })
    },
  }
}

export default defineConfig(({ mode }) => {
  const mock = loadEnv(mode, process.cwd(), "VITE_").VITE_MOCK === "1"
  return {
    plugins: [react(), babel({ presets: [reactCompilerPreset()] }), tailwindcss(), mock && mockFiles()],
    resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
    server: {
      proxy: mock
        ? undefined
        : {
            "/api": { target: backend, ws: true },
            "/sdapi": backend,
            "/files": backend,
          },
    },
    build: {
      outDir: "dist",
      sourcemap: false,
      target: "es2023",
      chunkSizeWarningLimit: 600,
      rolldownOptions: {
        output: {
          codeSplitting: {
            // Everything the shell needs before first paint in two files instead of dozens of
            // tiny shared chunks; lazily loaded code keeps rolldown's automatic splitting.
            groups: [
              { name: "vendor", tags: ["$initial"], test: /node_modules/, priority: 2 },
              { name: "shell", tags: ["$initial"], priority: 1 },
            ],
          },
        },
      },
    },
    test: {
      environment: "jsdom",
      setupFiles: ["./src/test/setup.ts"],
      include: ["src/**/*.test.{ts,tsx}"],
      css: false,
      restoreMocks: true,
    },
  }
})
