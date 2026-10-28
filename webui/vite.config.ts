/// <reference types="vitest/config" />
import { readFile, writeFile } from "node:fs/promises"
import path from "node:path"
import { promisify } from "node:util"
import { brotliCompress, constants as zlib, gzip } from "node:zlib"
import babel from "@rolldown/plugin-babel"
import tailwindcss from "@tailwindcss/vite"
import react, { reactCompilerPreset } from "@vitejs/plugin-react"
import { defineConfig, loadEnv, type Plugin } from "vite"

// The WebUI server (python launch.py) serves the built app from webui/dist;
// `pnpm dev` proxies its API and WebSocket to that server.
const backend = process.env.REMI_BACKEND ?? "http://127.0.0.1:7860"

/**
 * Mock mode (VITE_MOCK=1): the API is answered in the browser (src/api/mock);
 * this plugin only serves what the page requests directly: placeholder
 * artwork for /files and /thumbs ids (<img> tags) and an empty user.css
 * (the bootstrap fixture lists one, so the app links it).
 */
function mockFiles(): Plugin {
  return {
    name: "remi-mock-files",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (/^\/api\/v2\/user-assets\/user\.css(?:\?|$)/.test(req.url ?? "")) {
          res.setHeader("Content-Type", "text/css")
          res.setHeader("Cache-Control", "no-store")
          res.end("/* the mock backend's user.css: empty */\n")
          return
        }
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

/**
 * Writes `.br` and `.gz` copies of the built scripts, styles and SVGs next to them; the server
 * sends them to clients that accept the encoding (remi/server/spa.py), so nothing is compressed
 * per request.
 */
function precompress(): Plugin {
  const brotli = promisify(brotliCompress)
  const deflate = promisify(gzip)
  return {
    name: "remi-precompress",
    apply: "build",
    async writeBundle(options, bundle) {
      const dir = options.dir ?? "dist"
      const files = Object.keys(bundle).filter((name) => /\.(?:js|css|svg)$/.test(name))
      await Promise.all(
        files.map(async (name) => {
          const file = path.join(dir, name)
          const raw = await readFile(file)
          if (raw.length < 1024) return
          const [br, gz] = await Promise.all([
            brotli(raw, { params: { [zlib.BROTLI_PARAM_QUALITY]: 11, [zlib.BROTLI_PARAM_SIZE_HINT]: raw.length } }),
            deflate(raw, { level: 9 }),
          ])
          await Promise.all([writeFile(`${file}.br`, br), writeFile(`${file}.gz`, gz)])
        }),
      )
    },
  }
}

/**
 * The API documentation viewers the server shows at /docs (Swagger UI) and /redoc (ReDoc),
 * vendored into the build instead of loaded from a CDN: pinned versions (package.json), served by
 * the server itself under the page CSP, no third-party code on the app origin. Each package goes
 * to `assets/<package>-<version>/` (a new version is a new path, so the immutable caching of
 * /assets stays correct) with its license files; `api-docs.json` tells the server
 * (remi/server/api_docs.py) where they are.
 */
function apiDocs(): Plugin {
  const modules = path.resolve(import.meta.dirname, "node_modules")
  const packages = {
    swagger: { name: "swagger-ui-dist", files: ["swagger-ui-bundle.js", "swagger-ui.css", "swagger-ui-bundle.js.LICENSE.txt", "LICENSE", "NOTICE"] },
    redoc: { name: "redoc", files: ["bundles/redoc.standalone.js", "bundles/redoc.standalone.js.LICENSE.txt", "LICENSE"] },
  }
  // Resources the viewers load from outside hosts on their own, replaced by local ones: ReDoc's
  // "API docs by Redocly" attribution shows its logo from cdn.redoc.ly (the text and link stay).
  // A build whose viewer still names one of these hosts fails, so an upgrade cannot bring it back.
  const outside: [RegExp, string][] = [[/https:\/\/cdn\.redoc\.ly\/redoc\/logo-mini\.svg/g, "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='0' height='0'/%3E"]]
  const forbidden = /https:\/\/(?:cdn\.redoc\.ly|cdn\.jsdelivr\.net|unpkg\.com|fonts\.(?:googleapis|gstatic)\.com)\//
  return {
    name: "remi-api-docs",
    apply: "build",
    async generateBundle() {
      const urls: Record<string, string> = {}
      for (const [key, { name, files }] of Object.entries(packages)) {
        const root = path.join(modules, name)
        const { version } = JSON.parse(await readFile(path.join(root, "package.json"), "utf8")) as { version: string }
        for (const file of files) {
          const fileName = `assets/${name}-${version}/${path.basename(file)}`
          let source: string | Buffer = await readFile(path.join(root, file))
          if (/\.(?:js|css)$/.test(file)) {
            // the source maps are not shipped: no reference to a missing file
            source = source.toString("utf8").replace(/\n\/[/*]# sourceMappingURL=\S+?(?: ?\*\/)?\s*$/, "\n")
            for (const [pattern, local] of outside) source = source.replace(pattern, local)
            const remote = forbidden.exec(source)
            if (remote) this.error(`${name}/${file} loads ${remote[0]}…: add a local replacement in apiDocs()`)
            urls[`${key}${file.endsWith(".css") ? "Css" : "Js"}`] = fileName
          }
          this.emitFile({ type: "asset", fileName, source })
        }
      }
      this.emitFile({ type: "asset", fileName: "api-docs.json", source: `${JSON.stringify(urls, null, 2)}\n` })
    },
  }
}

/**
 * Phosphor ships every icon in six weights; the app draws three (`bold`, `regular`, `fill`; see
 * src/components/icons.tsx). Drop the others from each icon's definitions: about half of every
 * icon's bytes, in the first load as much as anywhere.
 */
const PHOSPHOR_WEIGHTS = new Set(["bold", "regular", "fill"])
function phosphorWeights(): Plugin {
  const entry = /\n {2}\[\n {4}"(\w+)",/g
  return {
    name: "remi-phosphor-weights",
    enforce: "pre",
    transform(code, id) {
      if (!/@phosphor-icons[\\/]react[\\/]dist[\\/]defs[\\/]/.test(id)) return null
      const starts = [...code.matchAll(entry)]
      const end = code.lastIndexOf("\n]);")
      if (starts.length === 0 || end < 0) return null
      const head = code.slice(0, starts[0]!.index)
      const kept = starts
        .map((match, i) => ({ weight: match[1]!, body: code.slice(match.index, i + 1 < starts.length ? starts[i + 1]!.index : end).replace(/,\s*$/, "") }))
        .filter((part) => PHOSPHOR_WEIGHTS.has(part.weight))
        .map((part) => part.body)
      return { code: `${head}${kept.join(",")}${code.slice(end)}`, map: null }
    },
  }
}

export default defineConfig(({ mode }) => {
  const mock = loadEnv(mode, process.cwd(), "VITE_").VITE_MOCK === "1"
  return {
    plugins: [phosphorWeights(), react(), babel({ presets: [reactCompilerPreset()] }), tailwindcss(), mock && mockFiles(), apiDocs(), precompress()],
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
    experimental: {
      // Chunks and preloads resolve relative to the importing module, so the app also
      // works under `--subpath`; index.html keeps absolute URLs (the server prefixes them).
      renderBuiltUrl: (_filename: string, { hostType }: { hostType: "js" | "css" | "html" }) =>
        hostType === "html" ? undefined : { relative: true },
    },
    build: {
      outDir: "dist",
      sourcemap: false,
      target: "es2023",
      chunkSizeWarningLimit: 600,
      rolldownOptions: {
        output: {
          codeSplitting: {
            // Everything the shell needs before first paint in three files (React, other libraries,
            // the app shell) instead of dozens of tiny shared chunks, none over the 150 KB gzip
            // chunk budget; lazily loaded code keeps rolldown's automatic splitting.
            groups: [
              { name: "react", tags: ["$initial"], test: /node_modules[\\/](?:\.pnpm[\\/])?(?:react|react-dom|scheduler)[@\\/]/, priority: 3 },
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
      // the measurement tools' own tests (tools/perf/*.test.mjs) run with the app's
      include: ["src/**/*.test.{ts,tsx}", "tools/**/*.test.mjs"],
      css: false,
      restoreMocks: true,
      // jsdom suites are CPU heavy: half the cores and generous timeouts keep runs stable under load
      // (the e2e server, pytest) instead of timing out; override with `--maxWorkers`.
      maxWorkers: "50%",
      testTimeout: 20_000,
      hookTimeout: 30_000,
      // `vitest run --coverage` (tools/check.sh): the floors sit a little under the measured
      // values (2026-10-20: lines 86.1%, statements 80.6%, functions 76.2%, branches 70.6%), so a
      // change that drops coverage noticeably fails the check. Raise them as coverage grows.
      coverage: {
        provider: "v8",
        include: ["src/**/*.{ts,tsx}"],
        exclude: ["src/**/*.test.{ts,tsx}", "src/test/**", "src/api/gen/**", "src/**/*.d.ts"],
        reporter: ["text-summary", ["text", { skipFull: true }]],
        reportsDirectory: "node_modules/.vitest-coverage",
        thresholds: { lines: 85, statements: 79.5, functions: 75, branches: 69.5 },
      },
    },
  }
})
