# Remi WebUI frontend

React 19 + Vite 8 + Tailwind 4 + shadcn/ui (Base UI) single-page app for the
Remi WebUI. The Python server serves the build output from `webui/dist`.

```sh
pnpm install
pnpm dev          # proxies /api to REMI_BACKEND (default http://127.0.0.1:7860)
pnpm dev:mock     # no backend: fixtures + simulated jobs (VITE_MOCK=1)
pnpm build        # type-check + production build into dist/
pnpm test         # vitest (jsdom)
pnpm lint         # oxlint
```

Structure, conventions and how to add pages, widgets and translations:
[docs/remi-webui/frontend.md](../docs/remi-webui/frontend.md).
