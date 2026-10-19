# Remi WebUI frontend (`webui/`)

Status: wave 1 (foundation), 2026-10-15. Binding contracts:
[architecture.md](architecture.md) (layers, HTTP/WS API) and
[schema-format.md](schema-format.md) (the `remi.ui` JSON the renderer draws).

Stack: React 19 (React Compiler), Vite 8 (rolldown), TypeScript 6 strict,
Tailwind 4, shadcn/ui `base-nova` on Base UI, Zustand 5, TanStack Query 5,
i18next, CodeMirror 6 (prompt editor), Vitest + Testing Library (jsdom).

## Scripts

Run in `webui/`:

| command | what it does |
|---|---|
| `pnpm dev` | Vite dev server; proxies `/api` (incl. WS), `/sdapi`, `/files` to `REMI_BACKEND` (default `http://127.0.0.1:7860`) |
| `pnpm dev:mock` | No backend: `VITE_MOCK=1`, API answered in the browser from fixtures, jobs simulated with live previews |
| `pnpm build` | `tsc -b` then the production build into `webui/dist` (served by the Python server) |
| `pnpm typecheck` | type-check only |
| `pnpm test` / `pnpm test:watch` | Vitest |
| `pnpm test:e2e` | Playwright against mock mode (`tests/e2e`; starts `vite` on port 5199, bypassing proxies for loopback) |
| `pnpm lint` | oxlint (must be silent) |
| `python3 tools/shadcn_fetch.py <names>` | add shadcn registry components with curl (Node cannot use the proxy) |
| `$PYTHON tools/dump_schemas.py` (requirements.txt, no Gradio) | re-dump the real tab schemas into `src/api/mock/contract/*.json` (contract tests) after a `remi/ui` / `remi/core` change |

## Layout of `src/`

```
main.tsx            boot: load locale, mount providers, <App/>
index.css           design tokens (Tailwind @theme), base styles, renderer row classes
styles/shadcn.css   vendored keyframes and variants from shadcn/tailwind.css
app/                shell: App, router, routes (nav + lazy pages), NavRail/BottomBar/MoreSheet,
                    Header (+ header slot, ModelQuickBar, AppearanceMenu), command palette,
                    ws-bridge (WS events → query cache / progress / toasts)
api/                http transport, typed endpoints, models, query keys, WS store + frames,
                    progress store, mock/ (fixtures, in-browser server, mock socket)
ui/                 the schema renderer (see below)
widgets/<name>/     custom widgets, one directory each (auto-registered)
features/           generate-view features: prompt-editor, gallery (session results, history grid),
                    lightbox, progress (run machine + controller, restore, wake lock / title /
                    notifications), model-bar, send-to, infotext
pages/<page>/       one lazily loaded chunk per page; pages/ext/ = extension pages (#/ext/<id>),
                    pages/custom/<name>/ = dedicated views for extension pages with `widget=`
components/         app-level components (SearchSelect, Segmented, ErrorBoundary, …)
components/ui/      vendored shadcn components (edit sparingly; lint-exempt)
i18n/               i18next setup, locale store, catalog loading
locales/            chrome translations, one JSON per locale (zh-CN.json)
lib/                appearance store (theme.ts), toast, media query, cn()
test/               setup.ts (jsdom shims, jest-dom, i18n init), fake-socket.ts (WebSocket double)
```

`src/canvas/` is the image editor (see "Image editor"); it is the only module
that may import Konva, and only `canvas/engine/renderer.ts` does, inside the
editor's lazy chunk. `react-konva` stays unused (the editor drives Konva
imperatively).

## App shell

- **Routing**: `app/router.ts` is a typed hash router (`#/txt2img`, `#/img2img`,
  `#/extras`, `#/png-info`, `#/gallery`, `#/networks`, `#/queue`,
  `#/settings`, `#/extensions`, `#/ext/<id>` (extension pages; `#/page/<id>`
  still opens them), optional `?query`). Hash routing
  keeps the server contract to "serve `index.html` at `/`". Use
  `navigate(route)`, `routeHref(route)` for links, `useRoute()` to read.
- **Pages** are code-split: `app/routes.tsx` maps each route name to a
  `lazy(() => import("@/pages/…"))`; nav items preload their chunk on
  hover/focus. A page component receives `{ route }` (its own route variant).
- **Navigation**: `BUILTIN_NAV` plus extension pages from `GET /app`
  (`pages`), ordered/filtered by `Bootstrap.tabs`; Queue, Settings and
  Extensions are always present. Desktop: slim left rail with tooltips;
  below `md`: bottom bar (primary items) + "More" sheet.
- **Header**: page title, a page slot (`<HeaderSlot>` from
  `app/header-slot.tsx` portals page controls into it), the model quick-bar,
  command palette button, WS connection dot, appearance menu, and a progress
  line while a job runs.
- **Command palette** (Ctrl/Cmd+K, `cmdk`, its own chunk loaded on first
  open). Register commands with
  `useRegisterCommands(source, commands)`; labels and groups are English and
  translated at render.
- **Keyboard shortcuts** (`app/ShortcutsDialog.tsx`, lazy): every app-wide
  shortcut in one dialog, opened with `?` outside text fields and dialogs
  (`app/shortcuts-help.ts`; a handler that uses `?` itself — the image
  editor's list — prevents its default), the palette's "Keyboard shortcuts"
  and the appearance menu. Add a shortcut there when you add one.
- **Toasts**: `sonner`; use `showToast(level, message)` from `lib/toast.ts`
  for server-provided levels, or `toast` directly.
- **Appearance** (`lib/theme.ts`, persisted as `remi-appearance`): theme
  (light/dark/system), density (comfortable/compact), accent
  (`iris|blue|teal|green|amber|coral|rose`). `index.html` applies the stored
  values before first paint; `useApplyAppearance` keeps `<html>` in sync.
  The theme comes from, first to last: `?__theme=light|dark|system` in the
  URL (A1111's theme link: it picks that theme, as the menu would, and is
  remembered), the user's pick (`themeChosen`), the server's `--theme`
  (`window.__REMI_THEME__`, written into the shell by `remi/server/spa.py`
  before the pre-paint script), then dark. A browser that never picked a
  theme follows `--theme` when it changes.

## Design system

- Tokens live in `index.css`. Surfaces are neutral with a faint cool tint;
  one brand hue drives `--primary` and `--ring`: `--brand-h/-c/-l/-fg`, set per
  accent with `[data-accent=…]` (works on any element, e.g. swatches). shadcn's
  `--accent` stays a neutral hover surface. Extra semantic colours:
  `success`, `warning`, `canvas` (image backdrops), utility `bg-checker`.
- **Density** scales Tailwind's `--spacing` (`[data-density=compact]` →
  0.215rem), so every `p-*`, `gap-*`, `h-*`, `size-*` shrinks together. Use
  spacing utilities, not fixed pixel sizes, for anything that should follow.
- Typography: Inter Variable (UI) and JetBrains Mono Variable via
  `@fontsource-variable` (self-hosted, unicode-range split, no CDN); CJK falls
  back to system fonts. `.tabular` / numeric inputs use tabular figures.
- Motion: `tw-animate-css` + base-nova animations; a global
  `prefers-reduced-motion` rule neutralises animations and transitions.
  Nothing that stays on screen while a job runs animates endlessly (no
  `animate-pulse` / `animate-spin` on running indicators): an infinite
  animation makes the compositor redraw at display rate for the whole job,
  which measurably slows sampling where the browser shares the GPU
  (docs/research/ui-performance.md). Spinners are for short waits only.
  A generation step is one React commit and one drawn frame: while previews
  come every step, `api/progress-gate.ts` holds a `job.progress` until the
  next preview frame, and `api/paint-batch.ts` applies both in one animation
  frame once the image has decoded. No `backdrop-filter` on surfaces that
  change every step (the stage's progress overlay, the header with its
  progress line).
- Scrollbars: thin, theme-aware (`--scrollbar`). Focus: visible rings from
  `--ring` everywhere (`outline-ring/50`, `focus-visible:ring-3`).
- shadcn components import `cn` from the `cn` package (what the base-nova
  registry ships); app code imports it from `@/lib/utils`. Local fixes to
  vendored components: scalar sliders render one thumb and forward
  `getAriaLabel`; a select's value ellipsizes (line-clamp does nothing on
  the flex value); `number-field.tsx` is ours.

## API layer (`src/api`)

- `gen/schema.d.ts`: types generated from the server's OpenAPI schema
  (`openapi-typescript`, pinned): `$PYTHON
  tools/export_openapi.py` writes `src/api/openapi.json`, then `pnpm gen:api`
  regenerates the types; `gen/gen.test.ts` fails when they drift.
- `http.ts`: the path-typed client (`openapi-fetch`, pinned) on the mock
  switch's transport, `unwrap()` (data, or `ApiError` for any non-2xx: status,
  decoded body in `detail`, a readable message from Remi's
  `{"errors": [{owner, param, message}]}` or FastAPI's `detail`),
  `validationErrors(error)` for the per-field list, `ROOT` / `API_BASE` /
  `apiUrl()` from `window.__REMI_ROOT__` (`--subpath`), and a 401 outside
  `/auth` reloads the page (the server shows its login page).
- `endpoints.ts`: **the only place that builds URLs** (`api.app()`,
  `api.schema(tab)`, `api.action(owner, action, body)`, `api.jobs.*`,
  `api.upload(file)`, `api.fileUrl(id)`, `api.thumbUrl(id, w)`, …). Paths,
  params and bodies are checked against the generated `paths`; request
  fields the server defaults are filled in here.
- `types.ts`: aliases of the generated models (`Job`, `JobOutput`,
  `Bootstrap`, …); only free-form JSON the server types as `dict` (schemas,
  param values, action patches, `JobRequest.core/scripts`) is refined by hand.
- `queries.ts`: the `QueryClient` (no retries on 4xx), the `keys` factory and
  `queryOptions` per resource. Schemas and bootstrap are `staleTime: Infinity`
  and are invalidated by WS events.
- `ws.ts`: `useWs` (Zustand). One connection to `/api/v2/ws`; on open sends
  `{"type":"hello","features":[…]}` and stores the server's `hello.features`
  (`useServerFeature`). JSON events are dispatched to `onWsEvent(type, fn)`
  subscribers (`"*"` for all); both `{type, …fields}` and
  `{type, data: {…}}` envelopes are accepted, and `events.ts` validates the
  required fields per type (unknown or malformed events are dropped). Binary frames
  (`u8 type, u8 fmt, u32 job_seq, u32 step`, little-endian, then bytes;
  `frames.ts`) become `preview = {jobSeq, step, url}`; the previous object URL
  is revoked, late frames of the same job are dropped. Reconnect: exponential
  backoff 0.5 s → 15 s with ±25 % jitter, immediate retry when the tab becomes
  visible or the network returns. Every connect attempt carries a token, so
  overlapping attempts (StrictMode's connect → disconnect → connect, a
  wake-up during a pending open) never leave two live sockets.
- `progress.ts`: the running job (global: one worker), the jobs this client
  queued (`track(jobId, tab)` after `POST /jobs` or a retry) and the newest
  result **per form** (`lastResult(s, tab)`; other clients' jobs never show;
  a job that finishes before its POST answer is attached when tracked), and
  model loading. Fed by `app/ws-bridge.ts`, which also clears the live
  preview on `job.started` / completion (the panel only shows frames whose
  `job_seq` is the running job's), invalidates queries on `queue.changed`,
  job events, `options.changed` (incl. `["schema"]`, `["page"]`, models and
  styles: core forms and the model lists follow the options),
  `networks.changed` (networks, models) and everything after a reconnect or a
  `resync` (`pong` is accepted and ignored). Model loading
  ends on `model.loaded`, a `model.loading` stage `done|loaded|error|failed`,
  a reconnect, or after 180 s without news.

Build: chunks and their preloads resolve relative to the importing module
(`experimental.renderBuiltUrl`), so the app works under `--subpath`; the
server prefixes `index.html`'s absolute URLs. A tab opened before an update
that asks for a chunk the new build lacks reloads once (`vite:preloadError`,
at most once a minute). `index.html` carries `darkreader-lock` (the app has
its own dark theme).

## Schema renderer (`src/ui`)

`<TabForm schema={TabSchema}/>` renders a generation tab (core form +
scripts); `<PageForm schema={PageSchema}/>` renders one owner (extension
pages). `jobRequest(store, schema)` (`ui/tab.ts`) builds the `POST /jobs` body.

| file | role |
|---|---|
| `schema.ts` | TypeScript mirror of schema-format.md |
| `Renderer.tsx` | `RenderNode`: dispatch by `type`, `visible_if`/`enabled_if`, patch visibility, row cells |
| `inputs/*` | one component per input kind: Toggle, Slider (+ number box), Number, Text (`prompt` → lazy `PromptEditor`), Choice (select / segmented or radio list / searchable; `choice_hints` as option tooltips), MultiChoice (chips or checkboxes), Seed (dice / recycle), Color, Image, ImageList, File (`DropZone`: drop, paste, click → `POST /uploads`), Json, Custom |
| `paste-targets.ts` | a pasted image goes to the image field under the pointer or holding the focus (`DropZone`, the image editor); with none, to the first empty image field on screen, else the last one on screen (A1111's `dragdrop.js` rule). Image inputs and the editor register while mounted |
| `layout/*` | Section (enable toggle in header, open state, section reset), Group, Tabs (`bind`), ActionButton, Gallery/Output, Slot, row sizing |
| `Field.tsx` | label row, modified dot (click = reset to default, tooltip shows the default), info text, server validation message (`FieldError`); `HintedLabel`: the param's `hint` as the label's tooltip (dotted underline) and as the control's accessible description (`hintRef`) — the former `hints.js` map and `tooltip=` |
| `a11y.ts` | ids of label / info / error texts and `describedBy(id, info, error)` for controls |
| `conditions.ts` | condition evaluator (`param` / `eq` / `ne` / `in` / `gt…lte` / `all` / `any` / `not`, `"owner.param"`) |
| `form-store.ts` | per-form Zustand store (`getFormStore(formId)`); `setValues`/`applyPatches` skip params locked by a toggle's `locks` (`locks.ts`), `persist: false` inputs are never drafted |
| `messages.ts` | translation of server texts (toasts, text outputs, `{text, params}` messages) |
| `actions.ts` | `runAction` + patch application, `runChangeEffects` (on_change + conflict locks after an edit), `resetParams` |
| `uploads.ts` | upload + local previews (object URLs revoked when unreferenced, at most 24 kept) |
| `RichText.tsx`, `rich-text.tsx` | Markdown → React elements (no raw HTML); HTML sanitised via DOMParser → React elements (allow-list, no `innerHTML`) |
| `widgets.ts` | custom widget registry |

Rules the renderer implements:

- **Owners**: `core` for the generation form, the script id for scripts,
  `page:<id>` for pages. Hooks: `useParam(node)` binds an input
  (value, default, modified, override, enabled, `setValue`, `reset`);
  `useCondition`, `useFormState`, `useRunAction` (`ui/hooks.ts`).
- **Form store** (one per form id: `txt2img`, `img2img`, `extras`,
  `page:<id>`): values and defaults per owner, action overrides
  (`choices/visible/enabled/label/info/minimum/maximum/step`), outputs, open
  sections, selected tabs. Drafts persist in `localStorage`
  (`remi-form:<formId>`, writes coalesced to 300 ms and flushed on page hide).
  On load, a draft value is kept only if it still fits the node (type,
  choice membership) and is coerced like the server does (range clamp,
  integer rounding, `number.precision`); upload ids (`image`, `image_list`,
  `file`) and hidden `state` are never persisted. A schema reload keeps
  current values. The store also holds the schema's `conflict_params` and
  the last rejected submit's `errors` (not persisted).
- **Visibility**: `visible_if` hides a node and its subtree; a patch
  `visible: false` hides an input (AND). `enabled_if` disables the subtree;
  patch `enabled: false` disables an input. `hidden: true` and `state` inputs
  are not rendered but are submitted. A `tab` with `visible_if` false has
  neither trigger nor content (an unbound strip then shows its first visible
  tab); its `enabled_if` disables its content. Patch `label` / `info` replace
  the static text; `null` clears it.
- **Conditions** follow Python (`remi/ui/conditions.py`): `True == 1`,
  `in` uses that equality, `gt…lte` order numbers (booleans as 0/1) and
  strings (by code point); any other pair, `null` or a missing param is false.
- **Actions**: body `{tab, values, core?, scripts?, …extra}`: `values` is the
  calling owner's, `core` the generation form's (for script owners) and
  `scripts` every other non-core owner's values by id, so cross-owner actions
  see the live form. Patches apply to the calling owner or `"owner.param"`;
  `__toast__` → toast, `__outputs__` → the owner's outputs, `__settings__`
  → merged into the cached option values (the server already stored them;
  the client never writes them back). A newer call of the same action wins
  (stale responses are dropped).
- **Change effects** (`runChangeEffects`): after a user edit (or reset) of a
  param, its `on_change` action runs — immediately for discrete kinds,
  debounced 300 ms for slider / number / text / color / json — and, when
  `"owner.param"` is in the tab schema's `conflict_params`, so does
  `core.__conflicts__` with `selected` = that reference. This covers section
  enable toggles, choices, toggling off, and resets. A section reset or
  "Reset all parameters" (`resetOwners`, over every owner at once) runs each
  distinct `on_change` once and one conflicts call (`selected` only when a
  single conflict param changed). A param's `on_input` action runs only
  after the user edits that control (`useParam`): never after pastes,
  presets, bulk resets or action patches. "Reset all parameters" asks first when the
  prompts or a source image would be lost.
  Values set by a patch do not trigger effects (no loops). After a schema
  load, if a draft restored a non-default conflict param, the locks are
  re-derived with one `__conflicts__` call (patch overrides are not persisted).
- **Validation errors**: a rejected `POST /jobs` (`{"errors": [...]}`) sets
  `errors[owner][param]` in the form store; the field shows the message
  (`aria-invalid`, `aria-describedby`), the sections / titled groups /
  unbound tabs around it are opened, and a toast lists all messages. Editing
  or resetting a field clears its error; a successful submit clears all.
- **Slots**: scripts render in the core layout's `slot` whose `section`
  matches, sorted by `order`; unknown/absent sections go to `scripts`, which
  is appended after the core layout if the schema has no such slot. A script
  whose layout is not a section or titled group (e.g. an untitled group,
  the `sampler` row) gets a titled frame; an untitled group's own box is
  replaced by it. The extensions panel lists every untitled script as a
  closed section titled by the script.
- **Uploads**: images are previewed from object URLs; removing / replacing
  an image revokes its URL once no form store references the id, older
  unreferenced previews are swept before each upload, and at most 24 are
  kept (then `/files/{id}` is used).
- **Keyboard** (generate view): Ctrl/Cmd+Enter generates (ignores key repeat
  and IME composition); while this form's job runs it interrupts and, unless
  `ctrl_enter_interrupt` is on, generates again once the job has ended.
  Ctrl/Cmd+Shift+Enter adds the form to the queue ("Add to queue").
  Alt+Enter skips. Escape interrupts only when nothing else owns it: not in a
  text field or prompt editor, not inside a dialog, menu or popup, and no
  popup open. Generate keys are ignored while a modal dialog is open.
- **Rows**: `width` 1–12 = share of 12 columns, `"fill"` = the rest,
  absent/`"auto"` = equal share, buttons hug content. Rows are CSS containers;
  under 30 rem spans and fills go full width (phones stack).
- **Prompt editor**: `text` nodes with `prompt: true` get the full editor of
  `features/prompt-editor` (`ui/prompt/PromptEditor.tsx` re-exports it; the
  pure editing functions stay in `ui/prompt/`). See "Generate page".

### Adding a widget

Create `src/widgets/<name>/index.tsx` whose default export is a component
taking `WidgetProps<Value, Props>` from `@/ui/widgets`:

```tsx
import type { WidgetProps } from "@/ui/widgets"

export default function StageTimeline({ value, onChange, props, disabled, id, label }: WidgetProps<number[], { stages: string[] }>) {
  // Put `id` on the primary focusable element; the field label points at it.
}
```

That is all: `import.meta.glob` discovers it at build time, it is loaded
lazily the first time a `custom` node with `widget: "<name>"` renders, and
the field chrome (label, modified dot, reset, info) is provided around it.
An unknown widget name falls back to the JSON editor so values are never
lost. Example: `widgets/block-weights`. Shared widgets with a server-side
node: `widgets/range-slider` (`ui.RangeSlider`).

A widget may also edit an input of another kind through its `widget` hint
(today `image_list`: `ui.ImageList(name, widget="image-list")`). The value
keeps that kind's rules (an image list's upload ids are never restored from
drafts, and count as source images), and the widget receives the node as
`WidgetNode`. An async widget edit (an upload) must commit against the
store's current value (`store.getState().values[owner][node.name]`), not the
value of the render that started it (see `widgets/image-list`).

A widget may sum up its values in the header of the section holding it
(shown while the section is collapsed, like the old ControlNet "2x Unit"
badge): a `src/widgets/<name>/summary.ts` exporting
`sectionSummary(values, nodes)` (the owner's values, the section's inputs of
that widget) returning translated text or `null`. It is imported eagerly
(`ui/section-summary.ts`), so keep it free of heavy imports. Example:
`widgets/controlnet-unit/summary.ts`.

### Adding a page

- Built-in page: create `src/pages/<page>/index.tsx` (default export
  `({ route })`), add the route name to `router.ts`, its loader in
  `routes.tsx` and a `BUILTIN_NAV` entry.
- Extension page with a dedicated view (`Page(..., widget="x")`): create
  `src/pages/custom/x/index.tsx` (default export `({ page })`). Without
  `widget`, the page is schema-rendered from `GET /api/v2/pages/{id}` by
  `pages/ext` (`PageView`): its form `page:<id>` is loaded once per schema
  object and kept across visits (values, upload ids and outputs survive
  leaving the page and a schema reload), and its `on_load` action runs each
  time the page is shown. Other views send values to a page with
  `sendToPage(id, values)` (`pages/ext/deliver.ts`: applied at once to a
  loaded page, else right after its first load; each value's `on_change`
  runs). Pages declaring `send_to` image targets get "Send to <label>"
  buttons wherever `PageSendButtons` (`pages/ext/send-targets.tsx`) is
  placed: PNG Info's "Send image" group.
- Page widgets: `task-runner` (start a page task and follow it: progress,
  status text, Cancel; the poll outlives the view, so a result lands even
  after leaving the page), `send-button` (a param of the page to the
  txt2img / img2img prompt — append with the old ", " joining, or replace,
  with or without switching — to another page's image input, optionally
  after an action such as a download, or an http(s) link in a new tab) and
  `secret-text` (a masked, write-only text such as an API key; declare it
  `persist=False, submit=False`).
- Extension page icons are names from the curated map in `app/icons.ts`
  (unknown → puzzle piece).
- The **Checkpoint Merger** (A1111's `modelmerger` tab) is such a
  schema-rendered page, from core: `remi/core/merger.py`, registered by
  `scripts/checkpoint_merger.py` so it follows "Reload UI" like extension
  pages (and "UI Tab Order" / "Hide UI Tabs" by its title). Its merge and
  the two "Save Current Model" buttons are `task-runner` tasks behind
  generation (progress per stage and per tensor, Cancel between steps); the
  new file reaches the checkpoint lists through `networks.changed`.

## Generate page (`pages/generate`, `features/`)

txt2img and img2img share one view; the mode switch (two links) sits in the
header slot, and each mode has its own form store and schema (the other
mode's schema is prefetched).

- **Layout**: desktop ≥ 72 rem: three resizable, collapsible panes
  (parameters | stage | history; `react-resizable-panels`, sizes persisted);
  48–72 rem: parameters | stage with a history strip; phones: stage and
  history strip first, then the parameters with the run controls stuck to
  the bottom (and the stacked model bar). The run controls are sticky at the
  top of the parameters pane; with the pane collapsed they move to the stage
  header.
- **Form loading** (`use-form-loader.ts`): the page loads its schema itself
  (not `TabForm`), once per schema object. On a reload (options changed, UI
  preset) image ids survive and params still at their old default follow
  the new default (`carryOver`).
- **Curated form** (`curate.ts`): prompt, negative prompt, styles and the
  `script` choice are taken out of the core layout; the remaining top-level
  blocks are classified by content (source image, sampler slot, size,
  sections, guidance, seed, other) only to frame them (cards, size tools);
  they keep the schema's order, so the user's category order
  (`ui_reorder_list`) decides the layout, and render with the schema
  renderer, so server-side layout changes still show. Size gets aspect
  presets (keeping the pixel budget, 64-px grid) and "W x H" typing / paste
  (paste anywhere in the size card). Scripts without a slot of their own are
  a searchable, collapsible list with an "only modified" filter, per-script
  reset and "reset all"; the selectable-script choice and the chosen script
  follow. img2img's source blocks render in `SourceSlot` (`CoreForm.tsx`)
  as the canvas module's `Img2imgSourcePanel`; the image is edited on the
  stage (see "Image editor").
- **Prompt editor** (`features/prompt-editor`, CodeMirror 6, lazy chunk):
  colouring from `tokenize.ts` (attention weights, `[a:b:0.5]` / `[a|b]`
  schedules, `<lora:…>` with numeric arguments, `BREAK` / `AND`,
  `__wildcards__`, `{a|b}`, escapes, `#` comment lines, bracket depth),
  bracket-balance warnings (lint underline plus a summary in a polite live
  region, so screen readers announce it; replaces prompt-bracket-checker),
  Ctrl/Cmd+↑/↓ attention with the `keyedit_*` precisions, Alt+←/→ tag
  moves, undo/redo (600 ms grouping; always on, so the Gradio-era
  `undo_redo` setting has nothing left to switch, and `prompt_debounce` is
  moot: typing never makes a request — only the token count does, debounced),
  image drops (PNG `tEXt`/`iTXt`/`zTXt`, JPEG/WebP EXIF UserComment read in
  the browser, else upload + `/infotext/image`). `PromptField` adds the label,
  reset, token counter (`POST /prompt/tokens`, 700 ms after the text and the
  steps stop changing, superseded requests aborted, last count kept; counts
  taken with no model loaded are redone after `model.loaded` or a finished job) and page actions: paste
  parameters (from the prompt text, or the last generation when empty) and
  clear (confirmed). Styles: the `styles` multiselect, "apply to prompts"
  (`POST /styles/apply`, clears the selection) and a style editor dialog
  (`/styles` CRUD; refreshes the schemas).
  **Completion sources**: `registerCompletionSource(id, source, {priority})`
  from `@/features/prompt-editor` adds a CodeMirror `CompletionSource` to every
  open and future editor (no autocompletion extension while none is
  registered); sources read `promptKind` (`facets.ts`) to tell positive from
  negative prompts. Tab accepts like Enter; `completionAcceptKey()` tells
  them apart inside a completion's `apply`; options may carry `className`
  and `decorate()` (`DecoratedCompletion`: a class on the row, a node after
  the label). **Prompt decorators**: `registerPromptDecorator(id,
  Component)` wraps every prompt editor (`PromptField` and the renderer's
  `prompt: true` text fields, via `PromptDecorators.tsx`); the component
  gets the editor as `children` (keep it mounted), the value and setter,
  the kind, the label and the editor handle.
- **Run controller** (`features/progress`): `run-machine.ts` is a pure
  reducer (Generate / queue another / Ctrl+Enter restart / Interrupt (also
  cancels this form's queued jobs and a submit in flight) / Skip / generate
  forever / repeat N / restore); `run-controller.ts` performs its effects
  (`POST /jobs`, `/interrupt`, `/skip`, `DELETE`) and feeds it WS job events
  (`job.completed` with status `interrupted` / `cancelled` counts as
  interrupted). Forever and repeat submit the next run when the previous one
  has finished (a failure or Interrupt ends them). Submits in flight are a
  counter (an Interrupt cancels every one of them when it answers; a second
  Ctrl/Cmd+Enter while one is pending queues nothing). Jobs in flight are
  kept in `localStorage` (`remi-run:<tab>`, written from the run state) and
  re-attached after a reload (`restore.ts`): attached first, then each looked
  up with `GET /jobs/{id}`, so events arriving meanwhile apply; finished ones
  go to the history. Events lost while the socket was down are recovered the
  same way: every `hello` and every `queue.changed` whose `running` is not
  the job a form believes runs re-check that form's jobs (`wireRunSync`), and
  `queue.changed` drops a stale live progress. While a job of
  this client runs: screen wake lock (`prevent_screen_sleep_during_generation`),
  `[42%] ETA 12s · …` in the title (`show_progress_in_title`), and on
  completion with the page in the background the user's `notification.mp3`
  when the server lists it in `user_assets` (`notification_audio`,
  `notification_volume`; silent otherwise, as the setting says) and a browser
  notification (permission asked from the Generate menu; a batch's grid is
  not counted when `return_grid` is on). The status line shows the queue
  position (from `GET /queue/items`, i.e. run order with priorities and
  reordering), step, batch and ETA.
- **Results** (`features/gallery`): this session's results (sessionStorage,
  400 max) in a virtualized thumbnail grid / strip (one listbox tab stop:
  arrows select, Enter opens, Delete removes). The stage shows the selected
  result, or the live preview while a job runs unless an older result was
  picked ("Show live preview" returns). The lightbox (lazy chunk): wheel zoom
  at the cursor, two-finger pinch zoom around the fingers' midpoint, drag to
  pan, double-click fit ↔ actual size, ←/→ (or a sideways wheel — tilt
  wheel, horizontal two-finger swipe — at most once per
  `js_modal_lightbox_gamepad_repeat`; or a gamepad's left stick / d-pad
  with `js_modal_lightbox_gamepad`, repeating per
  `js_modal_lightbox_gamepad_repeat`), Home/End,
  S downloads, +/−/0/1, T tile preview, middle click opens in a new tab,
  Esc closes; initial zoom per `js_modal_lightbox_initially_zoomed`; follows
  the live preview with `js_live_preview_in_modal_lightbox`. A job's video
  (output `kind: "video"`, e.g. Wan) belongs to its first frame: the stage
  plays it (controls; `video_player_auto` starts it, `video_player_loop`
  loops it), the history and the lightbox keep the frames.
- **Result actions** (`features/send-to`): download (`/files/{id}?download=1`,
  the server names the file), save to the Save folder or the whole
  generation as a zip (`POST /outputs/save`), open in new tab, copy
  parameters, reuse seed (the param whose `infotext` is `Seed`), use these
  parameters (all of them), send to img2img / inpaint / extras / txt2img.
  On txt2img's stage and lightbox, "Upscale with Hires. fix" (✨, A1111's
  button under the gallery) queues the txt2img form's values with
  `upscale_from` = the result (api.md): not for grids, extra images (output
  `role`) or videos. The upscaled result replaces its source in the history,
  or follows it with `hires_button_gallery_insert` (then
  `hires_insert_index` selects it, otherwise the source stays selected).
  "Send to" parses the infotext for the target form, drops seed / size per
  `send_seed` / `send_size`, puts the output file id into `init_img` (`mode`
  img2img / Inpaint) or Extras `image`, and switches page; values reach a
  form that is not loaded yet after its next schema load (`deliver`). A new
  source image clears the layers drawn over the old one (image params whose
  `source` names it: `sketch`, `mask`).
  Pasted values run each `on_change` once and one `__conflicts__` pass.
- **Model bar** (`features/model-bar`, the header's `ModelQuickBar`): UI
  preset (`__forge_preset__` on the form on screen, then all schemas
  refresh), checkpoint (searchable, architecture and size), VAE / text
  encoders (applied when the picker closes, so one change loads the model
  once), "Diffusion in Low Bits"; option writes are optimistic `PATCH
  /options`; `model.loading` stages show next to it. Narrow headers fold
  preset, modules and dtype into a popover.

## Image editor (`src/canvas`)

Replaces ForgeCanvas (core-frontend-features.md §3.33, §10.G). Public
surface (`@/canvas`): `<ImageEditor>` (lazy; Konva and the editor UI are one
chunk, `preloadImageEditor()` fetches it early), the img2img integration
(`Img2imgSourcePanel`, `Img2imgStage`) and the schema binding helpers.

| directory | role |
|---|---|
| `core/` | pure logic, unit-tested: geometry and view math, undo `History` (≥ 32 states, byte budget), pixel patches (empty / RLE / raw, masks as alpha), dab placement along strokes, flood fill, tiles, mask export / luminance fallback, resize-frame and "Only masked" area maths (mirrors `modules.images` / `modules.masking`), shortcuts, schema → mode layers (`modes.ts`) |
| `engine/` | `EditorDocument` (background, one raster per layer, history, per-layer upload sync), `CanvasRaster` / `MemoryRaster` (tests), `LayerSync` (debounced, hash-cached uploads), `Interaction` (pointer / wheel / pinch), `Renderer` (the only Konva file), `registry` (documents by key) |
| `state/` | tool settings persisted per browser (`remi-canvas-tools`); `activity.ts` (a painting gesture is in progress, readable from the main bundle) |
| `ui/` | the editor chrome: tool rail, options bar, colour popover, layers popover, shortcuts help, empty state, full-screen dialog |
| `img2img/` | the img2img form binding, the Source panel and the stage's Source / Result switch |
| `form/` | schema-declared layers in any form (`SchemaImageEditor`, used by the renderer's image input) and the undoable "replace the background" write |

- **Documents** live in `engine/registry.ts` by `docKey` (`"img2img"`, a
  ControlNet unit…), so layers and history survive unmounting (switching
  pages, the full-screen dialog, the Result view). A layer is one
  full-resolution offscreen canvas; a stroke is drawn into a scratch raster
  and composited once at the brush opacity (no per-stroke nodes). Undo keeps
  the pixels under the touched 256-px tiles before the edit (compressed) and
  the edit itself for redo. Undo and redo act on the edited layer only
  (never a hidden layer of another mode; edits of different layers commute,
  so the newest entry of that layer is taken out of the shared stack). The
  budget is 1/32 of `navigator.deviceMemory` within 96–384 MiB, 160 MiB when
  the browser does not report it, at most 96 MiB on coarse-pointer devices.
  The painted bounds are tracked by the edits ("Only masked" preview); after
  an erase or undo only the previous bounds are scanned.
- **Values**: the editor follows its `image` / layer values (set from outside
  by send-to, paste, reset: a new background clears its painted layers and
  the history). After an edit, once the user pauses (600 ms, then idle), the
  layer is exported as a PNG in wire format (sketch RGBA; mask white + alpha),
  hashed, uploaded unless the same content was uploaded in the last hour
  (undo back to it; older ids may have been pruned by the server), and
  committed with `onLayerChange`. `pendingKey` registers the
  document with `ui/pending-writes`; the run controller flushes it before
  `POST /jobs`, so the job carries what is on screen. Painting waits until
  the document shows the value's image; an image that fails to load clears
  the document and shows an error with Retry. Replacing or removing the
  background flushes the layers first and, when painted layers are dropped
  with it, offers Undo in a toast (`form/replace-background.ts`).
- **Schema layers**: any form's `image` input that `sketch` / `mask` inputs
  of the same owner name as `source` renders as one editor
  (`docKey` `formId:owner:param`, `pendingKey` the form); the layer inputs
  render nothing; the visible ones (own `visible_if`, `hidden`, patches) are
  offered, with a picker when there are several. A `mask` without `source`
  stays an upload field.
- **Modes** come from the schema, not a table (`core/modes.ts`): the image
  input whose name other image inputs' `source` refers to is the background;
  the layer edited is the visible `sketch` / `mask` input; a visible
  `mask_blur` makes a sketch's alpha the inpaint mask. img2img: no layer;
  Sketch / Inpaint sketch: `sketch`; Inpaint: `mask`; Inpaint upload:
  `mask_upload` (uploaded in the Source panel, shown and editable on the
  canvas); Batch: no canvas (the stage shows results).
- **Tools**: brush (size on a log slider, opacity, softness, colour with
  picker / hex / swatches / recent / eyedropper / screen picker), eraser
  (Shift while painting), fill (tolerance; by the layer or the image as
  shown), rectangle and lasso (Shift subtracts), invert mask, clear; masks
  always paint at full strength (the server thresholds at 50 %). Pointer
  events are coalesced and applied once per animation frame; pen pressure
  scales the dab.
- **View**: wheel zooms at the cursor (or pans, a preference; Ctrl+wheel /
  pinch always zoom), Space / right / middle drag / Hand / two fingers pan,
  R or 0 fit, 1 actual size, +/− steps, F full screen. With nothing to
  paint, one finger scrolls the page on touch screens (`touch-action:
  pan-y`); two fingers pan and zoom. The brush cursor shows
  the real diameter; brush sizes are image px, or screen px with
  `forge_canvas_consistent_brush`. Canvases follow `devicePixelRatio` (and
  moves between screens). `forge_canvas_plain(_color)` and
  `forge_canvas_toolbar_always` are honoured.
- **Keys** (pointer over the editor or focus inside it, never while typing):
  B E G M L I H tools, `[` `]` and W / A / S + wheel size / opacity /
  softness, Ctrl/⌘+Z / Y / Shift+Z, Ctrl/⌘+X or Delete clear, Ctrl/⌘+I
  invert, `?` the shortcuts popover.
- **Overlays**: while width / height / resize mode change, the target frame
  of the resize mode (crop / fill shading) shows for a moment ("Resize by":
  the image × scale, snapped like `snap_resolution`; the Source panel also
  fills the `scale_by_resolution` output); mask blur
  (CSS blur at the view scale) and the "Only masked" area (padding and
  processing ratio, as `modules.masking` computes it) show while their
  settings change, or always from the layers popover.
- **img2img**: the parameters pane's Source card has the mode picker
  (balanced tiles, no wrapping strip), one line per image (thumbnail, size,
  "use image size", replace, remove) and the mode's options from the schema,
  flattened into the card (untitled groups lose their box). The stage has a
  Source / Result switch: it turns to Result when the first job of a run of
  this form starts (not for the next job of Generate forever, a repeat or a
  queue, and never during a stroke; a stroke in progress when the editor
  unmounts is committed) and back to Source when a new source arrives or a
  submit is rejected for a source layer. Upload, paste, drop, replace, remove and a blank canvas of
  the target size are on the editor itself.
- **Widgets** (ControlNet units): `noUpload`, `noScribble`, `fixedColor`,
  `fixedSize`, `fixedOpacity`, `fixedSoftness`,
  `layers={{ mask: { kind: "mask", value } }}` + `activeLayer="mask"` for a
  mask-only editor, `height` (default: the "Canvas Height" setting,
  `forge_canvas_height`) or `fill` (fill a sized container), `hint`, `error`.

## Networks (`features/networks`, `pages/networks`)

One browser (`NetworkBrowser`) serves the full-screen Networks page and the
drawer in Generate (header button "Networks", Alt+N, also in the command
palette; its chunk loads on first open). Each surface keeps its own kind,
search, sort, view, card size and per-kind filters (`store.ts`, persisted as
`remi-networks`, with the drawer's open state and width and the folder
tree's width). Until a sort is chosen, `extra_networks_card_order_field` /
`extra_networks_card_order` decide it.

- **Data**: `GET /networks` (kinds with their card action), fixed windows of
  100 items per `{kind, q, folder, archs, compatible, sort, order}`
  (`useNetworkWindow`: the grid reports the indices it shows, debounced
  120 ms, and only the windows covering them — plus half a window around —
  are fetched with `offset`, so a jump to the end of 5,000 items is one
  request and an invalidation refetches only the windows on screen). The
  first window (with the total) of a previous search is kept while a new
  search of the same kind loads, never across kinds; search is debounced
  180 ms. `/facets` feeds the folder tree and the architecture chips.
  `networks.changed` invalidates everything under `["networks"]`, and so does
  `options.changed` for the options the answers depend on (`forge_preset`,
  `extra_networks_default_multiplier`, `lora_preferred_name`, the hidden
  folder options: `NETWORK_OPTION_KEYS`).
- **Grid** (`NetworkGrid`, TanStack Virtual): rows of cards (3:4 preview,
  name, folder, architecture / LyCORIS badges, a check when on) or list rows
  (thumbnail, path, badges, size, date). The overscan is computed so that at
  most 60 cards are mounted (`grid-layout.ts`); items not loaded yet are
  skeletons. A new kind, search, filter or sort mounts a new grid (keyed by
  them): it starts at the top with no cursor. Previews are `/thumbs` at the
  card's device-pixel width snapped to the server's widths (stable URLs while
  resizing), lazy, over the blurred `placeholder`; video previews are muted
  loops (still with `prefers-reduced-motion`); without one a tinted
  monogram. The card / row is the `option`; its hover buttons are a sibling
  overlay outside the accessibility tree, and the item count's live region
  sits outside the listbox. One listbox tab stop (`aria-multiselectable`, selected =
  on): arrows, Page Up/Down, Home/End, Enter/Space apply, I metadata, E
  editor, Shift+F10 / Menu / "." the item menu; "/" focuses the search.
- **Clicking a card** (`actions.ts`): prompt kinds toggle their text in the
  form's `core.prompt` / `core.negative_prompt` (`prompt-insert.ts`: a
  `<type:name:…>` tag matches whatever its weight and takes the activation
  text after it along; other text matches as a whole word; the separator is
  `extra_networks_add_text_separator`). A card with negative text fills both
  prompts; otherwise the target is the prompt editor focused last (read from
  its `promptKind` facet on `focusin`) or the Prompt / Negative switch.
  Writing through the form store lets the editors undo it; on the Networks
  page the target form is chosen in the header and keeps the text until its
  page loads. Checkpoint cards write `sd_model_checkpoint`, VAE / text
  encoder cards toggle `forge_additional_modules`.
- **Item actions** (hover buttons, one shared Base UI menu: at the pointer
  for a right click, under the "…" button, at the card for Shift+F10):
  metadata dialog (file facts, training facts, dataset tags
  as frequency-weighted chips — click copies, "Copy top 40" / "Copy all" —
  and the raw header, filterable), user-metadata editor (the kind's fields;
  dataset-tag chips toggle words in the activation text; random prompt from
  the tags; the preferred weight is bounded like the server, ±100), set
  preview (this session's results, the paged gallery or an upload; a radio
  group with arrow keys; every opening starts with nothing picked), copy
  path, open folder (only with the bootstrap feature `folders.open`, i.e.
  not with `--hide-ui-dir-config`). Edit, set preview and open folder are
  hidden for items without a model file (`editable: false`). The
  checkpoint editor no longer offers upstream's unused "Preferred VAE / Text
  encoder(s)" (`vae_te`) field; the key is kept in sidecars that have it.
- **Filters**: architecture chips with counts, "For <preset>" (default on
  for LoRA when `lora_preset_filter` is set; it filters by the *effective*
  architecture, the user's `sd version` or else the one read from the file,
  where upstream only used `sd version`), the folder (a resizable tree pane
  on wide pages, a popover elsewhere; a WAI-ARIA tree with counts whose
  "All" row counts every folder), "Clear filters". Failed requests show an
  error with Retry. On phones the Networks page puts the form and prompt
  target switches in a row above the browser.
- Mock mode serves 5,000 LoRA in nested folders, 454 of them in a hidden
  `styles/.wip` folder (`api/mock/networks.ts`).

## Settings, quick settings, extensions, server control

- **Settings** (`pages/settings`, `#/settings?view=&q=&filter=&section=`):
  `GET /options` is loaded into one form store (`settings-store.ts`, owner
  `settings`, not persisted) whose defaults are the schema's built-in
  `defaults`, so each control's modified dot resets to the default; the
  stored values are kept beside it and their difference is the unsaved
  edits (marked per row and per nav section, reverted per row, "Unsaved"
  filter, sticky Apply bar, Ctrl/Cmd+S). Apply is one `PATCH /options`,
  optimistic; on any refusal every edit stays an edit (rejected ones show the
  server message under the control) until the refetched values settle it.
  An options reload keeps unsaved edits. Edits survive leaving the page; the
  shell asks before the tab closes, and leaving shows a "Review" toast. Every
  setting renders with the schema renderer inside a virtualized list
  (category / section headings, `settings-model.ts` flattens it); search
  covers keys, labels, notes, links, help HTML text and choice labels, in
  English and the active language, all terms required, with matches
  highlighted through the CSS Custom Highlight API (`highlight.ts`; no DOM
  changes). Filters: All / Changed (from the default) / Unsaved / Pinned.
  The search text and filter live in a module store (`settings-filters.ts`)
  so they survive leaving the page, and are mirrored into `q` / `filter`
  (replace navigation, typing after a pause); a route change made elsewhere
  (a "Review" link, back/forward) wins, the page's own writes are not read
  back. Rows carry notes, links, sanitized help HTML, "Requires restart /
  Reload UI" badges, a refresh button (`refresh:<key>` action) and a pin
  toggle. Settings the server refuses (`locked`, schema-format.md §6) are
  disabled with their reason (not repeated per row under `--freeze-settings`,
  where a banner says it); import skips them. Applied settings that need a reload or restart raise a notice
  with the matching command. Export (changed or all values, JSON) and import
  (ours or an A1111 `config.json`; differences validated with
  `POST /options/validate`, then staged as edits). System views: server and
  models (the old footer's version and API documentation link `/docs`, the
  loaded checkpoint's hash with its CivitAI search link — `model_hash` of
  `GET /server/status`, a link the user follows, nothing is fetched —; the
  footer's startup profile is System info → "Startup"; Reload
  UI, restart, unload / load / load again from disk, the
  weights in memory, hash every checkpoint with progress, notification
  permission, settings file, localization template download), system info
  (summary, collapsible sections, download, copy, file check), licenses
  (structured data; license texts fetched on demand).
  `tools/dump_settings.py` writes the real schema to
  `api/mock/contract/settings.json` (tests and mock mode).
- **Former Settings tabs and actions** (decisions):
  - *Defaults* (the `ui-config.json` editor) is not carried over: Remi forms
    never read `ui-config.json`; their starting values are the schema
    defaults, the UI preset's defaults and the saved form drafts.
  - *Reload custom script bodies* is covered by "Reload UI": the schemas are
    built from the scripts, so reloading script bodies without rebuilding
    them would leave stale forms (and there is no injected JS/CSS to
    refresh).
  - *List loaded models* → Server and models → "Show the weights in memory"
    (`GET /models/loaded`).
  - *Calculate hash for all checkpoints* → "Hash all checkpoints"
    (`/models/hashes`, background, polled progress).
  - *Download localization template* → "Download localization template":
    the English keys of the SPA catalog, the server catalog and the settings
    schema, with the active language's translations filled in.
  - The `quicksettings_list` "info" link (`settingsHintsShowQuicksettings`,
    a popup of all setting names) is replaced by the pin buttons on every row
    and a "Show pinned settings" link under that setting (the Pinned
    filter); the settings search covers keys and labels.
- **Gradio-era settings that were removed, and what replaces each.** Every
  setting left has a reader (`tests/test_remi_option_consumers.py`); these
  only configured the Gradio page and are covered by controls of the SPA
  that the browser remembers:

  | removed setting(s) | replacement |
  |---|---|
  | `prompt_box_style` (Default / Compact / Scrollable / Accordion) | the prompt editors grow with their text up to 24 rem and then scroll (the former Scrollable); the form column of Generate is a resizable pane; the extra networks open in the Networks drawer beside the prompts (Alt+N) rather than the Compact layout's tabs |
  | `txt2img_settings_accordion`, `img2img_settings_accordion` | every section of the form collapses on its own and keeps its open state (`sectionKey`) |
  | `gallery_height` | the result stage fills the page height beside the form (the split between them is a drag-resizable pane) |
  | `do_not_show_images` | not carried over: the Generate page is built around the result stage (turn saving off with `samples_save` to keep results out of the output folder) |
  | `quicksettings_accordion`, `quicksettings_accordion_starts_closed` | quick settings live in the header's ⚡ popover, closed until opened |
  | `extra_networks_card_width`, `_card_height`, `_card_text_scale` | the S / M / L card size and grid / list switches of the network browser, remembered per surface (`remi-networks`) |
  | `extra_networks_card_show_desc`, `extra_networks_card_description_is_html` | descriptions show in the item's metadata dialog as plain text (raw HTML from model files is never rendered) |
  | `extra_networks_tree_view_default_enabled`, `_tree_view_default_width`, `_tree_view_style` (Tree / Dirs), `extra_networks_dir_button_function` | the folder tree pane (toggle and drag-resizable width, remembered) on wide layouts and a folder popover elsewhere; picking a folder filters to it, as the Dirs buttons did |
  | `img2img_sketch_default_brush_color`, `img2img_inpaint_sketch_default_brush_color` | the image editor's colour control (the last colour and eight recent ones are remembered) |
  | `img2img_inpaint_mask_brush_color`, `img2img_inpaint_mask_scribble_alpha`, `img2img_inpaint_mask_high_contrast` | the mask overlay colour (six presets, white and black for contrast) and opacity in the editor's Layers popover, remembered |
  | `keep_alive` | Generate forever is a server-side queue flag: it keeps going with the tab in the background or closed |
  | `live_preview_refresh_period` | previews are pushed over the WebSocket at `live_preview_fps` |
  | `no_spellcheck`, `undo_redo`, `prompt_debounce` | the prompt editor never spellchecks or autocorrects, always has undo / redo (Ctrl/Cmd+Z, Shift+Ctrl/Cmd+Z), and is never delayed (extensions no longer read the DOM) |
  | `remove_image_on_hover` | dropping an image on a filled image field always replaces it |
  | `save_selected_only` | "Save" saves the image on screen; "Save this generation as zip" saves all of it |
  | `send_image_info_not_ui` | "Send to" always sends the parameters recorded in the image (e.g. the expanded wildcards) |
  | `open_dir_button_choice` | "Show in folder" opens the folder of the image itself (local browsers only) |
  | `cross_attention_optimization` (a read-only "Automatic") | the attention backend is chosen by launch flags |
- **Quick settings** (`features/quicksettings`): the header's ⚡ button lists
  `quicksettings_list` minus the model bar's keys (checkpoint, VAE / TE,
  dtype, preset), rendered by the schema renderer in a lazy popover and
  written as they change (continuous kinds debounced; a refusal reverts with
  the message). Pins are set from the settings rows (optimistic
  `PATCH quicksettings_list`).
- **Extensions** (`pages/extensions`, `#/extensions?tab=installed|available|url|backup`):
  installed list (user and built-in apart; switches and update marks are
  staged, then "Save" or "Apply and reload UI"; per-extension or bulk update
  checks with progress, one request each, four at a time, stoppable;
  uninstall with confirmation, never for built-in ones; "Disable all
  extensions"), the extension index (server-fetched, virtualized, search,
  tag hide / only-show with any / all matching, eight sorts, install), install
  from URL (validated URL, branch, folder), and config-state backups (list,
  save, detail against the current commits, restore the extensions, delete;
  both confirmed). The WebUI checkout is never restored from here (a backup
  made before the rewrite points at an upstream commit; `git reset --hard`
  would drop the working tree). Without extension access every write is
  disabled with an explanation. Server errors are shown translated: their
  `template` (English, `{{name}}` placeholders) is a catalog key
  (`adminErrorMessage`). The Available list is virtualized inside the page's
  own scroll (`data-scroll-root`, `scrollMargin`), so there is one
  scrollbar; date sorts put undated entries last.
- **Server control** (`features/server-control`, mounted by the shell):
  "Reload UI" / "Restart server" from the settings page, the palette or
  after applying extension changes; a confirm dialog (warns about unsaved
  edits and a running job), then an overlay that follows the server through
  `/internal/ping` and reloads the page once a ping answers with another
  `instance` than before the command (a server without instance ids: after
  a missed ping); the old server answering again after a missed ping puts
  the overlay back to "stopping". Timeout, "Keep waiting" and "Reload page
  now". `unsaved.ts` is the shared unsaved-edits registry behind the unload
  guard; it is bypassed only right before the page reloads (a dismissed or
  failed command keeps the close-tab prompt).
- Menu and popover triggers with tooltips use `usePopupTooltip`
  (`components/popup-tooltip.ts`): focus returning to the trigger when the
  popup closes no longer reopens the tooltip (the Appearance menu bug).

## Extras, PNG Info, Gallery, Queue

- **Extras** (`pages/extras`, `features/extras`, `features/compare`): the
  `extras` schema rendered as is (source tabs Single Image / Batch Process /
  Batch from Directory / Single Video bound to `mode`, then every
  postprocessing operation's section in its slot: upscale with two upscalers
  and the second's visibility, face restoration, …). Process (Ctrl/Cmd+Enter,
  Ctrl/Cmd+Shift+Enter adds to the queue, Esc interrupts) submits an `extras` job tracked in the progress store
  (`run.ts`; jobs in flight survive a reload, this session's results are kept
  in sessionStorage with the source image of each run). A single-image result
  shows in `CompareSlider` against its source (drag anywhere, or the handle
  with ←/→, Shift for 10 %, Home / End; both images fitted to the frame so an
  upscale compares feature for feature) or alone; batches get a thumbnail
  strip. Result actions (download, save, send to img2img / inpaint / Extras,
  copy parameters), the viewer, "Add to queue".
- **PNG Info** (`pages/png-info`, `features/pnginfo`,
  `#/png-info?file=<id>`): drop, paste (anywhere on the page) or choose an
  image; it is uploaded once and read by `POST /pnginfo`. The infotext panel,
  "send parameters" to txt2img / img2img (page switch), copy, "send image" to
  img2img / inpaint / Extras, and three tabs: parsed parameters, every
  metadata entry grouped (image, text chunks, EXIF IFDs; binary values as
  sizes) and the PNG chunk list. Values are rendered as text only.
- **Gallery** (`pages/gallery`, `features/library`): `GET /gallery` pages
  (120 items) per filter set; filters and the view persist
  (`remi-gallery-view`, the search text is per visit): search (debounced, all
  words in the infotext, name or model), form / folder (txt2img, img2img,
  Extras, grids, saved, other, with counts), model (searchable, counts from
  `/gallery/facets`), date range (local days, presets), starred, sort
  (newest / oldest / name / largest), layout (justified rows or masonry
  columns at each image's exact aspect ratio, `layout.ts`, pure) and tile
  size. TanStack Virtual renders only the rows / tiles near the viewport
  (rows for the justified layout, lanes for masonry) and fetches the next
  page near the end; thumbnails are `/thumbs` at the tile's device-pixel
  width. Keyboard: the grid is one tab stop (roving focus), arrows move
  (↑ / ↓ to the image above / below), Home / End, Enter opens, Space selects
  (Shift extends), F stars; Ctrl/Cmd+A selects everything loaded, Delete
  removes the selection, Esc clears it. Selection (`selection.ts`, pure:
  toggle, Shift ranges from the anchor, pruned when items leave the list)
  shows a floating bar: download as zip (`/gallery/zip`), star / unstar
  (optimistic), compare (exactly two: `CompareDialog`, the slider), remove
  (confirm; "also delete the files" is opt-in). The viewer is the shared
  lightbox with an `aside`: file facts, star, PNG Info, show in folder
  (`/gallery/{id}/reveal`, refused for remote browsers), delete, and the
  infotext panel; its toolbar has the result actions ("use parameters",
  send to, download). `outputs.changed` and finished jobs refresh it;
  "Rescan" indexes the output folders.
- **Queue** (`pages/queue`, `features/queue`): two columns on wide screens
  (queue | history), tabs below 64 rem. The running card (live preview frames
  of the running job, step / batch / ETA, skip, interrupt), "Up next" in run
  order from `GET /queue/items` (`queue.changed` keeps it current) and the
  history. Queued jobs: drag the grip (pointer) or move it from the keyboard
  (Space picks up, ↑ / ↓, Space drops, Esc cancels; announced), move to top /
  bottom, priority (high / normal / low), duplicate, cancel, edit. A move
  across priorities takes the neighbours' priority (`reorder.ts`, pure: the
  list stays in run order) and is shown at once. **Edit** (`edit-session.ts`)
  loads the job's named values into its form (`deliver`, the page switches)
  and, for a queued job, opens an edit session: a banner over the run
  controls offers "Update job" (`PUT /jobs/{id}/params`, validation errors
  land on the fields) or stop editing; a job that started meanwhile leaves
  the values in the form. History: paged (50) and virtualized, search
  (debounced), status (all / completed / failed / stopped), form and starred
  filters; each row shows status, duration, up to four output thumbnails
  (open the viewer), star, queue again (`retry`), open in its form, remove;
  imported jobs show their notes, and jobs that are not replayable say so
  and cannot be queued again as they are. Header: pause / resume and the
  settings menu: import from Agent Scheduler / Prompt Queue (dialog: the
  sources found, one click each, then the report with counts and every task
  that was not imported exactly, with its notes), stop every generate
  forever, stop and cancel everything, clear the history (keeping starred
  jobs by default). **"Add to queue"** (`FormQueueControls.tsx`) sits next
  to Generate / Process in every form: it posts the form as a job without
  making it the form's run (a toast links to the Queue; Ctrl/Cmd+Shift+Enter
  on the Generate and Extras pages, the Agent Scheduler's "Enqueue"
  shortcut with a fixed key). Generate forever and
  repeat N stay the run controller's (client-driven, see "Generate page").
  The Agent Scheduler's queue settings live in Settings → System → Queue
  under their old keys (history retention, queueing failed jobs again, the
  action when the queue is done; server-side, `remi/jobs/queue.py`); its
  layout settings are not carried over (see
  `extensions-builtin/agent-scheduler/README.md`).

## Prompt tools (`features/prompt-tools`)

Tag autocompletion and the former Prompt All-in-One, on `/api/v2/prompt-tools`
(server: `remi/prompt_tools`, `remi/server/routes/prompt_tools.py`; data and
settings come from the built-in `sd-webui-tagcomplete-neo` and
`sd-webui-prompt-all-in-one-neo` extensions, disabling one turns its part off).
`register.ts` (imported by `main.tsx`, small) registers the completion source
and the prompt decorator and refreshes on `options.changed`, `model.loaded`,
`networks.changed` and `job.queued`; everything else is lazy.
`GET /prompt-tools/config` says what is on (`tags.*` mirrors the `tac_*`
settings, `tools.*` the `prompt_tools_*` ones); without it (mock mode, an
older server) nothing is shown.

- **Completion** (`completion/`, lazy chunk on the first keystroke):
  `context.ts` finds the target at the cursor (a tag back to the last
  unescaped delimiter; `<lora:` / `<l:` → LoRA, `<e:` → embedding, `<c:` /
  `<chant:` → chant, bare `<` → all three; `__name` → wildcard names,
  `__name__word` → that wildcard's values; nothing in weights, comments —
  every `#` / `//` to the end of the line and `/* … */`, as Forge strips
  them — or `BREAK`/`AND`; a typed `@artist` prefix is not searched but
  kept for an artist tag). Where it runs: `tags.enabled` (also off for the
  checkpoints of `tac_activeIn_modelList`), `tags.negative`, and per editor
  the `promptScope` facet (`features/prompt-editor/facets.ts`: the form, and
  whether it is an extension's prompt field — `Text` with `prompt: true` — or
  a main prompt — `PromptField`; the style editor has no form) against
  `tags.txt2img` / `tags.img2img` / `tags.third_party`. The server ranks
  (match tier, then posts, then the user's usage; tags found by an alias
  only with `tac_frequencyIncludeAlias`); the client keeps that order
  (`filter: false`), caches queries (LRU, 60 s; cleared on settings / network
  changes) and waits `tags.delay` (`tac_delayTime`, 40 ms) before an
  uncached request. `insert.ts` applies the insertion settings
  (underscores and their exclusions, escaped brackets, `, ` after a tag but
  not before `,` `)` `]` `}` `>` `|` `:`, `<lora:alias:weight>`, activation
  text at the start / end / before / after, Anima's `@` artist prefix, Tab
  completing nested wildcards one folder or to the first difference). An
  accepted item is counted (`POST /usage`, per positive / negative prompt).
  Options show the category colour, alias (`← alias`; with
  `tags.only_alias` the alias alone is the label), translation, kind and
  post count; the info panel has the LoRA preview, description,
  activation text and wiki link.
- **Decorator** (`PromptToolsDecorator.tsx`, in the shell): the editor stays
  mounted and is hidden in the tag view; the tools bar (`PromptToolbar`,
  lazy) offers text ↔ tags, quick tags, history, favorites, translation
  (only with a configured provider; a network provider's API key is entered
  in its menu, `PUT /translation-key`, stored apart from the settings and
  never sent back) and, when the provider is the OpenAI-compatible one
  (`tools.generate`), "Write a prompt" (`panels/GeneratePanel.tsx`: a
  description in any language, optionally changing the current prompt, →
  `POST /generate` with `prompt_tools_generate_instructions` or the built-in
  instructions; the answer is editable, then replaces or extends the
  prompt). `prompt_tools_toolbar` limits the bar to the main
  prompts (`PromptDecoratorProps.main`, set by `PromptField`). The view per
  prompt is persisted (`remi-prompt-tools`).
- **Tag chips** (`chips/`): `parse.ts` turns a prompt into chips and
  separators so `serialize(parse(text)) === text`; weights are effective
  (`((x))` 1.21, `[x]` 0.91, `(x:1.3)`, a network's first number); an edit
  rewrites only its chip (`(x:1.3)` canonical form). A disabled chip is a
  prompt comment `/*x, */` that takes one separator along (offered only with
  `enable_prompt_comments`). `ChipsView`: roving focus (one tab stop),
  ←/→ focus, Alt+←/→ move, Delete removes, Ctrl/Cmd+Z undoes in the hidden
  editor, pointer drag to reorder (touch: horizontal), a popover per chip
  (text, weight slider / number / ±0.1 / reset, enable / disable, favorite,
  delete; a weight is written once, on release / blur), an input with tag
  suggestions (`ui/combobox`; a picked tag is written and counted like a
  completion); moves and removals are announced in a live region.
- **Panels** (lazy, in a popover): quick tag groups (categories, search;
  a tag toggles in the prompt, written like a completion: underscores,
  escaped brackets), history (search, use, append, name, star, delete,
  clear; positive or negative prompts, one item per distinct text: naming
  and deleting cover the whole group), favorites (the prompt's own kind and
  tag snippets, paged by the server; collections, search, save the prompt,
  use / append, rename, move, reorder, delete, JSON export / import of
  everything). Translation: "write the prompt in English" (the non-English
  tags, all replaced on one parse, a translated list becomes several tags,
  weights kept, undoable) and "show the tags in …" (tag view, network
  providers); long lists are sent in chunks.
- **What did not come over.** Tag Autocomplete settings without an
  equivalent are listed with their reasons in the "Not carried over" table of
  `extensions-builtin/sd-webui-tagcomplete-neo/README.md` (a test keeps it in
  step with the old 58 `tac_*` keys); Prompt All-in-One features are mapped
  one by one (kept, merged into a core feature, or dropped and why) in
  `extensions-builtin/sd-webui-prompt-all-in-one-neo/README.md`. Their old
  HTTP APIs (`/tacapi/v1/*`, `/physton_prompt/*`) answer `410 Gone`
  (architecture.md).

## i18n

- Keys are the **English text** (`keySeparator`/`nsSeparator` off).
  Namespaces per locale: `chrome` (bundled `src/locales/<locale>.json`, one
  lazy chunk each) and `server` (`GET /api/v2/i18n/{locale}`, e.g. `zh_CN`,
  for schema labels, choices, info texts and extension strings). `t()` looks
  in `chrome`, then `server`, then returns the key.
- Every UI chrome string goes through `t()` with a **literal English key**:
  `t("Reset to default")`, `t("Step {{step}} of {{steps}}", { step, steps })`.
  Keep constants that are translated later as `label:` / `group:` /
  `title=` / `description=` literals (routes, commands, placeholders). No
  string concatenation, no plural suffixes; write sentences that work
  without plural forms.
- Schema strings are passed to `t()` as-is (`t(node.label)`); `html` node text
  is never translated, except a setting's HTML comment, which is looked up
  whole (the language packs translate them) and sanitized like the original.
- `src/locales/zh-CN.json` must contain every chrome key:
  `locales/locales.test.ts` scans the sources and fails on a missing key or a
  mismatched `{{placeholder}}`.
- Locale ids: BCP 47 in the app (`zh-CN`, `<html lang>`), underscore form for
  the server (`zh_CN`). The user's choice persists (`remi-locale`); without
  one, the server default (`Bootstrap.locale`) then the browser language is
  used. Switching does not reload the page.

## Mock mode

`pnpm dev:mock` (or `VITE_MOCK=1`): `api/mock/server.ts` answers every
`/api/v2` call from `api/mock/fixtures.ts`; `api/mock/socket.ts` replaces the
WebSocket and the simulated worker emits `job.*` events and real WebP
preview frames; a dev-server plugin serves generated SVG artwork for
`/api/v2/files|thumbs/mock-*`. The txt2img fixture mirrors the shape of the
real schema — `sampler`, `seed` and `refiner` are scripts in the `sampler`,
`seed` and `accordions` slots, Hires. fix is a core section whose enable
toggle is exclusive, has `on_change: hr_resolution` and an output line, and
`conflict_params` lists the lockable params — and covers every node kind and
condition operator, cross-owner conditions, an unknown slot, a selectable
script, a script with an untitled group of JSON params, a
bound tabs node, an Anima-like section (enable toggle, preset action,
exclusive locks, custom widget) and a ControlNet-like section (uploads,
`on_change`, sanitised HTML). The mock `__conflicts__` reads the `scripts`
context (Anima locks one-of-two, Hires. fix vs refiner, LCM pins its
schedule). `POST /jobs` validates like §3 and answers 422 `{"errors": …}`
(try `#invalid` in the prompt, or img2img without a source image);
`GET /jobs` is paged (`cursor`, `limit`), `repeat` queues copies and retry
works; interrupted jobs end as `job.completed` with status `interrupted`.
`/prompt/tokens`, `/styles` (+ `/styles/apply`), `/infotext/parse`,
`/infotext/image` and the `__forge_preset__` action answer too; model
changes emit `model.loading` stages. There is a
schema-rendered extension page (`Toolbox`) and a zh_CN server catalog
sample. `api/mock/library.ts` serves the Gallery (260 outputs of varied
sizes, forms, models and dates; stars, removal, zip, reveal), PNG Info and
the Queue tools (duplicate, edit, star, clear, an import report); the
simulated worker honours pause / resume. Production builds contain no mock
code.

`src/api/mock/contract/{txt2img,img2img,extras}.json` are the real
`tab_schema()` outputs (`tools/dump_schemas.py`, default options and the
built-in processing scripts). Tests only; never bundled.

## Tests

`tools/check.sh` runs every gate (Python and SPA) in one command; see
[testing.md](testing.md) for the steps, the end-to-end and axe suites, the
real-server smoke test and CI.

`pnpm test` runs Vitest in jsdom: condition evaluator (incl. Python
semantics), form store (drafts and their coercion, reset, patches,
validation errors and reveal), action requests and response splitting,
change effects (conflict flow with the scripts context, `on_change`
debounce, bulk resets, `__settings__`, stale-response drop), HTTP error
mapping (422), WS event validation, frame parser, WS store (handshake,
dispatch, preview URL revocation, reconnect, overlapping connects),
ws-bridge (schema refresh, preview clearing, per-form results, model
loading), router (incl. malformed hashes) and nav, Markdown/HTML
sanitising, attention/reorder editing and the CodeMirror key bindings,
renderer smoke tests for every node kind, tabs visibility/enablement,
upload preview revocation, `jobRequest`, the generate page (submit, 422
marking, Ctrl+Enter, interrupt), locale switching, zh-CN catalog coverage,
and **contract tests** rendering the real txt2img / img2img / extras
schemas (no console errors, every default round-trips, conflict params and
slots resolve). Generate view: prompt tokenizer and bracket problems,
editor commands on a real `EditorView` (weights with precisions, tag moves,
undo, lint, completion registry), the run machine (every transition),
restore after reload (incl. events during the lookup and submits during
the restore), resync after a reconnect / missed events, double submits with
interrupt, infotext paste planning (seed / size filters, one
`on_change` per action, one conflicts pass), send-to mapping and delivery,
image metadata reading (PNG chunks incl. zlib, JPEG / WebP EXIF), the
curated arrangement of the real schemas (and the user's order), size
helpers, lightbox zoom / pinch math, keyboard and pinch on the component,
gamepad mapping, history grid keys and `aria-activedescendant`, model bar
modules commit (incl. the picker nested in the narrow header's popover),
token counter debounce and recount after a model load, notifications,
reset confirmation, queue position,
form reload carry-over, and the page (submit, 422 marking, keys, history).
Library pages: justified / masonry layout and the above / below neighbour,
selection ranges and pruning, gallery filters → query (local day bounds),
the gallery page (rows at aspect ratio, check boxes and bulk star / delete,
Shift-click ranges, Escape, the roving keyboard focus, filters sent), the
compare slider (keyboard, clamping), Extras runs (single-source tracking,
finish / restore after a reload) and page, PNG Info (upload, tables, send),
queue reorder planning across priorities, the pending list (keyboard move
and its requests), edit sessions (update, 422 marking, a job that started)
and "Add to queue".
Image editor: undo history (order, ≥ 32 states, byte budget, per-layer
removal), patches round-trip, mask export (white + exact alpha whatever the
tint, luminance fallback), flood fill, dab placement (batch-independent,
pressure), view maths, overlays against the server's crop maths, shortcuts,
the document on memory rasters (strokes, opacity, erase, invert, clear, fill,
external loads, sampling, per-layer undo / redo, tracked bounds, upload sync
with hash reuse, its expiry and flush), the undo budget, input (a stroke
committed on unmount, one-finger touch left to the page), toolbar keys and
`aria-keyshortcuts`, the editor's value binding (`Editor.test.tsx`: uploaded
mask loading, send-to, a failed load with Retry, layer commits), schema
layers through the renderer, the real img2img schema → mode layers, the
Source panel (balanced mode grid, no nested frames, mode options, errors,
the "Resize by" size), source / size writes (undo toast, flush first),
stage view switches (first job of a run only, never mid-stroke) and the
submit waiting for layer uploads; pytest
(`tests/test_remi_canvas_layers.py`) for the server's layer composition and
the Source area's zh-CN catalog.
`tests/e2e` (Playwright, mock mode): generate + history, Escape interrupt,
editor keys and bracket warning, mode switch, viewer, send to img2img, the
phone layout, the image editor (`canvas.spec.ts`: paint / undo / redo a
mask kept across the stage views with the job carrying it, sketch with Shift
erasing, zoom / fit / full screen / shortcuts, an Inpaint-upload mask on the
canvas across mode switches with per-layer undo and the job's layer ids,
drop and paste, image size and removal, touch scrolling on phones), and the library (`library.spec.ts`): gallery search and viewer
details, two-image compare, PNG Info → txt2img, and the queue (pause, "Add
to queue" twice, keyboard reorder, resume, history). TypeScript runs with `noUncheckedIndexedAccess`.

## Performance

Budget: shell JS ≤ 200 KB gzip. Generate view figures measured 2026-10-15
(wave 2): a route's size is its chunk plus every statically imported chunk
that is not in the initial shell. The build groups everything the first paint
needs into `vendor` (node_modules) and `shell` chunks (`codeSplitting` groups
with the `$initial` tag in `vite.config.ts`); everything else is lazy:
pages, the prompt editor, the command palette, menus, the model picker,
widgets, locale catalogs. Measured 2026-10-15 (`pnpm build`, `gzip -9`):

| | raw | gzip |
|---|---|---|
| initial JS (runtime + `vendor` + `shell`) | 515 KB | **167 KB** |
| CSS (incl. `@font-face`) | 136 KB | 24 KB |
| `generate` page + schema renderer + features (all its static chunks) | 501 KB | 177 KB |
| lightbox (lazy; on top of the generate route) | 13 KB | 6 KB |
| image editor (Konva + editor UI; lazy, preloaded by img2img) | 209 KB | 67 KB |
| style editor (lazy; on top of the generate route) | 14 KB | 6 KB |
| model bar (own code; shared with the header) | 15 KB | 6 KB |
| `queue` page (own chunk, wave 3) | 53 KB | 20 KB |
| `gallery` page (own chunk; TanStack Virtual is shared) | 46 KB | 19 KB |
| `extras` page (own chunk; the schema renderer is shared with generate) | 16 KB | 7 KB |
| `png-info` page (own chunk) | 18 KB | 7 KB |
| prompt editor (CodeMirror + lint + autocomplete) | 328 KB | 105 KB |
| command palette (cmdk + dialog) | 59 KB | 20 KB |

Fonts load per unicode range (Inter latin ≈ 48 KB woff2).

## Backend contract assumptions (to confirm)

- `GET /app` returns `Bootstrap` (generated from the server's model).
- Schema-rendered extension pages: `GET /api/v2/pages/{id}` →
  `{version, id, title, owner: "page:<id>", layout, defaults, on_load?,
  send_to?}` (schema-format.md §8).
- WS job events carry `job_id` and `seq` (= binary `job_seq`); `job_no` is
  0-based; `tab` is optional. Binary preview headers are little-endian.
- Action bodies carry `scripts: {owner id: values}` (every non-core owner
  except the caller) next to `values` / `core`; the server fills
  `ActionContext.scripts` from it (needed by `__conflicts__`). Not yet in
  schema-format.md §5.
- `__settings__` in an action patch reports settings the server already
  stored (schema-format.md §5); the client only updates its cache.
- A model load ends with `model.loaded {model?, error?}` (or a
  `model.loading` stage `done`); not yet in architecture.md.
- `GET /jobs?cursor=&limit=` returns `{items, next_cursor, total?}` newest
  first.
- Uploaded images are previewed from a local object URL; ids from elsewhere
  are shown via `/api/v2/files/{id}`, so upload ids should resolve there.
- The checkpoint quick-bar writes `sd_model_checkpoint` with `PATCH /options`
  and shows `model.loading{stage}` while a model loads.
