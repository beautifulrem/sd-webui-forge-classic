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
| `pnpm lint` | oxlint (must be silent) |
| `python3 tools/shadcn_fetch.py <names>` | add shadcn registry components with curl (Node cannot use the proxy) |
| `/tmp/remi-testenv/bin/python tools/dump_schemas.py` | re-dump the real tab schemas into `src/api/mock/contract/*.json` (contract tests) after a `remi/ui` / `remi/core` change |

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
pages/<page>/       one lazily loaded chunk per page; pages/custom/<name>/ = dedicated
                    views for extension pages with `widget=`
components/         app-level components (SearchSelect, Segmented, ErrorBoundary, …)
components/ui/      vendored shadcn components (edit sparingly; lint-exempt)
i18n/               i18next setup, locale store, catalog loading
locales/            chrome translations, one JSON per locale (zh-CN.json)
lib/                appearance store (theme.ts), toast, media query, cn()
test/               setup.ts (jsdom shims, jest-dom, i18n init), fake-socket.ts (WebSocket double)
```

`src/canvas/` (Konva) arrives with the img2img editor; it is the only module
that may import Konva. `konva`, `react-konva`, `@codemirror/autocomplete` and
`@codemirror/lint` are installed for wave 2 (canvas editor, tag completion,
prompt diagnostics) and are not imported yet, so they cost nothing in the
bundle.

## App shell

- **Routing**: `app/router.ts` is a typed hash router (`#/txt2img`, `#/img2img`,
  `#/extras`, `#/png-info`, `#/gallery`, `#/networks`, `#/queue`,
  `#/settings`, `#/extensions`, `#/page/<id>`, optional `?query`). Hash routing
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
- **Toasts**: `sonner`; use `showToast(level, message)` from `lib/toast.ts`
  for server-provided levels, or `toast` directly.
- **Appearance** (`lib/theme.ts`, persisted as `remi-appearance`): theme
  (light/dark/system), density (comfortable/compact), accent
  (`iris|blue|teal|green|amber|coral|rose`). `index.html` applies the stored
  values before first paint; `useApplyAppearance` keeps `<html>` in sync.

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
- Scrollbars: thin, theme-aware (`--scrollbar`). Focus: visible rings from
  `--ring` everywhere (`outline-ring/50`, `focus-visible:ring-3`).
- shadcn components import `cn` from the `cn` package (what the base-nova
  registry ships); app code imports it from `@/lib/utils`. Local fixes to
  vendored components: scalar sliders render one thumb and forward
  `getAriaLabel`; `number-field.tsx` is ours.

## API layer (`src/api`)

- `http.ts`: `request()`/`http.*` with JSON or multipart bodies, `ApiError`
  (status, decoded body in `detail`, a readable message from Remi's
  `{"errors": [{owner, param, message}]}` or FastAPI's `detail`),
  `validationErrors(error)` for the per-field list, and the mock switch.
- `endpoints.ts`: **the only place that builds URLs** (`api.app()`,
  `api.schema(tab)`, `api.action(owner, action, body)`, `api.jobs.*`,
  `api.upload(file)`, `api.fileUrl(id)`, `api.thumbUrl(id, w)`, …). When
  `webui/src/api/gen` is generated from `/openapi.json`, re-implement `api`
  on top of it and turn `types.ts` into re-exports; callers do not change.
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
  job events, `options.changed` (incl. `["schema"]` and `["page"]`: core
  forms follow the options) and everything after a reconnect. Model loading
  ends on `model.loaded`, a `model.loading` stage `done|loaded|error|failed`,
  a reconnect, or after 180 s without news.

## Schema renderer (`src/ui`)

`<TabForm schema={TabSchema}/>` renders a generation tab (core form +
scripts); `<PageForm schema={PageSchema}/>` renders one owner (extension
pages). `jobRequest(store, schema)` (`ui/tab.ts`) builds the `POST /jobs` body.

| file | role |
|---|---|
| `schema.ts` | TypeScript mirror of schema-format.md |
| `Renderer.tsx` | `RenderNode`: dispatch by `type`, `visible_if`/`enabled_if`, patch visibility, row cells |
| `inputs/*` | one component per input kind: Toggle, Slider (+ number box), Number, Text (`prompt` → lazy `PromptEditor`), Choice (select / segmented or radio list / searchable), MultiChoice (chips or checkboxes), Seed (dice / recycle), Color, Image, ImageList, File (`DropZone`: drop, paste, click → `POST /uploads`), Json, Custom |
| `layout/*` | Section (enable toggle in header, open state, section reset), Group, Tabs (`bind`), ActionButton, Gallery/Output, Slot, row sizing |
| `Field.tsx` | label row, modified dot (click = reset to default, tooltip shows the default), info text, server validation message (`FieldError`) |
| `a11y.ts` | ids of label / info / error texts and `describedBy(id, info, error)` for controls |
| `conditions.ts` | condition evaluator (`param` / `eq` / `ne` / `in` / `gt…lte` / `all` / `any` / `not`, `"owner.param"`) |
| `form-store.ts` | per-form Zustand store (`getFormStore(formId)`) |
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
  `__toast__` → toast, `__outputs__` → the owner's outputs, `__options__` →
  `PATCH /options` by the client (the server only returns them). A newer
  call of the same action wins (stale responses are dropped).
- **Change effects** (`runChangeEffects`): after a user edit (or reset) of a
  param, its `on_change` action runs — immediately for discrete kinds,
  debounced 300 ms for slider / number / text / color / json — and, when
  `"owner.param"` is in the tab schema's `conflict_params`, so does
  `core.__conflicts__` with `selected` = that reference. This covers section
  enable toggles, choices, toggling off, and resets. A section reset or
  "Reset all parameters" runs each distinct `on_change` once and one
  conflicts call (`selected` only when a single conflict param changed).
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
  whose layout is not a section or titled group (e.g. a legacy script's
  untitled group of `#i` inputs, the `sampler` row) gets a titled frame; an
  untitled group's own box is replaced by it.
- **Uploads**: images are previewed from object URLs; removing / replacing
  an image revokes its URL once no form store references the id, older
  unreferenced previews are swept before each upload, and at most 24 are
  kept (then `/files/{id}` is used).
- **Keyboard**: Ctrl/Cmd+Enter generates (ignores key repeat and IME
  composition). Interrupt and Skip are buttons and palette commands; a bare
  Escape never interrupts (it closes popups and editors).
- **Rows**: `width` 1–12 = share of 12 columns, `"fill"` = the rest,
  absent/`"auto"` = equal share, buttons hug content. Rows are CSS containers;
  under 30 rem spans and fills go full width (phones stack).
- **Prompt editor** (`ui/prompt/`, CodeMirror 6, lazy chunk): highlighting of
  weights / `<lora:…>` / brackets / BREAK/AND, Ctrl/Cmd+↑/↓ attention editing
  (ported from `edit-attention.js`, precision from `keyedit_*` options),
  Alt+←/→ tag moves (`keyedit_move`), undo/redo history (600 ms grouping).
  Ctrl/Cmd+Enter is left to the page (Generate). The pure editing functions
  are in `attention.ts` / `reorder.ts`.

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
lost. Example: `widgets/block-weights`.

### Adding a page

- Built-in page: create `src/pages/<page>/index.tsx` (default export
  `({ route })`), add the route name to `router.ts`, its loader in
  `routes.tsx` and a `BUILTIN_NAV` entry.
- Extension page with a dedicated view (`Page(..., widget="x")`): create
  `src/pages/custom/x/index.tsx` (default export `({ page })`). Without
  `widget`, the page is schema-rendered from `GET /api/v2/pages/{id}`.
- Extension page icons are names from the curated map in `app/icons.ts`
  (unknown → puzzle piece).

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
  is never translated.
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
script, a legacy positional script (untitled group, like `legacy_layout`), a
bound tabs node, an Anima-like section (enable toggle, preset action,
exclusive locks, custom widget) and a ControlNet-like section (uploads,
`on_change`, sanitised HTML). The mock `__conflicts__` reads the `scripts`
context (Anima locks one-of-two, Hires. fix vs refiner, LCM pins its
schedule). `POST /jobs` validates like §3 and answers 422 `{"errors": …}`
(try `#invalid` in the prompt, or img2img without a source image);
`GET /jobs` is paged (`cursor`, `limit`) and retry works. There is a
schema-rendered extension page (`Toolbox`) and a zh_CN server catalog
sample. Production builds contain no mock code.

`src/api/mock/contract/{txt2img,img2img,extras}.json` are the real
`tab_schema()` outputs (`tools/dump_schemas.py`, default options and the
built-in processing scripts). Tests only; never bundled.

## Tests

`pnpm test` runs Vitest in jsdom: condition evaluator (incl. Python
semantics), form store (drafts and their coercion, reset, patches,
validation errors and reveal), action requests and response splitting,
change effects (conflict flow with the scripts context, `on_change`
debounce, bulk resets, `__options__`, stale-response drop), HTTP error
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
slots resolve). TypeScript runs with `noUncheckedIndexedAccess`.

## Performance

Budget: shell JS ≤ 200 KB gzip. The build groups everything the first paint
needs into `vendor` (node_modules) and `shell` chunks (`codeSplitting` groups
with the `$initial` tag in `vite.config.ts`); everything else is lazy:
pages, the prompt editor, the command palette, menus, the model picker,
widgets, locale catalogs. Measured 2026-10-15 (`pnpm build`, `gzip -9`):

| | raw | gzip |
|---|---|---|
| initial JS (runtime + `vendor` + `shell`) | 526 KB | **170 KB** |
| CSS (incl. `@font-face`) | 129 KB | 23 KB |
| `generate` page + schema renderer | 231 KB | 78 KB |
| `queue` page (incl. TanStack Virtual) | 34 KB | 11 KB |
| prompt editor (CodeMirror) | 277 KB | 91 KB |
| command palette (cmdk + dialog) | 59 KB | 20 KB |

Fonts load per unicode range (Inter latin ≈ 48 KB woff2).

## Backend contract assumptions (to confirm)

- `GET /app` returns `Bootstrap` as typed in `api/types.ts`:
  `{version, features, tabs, pages: [{id, title, icon?, widget?}],
  locales: [{id, name}], locale?, options, models: {checkpoint?, …}}`.
- Schema-rendered extension pages: `GET /api/v2/pages/{id}` →
  `{version, id, title, owner: "page:<id>", layout, defaults}` (not in the
  architecture table yet).
- WS job events carry `job_id` and `seq` (= binary `job_seq`); `job_no` is
  0-based; `tab` is optional. Binary preview headers are little-endian.
- Action bodies carry `scripts: {owner id: values}` (every non-core owner
  except the caller) next to `values` / `core`; the server fills
  `ActionContext.scripts` from it (needed by `__conflicts__`). Not yet in
  schema-format.md §5.
- `__options__` in an action patch is applied by the client with
  `PATCH /options`; the server must not save it as well.
- A model load ends with `model.loaded {model?, error?}` (or a
  `model.loading` stage `done`); not yet in architecture.md.
- `GET /jobs?cursor=&limit=` returns `{items, next_cursor, total?}` newest
  first.
- Uploaded images are previewed from a local object URL; ids from elsewhere
  are shown via `/api/v2/files/{id}`, so upload ids should resolve there.
- The checkpoint quick-bar writes `sd_model_checkpoint` with `PATCH /options`
  and shows `model.loading{stage}` while a model loads.
