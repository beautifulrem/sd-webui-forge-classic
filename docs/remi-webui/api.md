# Remi WebUI HTTP and WebSocket API

The Remi server is one FastAPI application (`remi/server/app.py`). It serves:

| prefix | what | enabled |
|---|---|---|
| `/api/v2` | the WebUI's own REST API (what the SPA uses) | always |
| `/api/v2/ws` | WebSocket: job events, queue state, binary live previews | always |
| `/internal` | small A1111-compatible helpers (`ping`, `progress`, `pending-tasks`, `sysinfo`, …) | always |
| `/sdapi/v1` | the AUTOMATIC1111 / Forge API (`modules/api/api.py`) | with `--api` or `--nowebui` |
| `/docs`, `/redoc`, `/openapi.json` | interactive API description of everything above | always |
| everything else | the SPA (`webui/dist`) | not with `--nowebui` |

The OpenAPI document is the authoritative list of routes, bodies and
responses: every `/api/v2` body and response is a Pydantic model
(`remi/server/models.py`). `tools/export_openapi.py` writes it to
`webui/src/api/openapi.json` (the SPA's typed client is generated from it;
`--check` fails when it is stale). This page explains the parts the schema
cannot: the protocol, the value model, auth, and worked examples. The full
route table with semantics is in [architecture.md](architecture.md#http-api-v2-apiv2);
the JSON shapes of forms, conditions, actions and pages are in
[schema-format.md](schema-format.md).

Contents: [Conventions](#conventions) · [Authentication](#authentication) ·
[Generating](#generating-images) · [Routes by area](#routes-by-area) ·
[WebSocket](#websocket-apiv2ws) · [`/sdapi/v1` compatibility](#sdapiv1-compatibility) ·
[Examples](#examples)

## Conventions

- **Base URL.** `http://127.0.0.1:7860` by default (`--port`, `--listen`,
  `--server-name`). With `--subpath /sd` every path below is prefixed:
  `/sd/api/v2/app`. With `--nowebui` the default port is 7861.
- **JSON** everywhere except uploads (multipart) and files. No `NaN` or
  `Infinity` is accepted or produced.
- **Errors.** A rejected parameter value answers **422** with every problem
  at once:

  ```json
  {"errors": [{"owner": "core", "param": "width", "message": "expected a number, got 'abc'"}]}
  ```

  `owner` is `core`, a script id, `settings`, a page owner `page:<id>`, or
  `null` when the body itself has the wrong shape. Other errors are
  `{"detail": "..."}` with the usual status codes (404 unknown id, 409 wrong
  state, 413 too large, 415 unsupported type, 403 refused, 401 not logged in).
  Extension-management errors add `template` and `params` (a translatable
  message with `{{name}}` placeholders).
- **Paging.** Lists return `{"items": [...], "next_cursor": "...", "total": n}`;
  pass `cursor=<next_cursor>` for the next page. Cursors are opaque.
- **Ids.** Jobs, uploads and output files have opaque string ids. Files are
  only ever served by id (`/api/v2/files/{id}`), never by path.

## Authentication

Without `--auth` / `--auth-file` the server is open (as the old WebUI was
without `--gradio-auth`). With logins configured:

- `/api/v2`, `/internal`, the WebSocket, `/api/v2/files/*`, the SPA and the
  API docs need either the **session cookie** or **HTTP Basic** credentials
  of one of the `--auth` users. Only `/api/v2/auth/*`, `/internal/ping` and
  the SPA's static assets are public.
- Log in with `POST /api/v2/auth/login` `{"username", "password"}` → 204 and
  a `remi_session` cookie (signed, 30 days, `HttpOnly`; `Secure` with
  `--tls-*` or when the request came over HTTPS). Wrong credentials → 401;
  ten failures from one client within a minute (login form or Basic) → 429
  for a minute. `POST
  /api/v2/auth/logout` revokes the cookie server-side. `GET
  /api/v2/auth/status` → `{"required": bool, "user": str | null}`.
- Scripts and tools may skip the cookie and send `Authorization: Basic …`
  on every request (also on the WebSocket handshake).
- `/sdapi/v1` (and extension routes under `/sdapi`) uses `--api-auth
  user:pass[,…]` (HTTP Basic, throttled the same way) when it is set; without
  it, the `--auth` login (session cookie or HTTP Basic) on any bind, since a
  reverse proxy or tunnel on the same machine also connects from loopback.
  `--api-no-auth-on-localhost` restores the A1111 behaviour for local tools:
  requests from a loopback peer that came through no proxy (no
  `X-Forwarded-*` / `Forwarded` / `X-Real-IP` header) need no login. It has no
  effect with `--api-auth`, which always applies.
- Extension API routes (`/controlnet/*`, `/adetailer/*`, ...) follow the same
  rule: with `--auth` they need the WebUI login or `--api-auth` credentials,
  local clients exempt only with `--api-no-auth-on-localhost`.
- The API docs need the `--auth` login, or `--api-auth` credentials when only
  that is set.

Independently of logins, these checks protect against browsers being abused
(details and the threat model: [security.md](security.md)):

- **Cross-site requests are refused (403).** A state-changing request
  (`POST`/`PUT`/`PATCH`/`DELETE`) or WebSocket handshake whose `Origin` is
  another site is refused unless that origin is allowed with
  `--cors-allow-origins` / `--cors-allow-origins-regex`. Clients that send
  no `Origin` (curl, Python, other servers) are not affected.
- **DNS rebinding is refused (403 "Unknown host refused").** When the server
  only listens on loopback (no `--listen`, `--server-name`, `--share`,
  `--ngrok`), requests must use a loopback `Host` (`localhost`,
  `127.0.0.1`, `[::1]`) or a host allowed through `--cors-allow-origins`.
- **Cross-site subresource reads are refused (403).** A `GET` that the
  browser's fetch metadata marks as sent by another site's page (an
  `<img>`, a `fetch`, a frame) is refused unless its origin is allowed;
  links and bookmarks (top-level navigations) and API clients pass.

These checks cover `/api/v2`, `/internal`, the API docs and `/sdapi`, and the
routes extensions add (`modules/extension_route_guard.py`).

## Generating images

### The value model

A generation is a **tab** (`txt2img`, `img2img` or `extras`) plus named
values per **owner**:

- `core`: the generation form (`prompt`, `negative_prompt`, `width`,
  `height`, `cfg_scale`, `batch_size`, `batch_count`, `enable_hr`, `hr_*`,
  `styles`, `script`, img2img `mode`, `init_img`, `mask`, `denoising_strength`,
  …).
- one entry per **script**, keyed by its script id: `sampler` (`steps`,
  `sampler_name`, `scheduler`), `seed` (`seed`, `subseed`, `subseed_strength`,
  …), `refiner`, and every extension panel (`anima_dave_diversity`,
  `controlnet`, `adetailer`, …).

`GET /api/v2/schema/{tab}` returns the form: the core layout, every script
with its `id`, `layout`, `defaults` and `args` (the positional order the
script's processing hooks receive). **Names you leave out take their
defaults**, so a request only needs what it changes. The server validates and
coerces every value before anything is queued (sliders are clamped, numbers
outside their bounds and unknown choices are rejected; schema-format.md §3).

`GET /api/v2/app` (the bootstrap) also lists `not_ported`: scripts and
extension pages that could not be loaded (for example a third-party script
that still declares A1111's Gradio `ui()`), with the reason.

### Jobs and the queue

All generations go through one persistent queue (`remi-data/jobs.db`) and one
worker, so the SPA, the Queue page and API clients see the same jobs.

```
POST /api/v2/jobs   {"tab", "core", "scripts", "priority"?, "repeat"?, "forever"?, "upscale_from"?}
  → 201 {"job_id", "job_ids", "seq"}
```

- `priority` (−1000 … 1000, default 0): higher runs first; the Queue page uses 10 /
  0 / −10 for high / normal / low.
- `repeat` (1–100): queue that many copies (`job_ids` lists them).
- `forever`: re-queue the job each time it completes until
  `POST /api/v2/queue/stop-forever` or `PATCH /api/v2/jobs/{id}`
  `{"forever": false}`.
- `upscale_from` (txt2img only): an output or upload id. The job Hires. fixes
  that image instead of generating a first pass (A1111's ✨ button under the
  gallery; the SPA's "Upscale with Hires. fix" result action): Hires. fix is
  turned on, the size is the image's (the form's goes to the infotext as
  `Original Size`), `txt2img_upscale_single_batch` makes it one image and
  `txt2img_upscale_same_seed` takes the seeds from the image's parameters.
  Retries keep it. An unknown id, a non-image or another tab is a 422 on
  `upscale_from`.

A job's `status` is `queued`, `running`, `completed`, `interrupted` (stopped,
results kept), `cancelled` or `failed`. `GET /api/v2/jobs/{id}` returns it
with `outputs` (`[{id, kind, width, height, infotext, role?}]`; `kind`
`video` for a generated video (e.g. Wan), next to its frames; `role` `grid`
or `extra` (e.g. a ControlNet detected map) for images that are not one of
the batch), `error` /
`errors`, `progress`, and the `core` / `scripts` values it ran with. Output
ids are file ids: `GET /api/v2/files/{id}` (add `?download=1` for an
attachment), `GET /api/v2/thumbs/{id}?w=256` for a WebP thumbnail.

Control: `POST /api/v2/jobs/{id}/interrupt` (a queued job is cancelled),
`/skip`, `/retry` (queue a copy), `/duplicate`, `PUT /api/v2/jobs/{id}/params`
(replace a queued job's values), `DELETE /api/v2/jobs/{id}` (cancel or
remove), `POST /api/v2/interrupt` (whatever is generating, also an `/sdapi`
request; it also ends every generate-forever job, and `{"clear_queue": true}`
cancels the queued jobs), `POST
/api/v2/queue/pause` / `resume`, `POST /api/v2/queue/move`, `GET
/api/v2/queue` (`{size, paused, running}`) and `GET /api/v2/queue/items`
(queued jobs in run order). `GET /api/v2/jobs?status=&q=&tab=&starred=` is
the history, newest first.

### Images as input

Upload once, then pass the id:

```
POST /api/v2/uploads  (multipart, field "file")
  → 201 {"id", "width", "height", "name", "mime", "size"}
```

Uploads are identified by their content (sniffed, not by extension): images
(PNG, JPEG, WebP, GIF, BMP, TIFF, AVIF, HEIC, JXL), videos (MP4, MOV, WebM,
AVI), safetensors files and small text files (`.txt`, `.csv`, `.json`,
`.yaml`, `.cube`). The size limit is the `remi_upload_max_mb` setting
(default 64 MB; 413 above it, 415 for other types). Identical uploads are
deduplicated. Uploads (and outputs the save path did not write) older than
`remi_upload_retention_days` are pruned by the hourly housekeeping.

An `image` param takes an upload id (or an output file id). img2img needs
`init_img` in every mode but "Batch"; "Inpaint" also needs `mask` (alpha 255
= inpaint; a mask without transparency is read by luminance, white =
inpaint), "Inpaint upload" needs `mask_upload`, "Inpaint sketch" needs
`sketch`.

### Infotext and presets

- `POST /api/v2/infotext/parse` `{"text", "tab"?}` turns generation
  parameters text into `{tab, core, scripts, params, problems}`: named values
  you can send straight to `POST /jobs`. `POST /api/v2/infotext/image`
  `{"file_id"}` reads them from an uploaded or generated image.
- `GET /api/v2/presets` lists the Anima presets and UI presets;
  `GET /api/v2/presets/{name}?tab=txt2img` returns an Anima preset's patch.

### Actions

Buttons and change handlers of forms and pages are server **actions**:
`POST /api/v2/actions/{owner}/{action}` with `{"tab", "values", "core"?,
"scripts"?}` returns a patch (`{param: {"value"?, "choices"?, "visible"?,
…}, "__toast__"?, "__outputs__"?, "__settings__"?}`, schema-format.md §5).
`owner` is a script id, `core`, `page:<id>` or `settings`. Example: the size
swap button is `POST /api/v2/actions/core/swap_dimensions`.

## Routes by area

Paths are relative to `/api/v2`. See [architecture.md](architecture.md#http-api-v2-apiv2)
for the detailed semantics of each route and `/docs` for the bodies.

| area | routes |
|---|---|
| bootstrap and forms | `GET /app`, `GET /schema/{tab}`, `GET /pages`, `GET /pages/{id}`, `POST /actions/{owner}/{action}`, `POST /rules/derive`, `GET /presets[/{name}]`, `GET /i18n/{locale}` |
| jobs and queue | `POST/GET /jobs`, `GET/PATCH/DELETE /jobs/{id}`, `POST /jobs/{id}/interrupt\|skip\|retry\|duplicate`, `PUT /jobs/{id}/params`, `POST /jobs/star`, `POST /jobs/clear-history`, `GET /queue`, `GET /queue/items`, `POST /queue/reorder\|move\|pause\|resume\|stop-forever`, `POST /interrupt`, `GET /queue/import`, `POST /queue/import/{agent-scheduler\|prompt-queue}` |
| files | `POST /uploads`, `GET /files/{id}`, `GET /thumbs/{id}`, `GET /outputs`, `DELETE /outputs/{id}`, `POST /outputs/save\|refresh\|open-folder` |
| gallery and image info | `GET /gallery`, `GET /gallery/facets`, `GET /gallery/{id}`, `POST /gallery/star\|delete\|zip`, `POST /gallery/{id}/reveal`, `POST /pnginfo`, `POST /infotext/parse\|image` |
| models | `GET /models`, `POST /models/refresh\|unload\|reload`, `GET /models/loaded`, `GET/POST /models/hashes` |
| networks (LoRA, embeddings, checkpoints, VAE, text encoders, extension pages) | `GET /networks`, `GET /networks/{kind}`, `GET /networks/{kind}/facets`, `GET/PATCH /networks/{kind}/{name}/meta`, `PUT /networks/{kind}/{name}/preview`, `POST /networks/{kind}/{name}/open-folder`, `POST /networks[/{kind}]/refresh` |
| settings and styles | `GET/PATCH /options`, `POST /options/validate`, `GET /styles`, `PUT/DELETE /styles/{name}`, `POST /styles/apply`, `POST /prompt/tokens` |
| prompt tools | `/prompt-tools/config`, `/complete`, `/usage`, `/lookup`, `/translate`, `/translation-key`, `/groups`, `/refresh`, `/history`, `/collections`, `/favorites`, `/favorites-export`, `/favorites-import` |
| extensions | `GET/PATCH /extensions`, `POST /extensions/check-updates`, `POST /extensions/{name}/check`, `POST /extensions/update\|install`, `DELETE /extensions/{name}`, `GET /extensions/index`, `/extensions/config-states[/{id}[/restore]]` |
| server | `GET /auth/status`, `POST /auth/login\|logout`, `GET /server/status`, `POST /server/reload\|restart`, `GET /sysinfo`, `POST /sysinfo/check`, `GET /licenses`, `GET /user-assets/{name}` |

`/internal/ping` answers `{"instance"}` (an id that changes with every
reload or restart), `/internal/progress` and `/internal/pending-tasks` keep
the A1111 progress protocol (a Remi job's task id is `task(remi-<job id>)`).

Extension management writes (install, update, uninstall, enable) are refused
with 403 when the server is reachable from other machines (`--listen`,
`--server-name`, `--share`, `--ngrok`) unless
`--enable-insecure-extension-access` is given.

## WebSocket `/api/v2/ws`

One connection carries every event; the SPA keeps exactly one.

**Handshake.** Connect (with the cookie or Basic credentials when logins are
on; a browser page from another origin is refused with close code 1008). The
server sends `hello`, then a snapshot: `queue.changed`, and `job.started` +
`job.progress` when a job is running.

**Text frames** are JSON objects with a flat envelope `{"type": ..., <fields>}`:

| type | fields | when |
|---|---|---|
| `hello` | `features` (`events.v1`, `preview.binary`, `preview.jpeg`, `preview.webp`, `jobs.v1`), `version` | on connect |
| `queue.changed` | `size`, `paused`, `running` (job id or null) | the queue changed |
| `job.queued` | `job_id`, `seq`, `tab`, `position` | a job was added |
| `job.started` | `job_id`, `seq`, `tab` | the worker holds the generation lock and starts it |
| `job.progress` | `job_id`, `seq`, `tab`, `step`, `steps`, `eta` (s), `job_no` (0-based), `job_count`, `progress` (0–1), `text` | while running |
| `job.completed` | `job_id`, `seq`, `tab`, `status` (`completed`, `interrupted`, `cancelled`), `outputs`, `infotext`, `warnings` (only when a script step raised and was skipped, or the processing left a note with `p.comment()`, e.g. a feature skipped on this device; also the job's `warnings`) | finished with results (possibly none) |
| `job.failed` | `job_id`, `seq`, `tab`, `status` (`failed`, `cancelled`), `error`, `errors`? | failed, or cancelled before it ran |
| `options.changed` | `keys` | settings changed (by anyone) |
| `networks.changed` | `kind`? | a network list was rescanned |
| `outputs.changed` | `ids` | gallery stars or removals |
| `model.loading` | `stage`, `model` | a checkpoint is being loaded |
| `model.loaded` | `model`, `error`? | loading finished |
| `pong` | – | answer to `ping` |
| `resync` | `dropped` | the client fell behind; a fresh snapshot follows, refetch what you show |

Unknown types must be ignored (new events may be added).

**Client messages:** `{"type": "ping"}` and `{"type": "hello", "features":
[...]}`. After a client `hello` the server only sends previews in the formats
listed (`preview.binary` plus `preview.jpeg` and/or `preview.webp`); a client
that never says hello receives every preview.

**Binary frames** are live previews of the running job:

```
offset  size  field
0       u8    type     = 1 (preview)
1       u8    fmt      = 1 JPEG, 2 WebP
2       u32   job_seq  (little-endian) = the job's "seq"
6       u32   step     (little-endian)
10      …     image bytes
```

A dedicated encoder thread sends each new preview as soon as the sampler
produces it, at most `live_preview_fps` frames per second (setting, default
6), JPEG when the live preview format setting is
JPEG, else WebP, and nothing while no client is connected. `job.progress`
is published when the step counter changes (checked every 0.2 s), so it
reports the step being sampled and arrives shortly after the previous
step's frame; a client that wants one update per step can show it with the
next frame, as the SPA does. Each client has
one preview slot: a newer frame replaces one not yet sent, so slow clients
skip frames instead of lagging. Events have a bounded queue per client
(256); on overflow the client gets `resync`.

Previews are enabled by the usual settings (Settings → Live previews: show
live previews, preview method, every N steps).

## `/sdapi/v1` compatibility

Start with `--api` (WebUI plus API) or `--nowebui` (API only). The A1111 /
Forge routes are kept with their request and response models (`txt2img`,
`img2img`, `extra-single-image`, `extra-batch-images`, `png-info`,
`progress`, `interrupt`, `skip`, `options`, `cmd-flags`, `samplers`,
`schedulers`, `upscalers`, `latent-upscale-modes`, `sd-models`, `sd-modules`,
`face-restorers`, `prompt-styles`, `embeddings`, `refresh-*`, `memory`,
`unload-checkpoint`, `scripts`, `script-info`, `extensions`,
`server-kill|restart|stop` with `--api-server-stop`), plus extension routes
such as `/sdapi/v1/loras`.

What changed underneath, and what it means for clients:

- **`alwayson_scripts` and `script_args` stay positional.** `alwayson_scripts`
  is keyed by the script title (case-insensitive) with `{"args": [...]}`; the
  argument order is the script's `params()` declaration order (or its
  `param_order`), which for every built-in script equals the order its old
  Gradio `ui()` returned. Arguments you omit take the defaults from the
  parameter layout (no `ui()` is re-run). `GET /sdapi/v1/script-info` lists
  every script's arguments with labels, defaults, bounds and choices from
  the same layouts.
- **Infotext** (`"infotext"` in a txt2img/img2img request) is parsed by the
  same code as the SPA's paste (`remi.core.infotext.parse_infotext`), so it
  also restores extension parameters; explicit request fields win.
- `/sdapi/v1` requests run on the generation lock directly; they are not
  stored in the Remi queue or its history, but they wait for (and block) the
  queue's worker, and `/api/v2/interrupt` stops them.
- `/sdapi/v1/options` cannot change settings marked "restricted"; the
  WebUI's own `PATCH /api/v2/options` can.
- The Agent Scheduler (`/agent-scheduler/v1/*`) and Prompt Queue
  (`/prompt-queue/*`) APIs are **not** kept: use `POST /api/v2/jobs` and the
  `/api/v2/queue/*` routes. Their data is imported with
  `POST /api/v2/queue/import/{agent-scheduler|prompt-queue}` (or Queue → ⋯ →
  Import in the UI).

New integrations should prefer `/api/v2`: named values instead of argument
positions, validation with per-field errors, a real queue, and push events
instead of polling.

## Examples

### curl

```bash
BASE=http://127.0.0.1:7860/api/v2

# What the txt2img form takes (core params and script ids)
curl -s $BASE/schema/txt2img | jq '{core: .core.defaults | keys, scripts: [.scripts[] | {id, args}]}'

# Queue a txt2img job: only the values that differ from the defaults
curl -s -X POST $BASE/jobs -H 'Content-Type: application/json' -d '{
  "tab": "txt2img",
  "core": {"prompt": "1girl, watercolor", "width": 768, "height": 768, "cfg_scale": 1},
  "scripts": {"sampler": {"steps": 8, "sampler_name": "Euler", "scheduler": "Simple"},
              "seed": {"seed": 12345}}
}'
# → {"job_id":"84d3…","job_ids":["84d3…"],"seq":17}

# Poll it, then download the first image
curl -s $BASE/jobs/84d3… | jq '{status, outputs}'
curl -s -o out.png "$BASE/files/<output id>?download=1"

# img2img: upload the source first
ID=$(curl -s -F file=@source.png $BASE/uploads | jq -r .id)
curl -s -X POST $BASE/jobs -H 'Content-Type: application/json' \
  -d "{\"tab\":\"img2img\",\"core\":{\"prompt\":\"oil painting\",\"init_img\":\"$ID\",\"denoising_strength\":0.45}}"

# Reuse the parameters of an infotext
curl -s -X POST $BASE/infotext/parse -H 'Content-Type: application/json' \
  -d '{"text":"a cat\nSteps: 8, Sampler: Euler, Schedule type: Simple, CFG scale: 1, Seed: 1, Size: 768x768"}' \
  | jq '{tab, core, scripts}'

# With --auth: log in once and keep the cookie …
curl -s -c jar -X POST $BASE/auth/login -H 'Content-Type: application/json' -d '{"username":"me","password":"secret"}'
curl -s -b jar $BASE/queue
# … or send Basic credentials every time
curl -s -u me:secret $BASE/queue
```

### Python: queue a job and wait for it (requests)

```python
import time
import requests

BASE = "http://127.0.0.1:7860/api/v2"
session = requests.Session()
# session.auth = ("me", "secret")  # with --auth

job = session.post(f"{BASE}/jobs", json={
    "tab": "txt2img",
    "core": {"prompt": "1girl, watercolor", "width": 768, "height": 768, "cfg_scale": 1},
    "scripts": {"sampler": {"steps": 8, "sampler_name": "Euler", "scheduler": "Simple"}, "seed": {"seed": 12345}},
})
if job.status_code == 422:
    raise SystemExit(job.json()["errors"])  # every rejected value, with owner and param
job.raise_for_status()
job_id = job.json()["job_id"]

while (detail := session.get(f"{BASE}/jobs/{job_id}").json())["status"] in ("queued", "running"):
    time.sleep(1)
print(detail["status"], detail.get("error"))
for index, output in enumerate(detail["outputs"]):
    data = session.get(f"{BASE}/files/{output['id']}").content
    open(f"out_{index}.png", "wb").write(data)
```

### Python: follow events and live previews (websockets)

```python
import asyncio
import json
import struct

import requests
import websockets  # pip install websockets

HTTP = "http://127.0.0.1:7860/api/v2"
WS = "ws://127.0.0.1:7860/api/v2/ws"


async def main() -> None:
    async with websockets.connect(WS) as ws:  # with --auth: additional_headers={"Authorization": "Basic …"}
        await ws.send(json.dumps({"type": "hello", "features": ["events.v1", "preview.binary", "preview.jpeg", "preview.webp"]}))
        job = requests.post(f"{HTTP}/jobs", json={"tab": "txt2img", "core": {"prompt": "a lighthouse at dusk"}}).json()
        seq = job["seq"]
        async for message in ws:
            if isinstance(message, bytes):  # live preview
                kind, fmt, job_seq, step = struct.unpack_from("<BBII", message)
                if kind == 1 and job_seq == seq:
                    ext = "jpg" if fmt == 1 else "webp"
                    open(f"preview.{ext}", "wb").write(message[10:])
                continue
            event = json.loads(message)
            if event.get("job_id") != job["job_id"]:
                continue
            if event["type"] == "job.progress":
                print(f"step {event['step']}/{event['steps']}")
            elif event["type"] in ("job.completed", "job.failed"):
                print(event["type"], event["status"], [output["id"] for output in event.get("outputs", [])])
                return


asyncio.run(main())
```

### `/sdapi/v1` (unchanged clients)

```bash
# needs --api (or --nowebui)
curl -s -X POST http://127.0.0.1:7860/sdapi/v1/txt2img -H 'Content-Type: application/json' -d '{
  "prompt": "a lighthouse", "steps": 8, "sampler_name": "Euler", "scheduler": "Simple",
  "cfg_scale": 1, "width": 768, "height": 768, "seed": 12345,
  "alwayson_scripts": {"anima dave diversity": {"args": [true, 0.15, 0.1, "8-18"]}}
}' | jq '.info | fromjson | .infotexts[0]'
```
