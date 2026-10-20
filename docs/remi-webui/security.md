# Security

How the Remi WebUI protects itself and its user: the threat model, the
controls and where they live, how to run the server where others can reach
it, and what remains out of scope. The controls are tested in
`tests/test_remi_security.py`, `tests/test_remi_server_auth.py`,
`tests/test_extension_route_guard.py`, `tests/test_remi_server_files.py`,
`tests/test_remi_server_runtime.py` (startup makes no outbound
connection), the ADetailer suite (detector downloads) and
`webui/src/ui/{rich-text,image-ref}.test.*`.

Contents: [Threat model](#threat-model) · [Controls](#controls) ·
[Running exposed](#running-exposed) · [Residual risks](#residual-risks) ·
[Audit record](#audit-record-2026-10)

## Threat model

**Trust model.** The WebUI is single-tenant. Every logged-in user is an
administrator: they can generate, change settings (including output
folders), manage the queue and, where extension access is allowed, install
extensions — which is running arbitrary code. There are no roles. Without
`--auth` the server trusts everyone who can reach it, as the Gradio WebUI
did.

**What is protected:** the machine (no code execution, file reads or writes
outside what the UI offers), the user's files (outputs, uploads, models), the
login secrets, and the user's browser session.

| adversary | can | defended by |
|---|---|---|
| a web page the user visits | make the user's browser send requests to the WebUI (with its cookie), re-point a domain at `127.0.0.1` (DNS rebinding), embed the UI in a frame | cross-site write and WebSocket checks (Origin / fetch metadata), cross-site read checks, rebound-`Host` refusal, `SameSite=Lax` cookies, CSP `frame-ancestors` |
| a peer on the network (exposed server) | send any request, guess passwords | `--auth` login on everything, throttled failures, `--api-auth` for `/sdapi`, extension access off by default when exposed |
| untrusted content | images (uploads, URLs, PNG text chunks), infotext, network metadata, extension index JSON, booru API responses, extension HTML/Markdown | content sniffing, size/pixel caps, opaque file ids, no `innerHTML` in the SPA, URL scheme allow-lists, CSP, no pickle loading of uploads |
| an API client | everything a user can, through `/api/v2` and `/sdapi/v1` | the same logins; restricted settings stay out of `/sdapi`; URL fetching limited to public addresses |

**Out of scope:** extensions the administrator installed (they run with the
WebUI's privileges), other OS users with access to the data folder, and
attackers who already have a login (they are administrators by design).

## Controls

### Authentication (`remi/server/auth.py`)

- `--auth user:pass[,…]` / `--auth-file`: every request to `/api/v2` (REST
  and WebSocket), `/internal`, files and thumbnails, the SPA shell and the API
  docs (`/docs`, `/docs/*`, `/redoc`, `/openapi.json`) needs the session
  cookie or HTTP Basic credentials. Public: `/api/v2/auth/*`,
  `/internal/ping`, `/favicon.svg`, `/assets/*` (the SPA's build files,
  public so the shell can load and show its sign-in form: only existing
  build files; anything else there is a 404, never the SPA shell).
- Signing in: the server renders no login page. A browser page load
  without a login (the SPA's routes and the API docs) gets the SPA shell
  with status 401 and `Cache-Control: no-store` (no data: the built
  `index.html` plus the root path and `--theme`); the SPA asks
  `GET /api/v2/auth/status` and shows its own form (`webui/src/features/auth`),
  translated from its catalogs, which posts to `/api/v2/auth/login`. A 401
  while the app is open shows the form over it. Without a built SPA (or
  under `--nowebui`) the 401 offers HTTP Basic (`WWW-Authenticate`).
- Session cookie `remi_session`: `user.expiry.HMAC` signed with a
  per-installation secret (`secret.key`, mode 0600) and the user's password;
  30 days; `HttpOnly`, `SameSite=Lax`, `Secure` with `--tls-*` or an HTTPS
  request; scoped to `--subpath`. Logout revokes it server-side (denylist
  until expiry); a password change invalidates all sessions.
- Brute force: ten failed logins per client per minute — through the login
  form **or HTTP Basic** — and the client gets 429 for a minute, even with the
  right password. Password comparisons are constant-time; unknown users cost
  the same as wrong passwords; credentials over 1024 characters are refused
  unchecked.
- `/sdapi/v1` (core and extension routes under it): `--api-auth` Basic
  credentials when set (throttled the same way); without `--api-auth`, the
  WebUI login (cookie or Basic) on any bind — a loopback-only server is
  still reached through a reverse proxy or tunnel on the same machine. The
  explicit opt-out `--api-no-auth-on-localhost` restores the A1111
  behaviour for local tools: a loopback peer that came through no proxy (no
  `X-Forwarded-For`/`-Host`, `X-Real-IP` or `Forwarded` header) needs no
  login for `/sdapi` and extension API routes; it never applies with
  `--api-auth`, nor to `/api/v2`, files, pages or the docs.
- Docs without `--auth` but with `--api-auth` need the API credentials.
- Extension routes outside these prefixes (`/controlnet/*`, `/adetailer/*`):
  `modules/extension_route_guard.py` — cross-site checks always; a login
  whenever `--auth` / `--auth-file` is set, on any bind (session, Basic, or
  `--api-auth` credentials; local clients exempt only with
  `--api-no-auth-on-localhost`); under `--nowebui`, `--api-auth` whenever it
  is set.

### Browser-based attacks (`modules/extension_route_guard.py`, `remi/server/auth.py`)

- **CSRF / cross-site WebSocket:** `POST`/`PUT`/`PATCH`/`DELETE` and
  WebSocket handshakes whose `Origin` is not this site (scheme, host and
  port) or an allowed `--cors-allow-origins(-regex)` origin are refused (403 /
  close 1008); `Origin: null` always is; without an `Origin`,
  `Sec-Fetch-Site: cross-site` is refused. API clients (no `Origin`) pass.
- **Cross-site reads:** a `GET`/`HEAD` that fetch metadata marks as a
  cross-site or same-site *subresource* request (`<img>`, `fetch`, frames) is
  refused unless its origin is allowed; top-level navigations (links,
  bookmarks) pass. This stops blind requests such as
  `<img src="http://127.0.0.1:7860/api/v2/extensions/index?url=…">`.
- **DNS rebinding:** a loopback-only server (default bind, loopback
  `--server-name`, no `--ngrok`) refuses a `Host` that is not a loopback name
  or an allowed CORS host (403 "Unknown host refused").
- **Headers:** every response carries `X-Content-Type-Options: nosniff` and
  `Referrer-Policy: same-origin`. The SPA shell and the API docs pages carry a CSP:
  scripts only from the server plus the page's own inline scripts by SHA-256
  hash (no `unsafe-inline`/`unsafe-eval` scripts), no plugins, `base-uri` /
  `form-action` `'self'`, `frame-ancestors 'self'` plus the literal
  `--cors-allow-origins` (clickjacking). `connect-src` is `'self'` (the
  API and its WebSocket; CSP Level 3 browsers, which the SPA's ES2023 build
  requires anyway, match `ws:`/`wss:` to the page's own origin) plus
  `https://raw.githubusercontent.com` for the license texts of Settings →
  Licenses, so an injected script could not send data elsewhere. `img-src`
  is `'self' data: blob:` plus the hosts the user lists in Settings → System
  → "Remote image hosts" (`remi_image_hosts`; only `http(s)://host[:port]`
  and `https://*.host` tokens are taken, so the setting cannot add other
  sources or directives); remote images in extension help or outputs are
  blocked otherwise, so injected markup cannot load tracking pixels.

### Files and uploads (`remi/files.py`, `remi/server/routes/files.py`)

- Files are served only by opaque id (`/api/v2/files/{id}`,
  `/api/v2/thumbs/{id}`; ids match `^[A-Za-z0-9_-]{8,64}$`); the browser never
  sends or receives a filesystem path. Non-media files are sent as
  attachments (`application/octet-stream`); SVG is never accepted.
- Uploads: 64 MB by default (`remi_upload_max_mb`), enforced while streaming
  (also for chunked bodies), one file per form; the type comes from the
  content (PNG, JPEG, GIF, WebP, BMP, TIFF, JXL, AVIF/HEIC, MP4/MOV/WebM/AVI,
  safetensors headers, UTF-8 text by extension), never from the name or
  declared type. Images are capped at 179 MP and 2000 frames from the header
  before any decode, independently of `Image.MAX_IMAGE_PIXELS` (which the
  X/Y/Z plot lifts). Stored as `uploads/<sha256><ext>`; the display name is
  cleaned (last path component, no control characters, ≤ 200 characters).
- `/sdapi` images (base64 or URL) get the same pixel cap before decoding.
- Upload errors are fixed messages ("not a readable image"): Pillow's own
  message names the incoming file in the data folder, so it is only logged.
- Every other request body (JSON jobs, `/sdapi` base64 images, extension
  routes) is capped at twice `remi_upload_max_mb` (at least 16 MB),
  enforced while it streams in, chunked bodies included
  (`remi/server/limits.py`): 413 past it, without buffering more.
- Unsafe deserialization: uploads can only be images, video, text or
  safetensors (a JSON header); nothing reachable from an upload or URL calls
  `pickle`/`torch.load`. Model files are only loaded from the model folders.

### Paths and local actions

- Static SPA files are resolved inside `webui/dist` only; locale, preset,
  page, network and config-state names are looked up in known lists, never
  joined into paths. Network metadata and previews are written next to an
  indexed model file only.
- Opening folders (`/outputs/open-folder`, `/networks/…/open-folder`,
  `/gallery/{id}/reveal`) is refused with `--hide-ui-dir-config` and for any
  client that is not a loopback peer without proxy headers. On macOS a folder
  that is a bundle (`Some.app`) is revealed in Finder, never launched.
- Path settings are refused with `--hide-ui-dir-config` (`restricted_opts`),
  and `--freeze-settings*` freeze settings for `/api/v2` and `/sdapi` alike.

### Extensions (`remi/extensions_admin.py`)

- Installing, updating, uninstalling, restoring and the index download are
  refused (403) when the server is reachable from other machines, unless
  `--enable-insecure-extension-access`.
- Install input: URL `https://`, `http://`, `ssh://`, `git://` or
  `user@host:path`, no control characters, a host or user that cannot be read
  as an option (`ssh://-oProxyCommand=…`); branch names without a leading `-`
  or `/`, `..`, `//` or a `.lock`/`/`/`.` ending; folder names
  `[A-Za-z0-9][A-Za-z0-9._-]{0,99}`, no Windows device names. Git runs without
  a shell, with `--` before the URL, no terminal or credential prompts, a
  timeout, and `GIT_ALLOW_PROTOCOL=https:http:ssh:git` for the clone and its
  submodules (no `file://` or `ext::` helpers). Uninstall only deletes a
  direct child of `extensions/`.
- The index is fetched only over http(s), at most 16 MB, entries whose URL
  fails the same validation are dropped. An index URL the client names
  (other than the default or `WEBUI_EXTENSIONS_INDEX`) must resolve to public
  internet addresses only, and so must every redirect hop (422 otherwise),
  so the field cannot be used to probe the server's own network.

### Outbound network calls

Nothing contacts the network unless the user configured or asked for it:

| call | when |
|---|---|
| extension index (`raw.githubusercontent.com` or `WEBUI_EXTENSIONS_INDEX`) | opening Extensions → Available; another index URL typed there must be a public address (each redirect hop checked) |
| `git fetch` / `clone` | checking for updates, installing, updating |
| prompt translation | only with a network provider and its URL/key set (`prompt_tools_translation_*`); the key is write-only through the API |
| prompt writing ("Write a prompt", `POST /api/v2/prompt-tools/generate`) | only when the user sends a description and the translation provider is the OpenAI-compatible one with its URL and model set; the description (and the current prompt, when changing it) goes to that provider's `/chat/completions` with the same write-only key |
| booru grabber (Prompt Workshop) | on a grab; https only, the booru's own image hosts, public addresses checked per redirect hop, 64 MB cap |
| model downloads (upscalers, face restorers, preprocessors, ADetailer, TAESD) | the first time the user runs a feature whose model is missing; fixed upstream URLs. ADetailer lists its 10 stock detectors from the start and downloads one when a unit first runs with it (never at startup or on Refresh; `--ad-no-huggingface`: never). A test starts the whole WebUI with an empty data folder and fails on any connection to a non-local host |
| `/sdapi` image URLs | an API request names one; off with `api_enable_requests`; with `api_forbid_local_requests` (default) every address of every redirect hop must be public; 64 MB cap. Both settings are `restrict_api`: `/sdapi/v1/options` cannot change them |
| license texts | opening an entry in Settings → Licenses (fetched by the browser from `raw.githubusercontent.com`, the only outside host the CSP lets the page connect to, without cookies or referrer) |
| remote images | only from hosts listed in "Remote image hosts" (`remi_image_hosts`, empty by default) |

The API docs (`/docs` Swagger UI, `/redoc` ReDoc, `remi/server/api_docs.py`)
are not FastAPI's default pages, which load floating CDN versions
(`cdn.jsdelivr.net/npm/swagger-ui-dist@5`, `redoc@2`), Google Fonts and a
`fastapi.tiangolo.com` favicon into a page on the app's origin. The SPA build
vendors pinned copies (`swagger-ui-dist` and `redoc` in `webui/package.json`,
`apiDocs` in `webui/vite.config.ts`, which writes them under
`assets/<package>-<version>/` with their licenses and fails when a viewer
still names a CDN host; ReDoc's attribution logo from `cdn.redoc.ly` is
replaced). The pages carry the same CSP as the SPA shell (scripts from the
server and the page's inline script by hash, `connect-src 'self'`), Swagger
UI's validator badge (`validator.swagger.io`) is off, and the packages'
install scripts are disabled (`webui/pnpm-workspace.yaml`: `@scarf/scarf`
would report the install). `tests/test_remi_api_docs.py` checks the pages;
a browser load of both against a built server makes no outside request and
reports no CSP violation.

### XSS (`webui/src/ui/rich-text.tsx`, `webui/src/ui/image-ref.ts`)

- The SPA never uses `innerHTML`/`dangerouslySetInnerHTML`; React escapes all
  text (infotext, file names, network metadata, extension strings).
- Markdown nodes are rendered by a small parser into React elements; raw
  HTML stays text. HTML nodes/outputs are parsed inert (DOMParser) and
  rebuilt from an allow-list of tags and attributes; scripts, styles,
  frames, forms, SVG and event handlers are dropped.
- Link and image URLs pass `safeUrl` (http, https, mailto or relative, read
  with the browser's URL parser, so `java\tscript:` and entity tricks fail);
  output images pass `safeImageUrl` (http(s), blob, `data:image/*`, same-site
  paths). The CSP is the second line of defence.

### Error responses (`remi/server/app.py`)

An unhandled exception answers `500 {"error": "Internal Server Error",
"detail": "Internal server error (id …); …", "id": "…"}`: no exception text,
no request or response body (a `ResponseValidationError` carries the whole
response), no paths. The exception and traceback are logged with the same
id. Exceptions that carry a 4xx `status_code` keep their `detail`, which is
meant for the client.

A request that fails its model is a 422, logged at debug level without a
traceback, so no client (the public login form included) can flood the log.
That holds for bodies with `1e999` or `NaN` too: the error responses encode
non-finite numbers as text, and `/api/v2` models refuse them as values.
`POST /sdapi/v1/options` with a setting that does not exist is a 422 that
sets nothing (it was a `KeyError`, a 500).

### Secrets in logs

`modules/redact.py`: the launch line, `sysinfo` (command line,
`COMMANDLINE_ARGS`, settings with key/token/secret/password-like names,
extension remotes) and `/sdapi/v1/cmd-flags` hide `--auth`, `--api-auth`,
`--ngrok`, `--ngrok-options` values (also as `--flag=value` and argparse
abbreviations) and credentials inside URLs. Logins are never logged; the
translation API key is stored write-only.

## Running exposed

The default (no `--listen`) only accepts connections from this machine.
To let other machines in:

1. **Always set a login:** `--auth user:password` or `--auth-file logins.txt`
   (one or more `user:pass` per line). Use long random passwords; the server
   prints a warning at startup when it is reachable without one.
2. **Use TLS:** `--tls-keyfile key.pem --tls-certfile cert.pem`, or put the
   WebUI behind an HTTPS reverse proxy (bind it to loopback and let the proxy
   forward; the session cookie becomes `Secure` when the proxy reports
   HTTPS). Without TLS, passwords and cookies cross the network in clear text.
3. **API:** `--api` exposes `/sdapi/v1`; set `--api-auth` for it (otherwise
   it uses the WebUI login). Scripts can send Basic credentials. Do not use
   `--api-no-auth-on-localhost` when other people can run programs on the
   machine or when a proxy that adds no forwarding headers fronts it (the
   server prints a startup warning while it is on).
4. **Keep extension access off** (the default when exposed). Only add
   `--enable-insecure-extension-access` if every user may run code on the
   machine.
5. **Hide paths** with `--hide-ui-dir-config` when users should not change
   output folders or open folders on the server, and consider
   `--freeze-settings` / `--freeze-settings-in-sections`.
6. **CORS:** list exact origins in `--cors-allow-origins` only for tools you
   trust: allowed origins may send requests with the user's cookie and read
   the answers. Never use `*` on an exposed server.
7. **Reverse proxies:** forward `Host` (or `X-Forwarded-Host`) and the
   WebSocket upgrade for `/api/v2/ws`; with a path prefix use `--subpath`.
   Login throttling counts per peer address, so behind a proxy all clients
   share one counter (a failed-login flood locks everyone out for a minute,
   which errs on the safe side).
8. `--ngrok` tunnels count as exposed (HTTPS is provided by ngrok).

## Residual risks

- **Exposed without `--auth`:** anyone who can reach the server controls it,
  and a site that a user on the same network visits can reach it through DNS
  rebinding (the `Host` check only protects loopback-only servers). Use a
  login.
- **Same-machine tunnels without forwarding headers** (`ssh -L`, plain TCP
  forwarders) look like local clients to the folder actions and, with
  `--api-no-auth-on-localhost`, to the API; use `--hide-ui-dir-config` and
  leave that flag off when sharing that way.
- **DNS answers that change between the address check and the connection**
  are not caught for a client-named extension index URL either (it is only
  reachable where extension access is allowed).
- **DNS answers that change between the address check and the connection**
  are not caught for `/sdapi` image URLs; turn `api_enable_requests` off on
  servers that must not reach their network.
- **Request bodies** are buffered up to their cap (twice
  `remi_upload_max_mb`): each concurrent request of a client allowed to call
  the API can hold that much memory while it is parsed.
- **Model downloads** come from fixed upstream URLs and trust those hosts and
  TLS; downloaded `.pth`/`.pt` files are loaded by their libraries
  (`torch.load` defaults to `weights_only=True` in the bundled PyTorch).
- **Installed extensions** are trusted code: their routes are guarded, but
  what they do inside is theirs.

## Audit record (2026-10)

Findings of the wave-5 audit and their fixes (severity for an exposed
server with `--auth`, or a local server visited by a hostile page).

| # | finding | severity | fix |
|---|---|---|---|
| 1 | `/sdapi/v1/options` called `set_config` without `is_api`, so API clients could turn off `api_forbid_local_requests` / `api_enable_requests` (then SSRF) | high | `set_config(req, is_api=True)` |
| 2 | `/sdapi` image URLs: redirects followed blindly after checking only the first host (SSRF to internal hosts), IPv4-only resolution, no size limit | high | manual redirects with a check per hop, `getaddrinfo` (IPv4+IPv6, mapped addresses), 64 MB cap |
| 3 | `/sdapi` and extension `/sdapi/*` routes were open on an exposed server with `--auth` but no `--api-auth`; extension `/sdapi/*` routes ignored `--api-auth` on loopback | high | `sdapi_refusal`: `--api-auth`, else the WebUI login when exposed |
| 4 | HTTP Basic credentials were checked on every request without throttling (unlimited password guessing) | medium | per-client throttle shared with the login form; `--api-auth` throttled too |
| 5 | `GET /sdapi/v1/cmd-flags`, the launch log line and `sysinfo` (`--auth=…` form, `--ngrok` token, `COMMANDLINE_ARGS`, URL credentials) leaked secrets | medium | `modules/redact.py` |
| 6 | Cross-site `GET`s (e.g. `<img src=…/extensions/index?url=http://intranet/>`) ran with no check: blind SSRF/CSRF through side-effecting reads | medium | fetch-metadata check for subresource reads |
| 7 | `safeUrl` let `java\tscript:` (and entity-encoded tabs in HTML nodes) through; output image `url`s were used as links unchecked | medium | URL-parser-based allow-list; `safeImageUrl` |
| 8 | Upload decompression-bomb protection depended on `Image.MAX_IMAGE_PIXELS`, which the X/Y/Z plot sets to `None` for the whole process | medium | explicit 179 MP / 2000-frame cap from the header; same for `/sdapi` images |
| 9 | No CSP or framing protection (clickjacking of the extension installer, no XSS backstop) | medium | CSP with inline-script hashes, `frame-ancestors`; `nosniff` and `Referrer-Policy` everywhere |
| 10 | Open-folder endpoints could be triggered by remote clients; on macOS a "folder" that is an `.app` bundle was launched | low | loopback-only like reveal; bundles revealed |
| 11 | Extension install: `ssh://-o…` style hosts and `file`/`ext` transports in submodules were not excluded; `..` branches and Windows device names accepted | low | `valid_git_url` / `valid_branch` / `valid_dirname`, `GIT_ALLOW_PROTOCOL` |
| 12 | `/docs/oauth2-redirect` was outside the docs login; docs open under `--api-auth` without `--auth` | low | docs prefix check, `--api-auth` for docs |
| 13 | Unbounded login field sizes; the login page interpolated `--subpath` into a script unescaped | low | 1024-character cap; JSON-escaped root (the server-side login page was later removed: the SPA signs in) |
| 14 | No warning when the server is exposed without a login, with `--cors-allow-origins=*`, insecure extension access or without TLS | low | startup warnings (`remi.server.launch.exposure_warnings`) |
| 15 | With `--auth` and no `--api-auth`, `/sdapi` and extension API routes (`/controlnet/*`, `/adetailer/*`) were open on a loopback-only server, which a reverse proxy on the same machine makes reachable from anywhere | high | the WebUI login on any bind; explicit `--api-no-auth-on-localhost` opt-out for local tools |
| 16 | The error handler returned the exception text and its `body` (`GET /sdapi/v1/cmd-flags` echoed the whole flag set, with paths, in a `ResponseValidationError` because `--port` is an int with a `None` default) | medium | generic 500 with a correlation id, details only in the log; `FlagsModel` types from the parser's `type=`; the unused A1111 `api_middleware` (same leak, one call away from being re-enabled) removed |
| 17 | CSP allowed `connect-src https: ws: wss:` and `img-src *`: an XSS could exfiltrate to any host, markup could load tracking pixels | medium | `connect-src 'self'` + the license host; `img-src 'self' data: blob:` + an opt-in host list |
| 18 | `GET /api/v2/extensions/index?url=` fetched any client-named URL, including loopback and private addresses (port/service probe through the error text) | low | client-named index URLs and their redirects must be public addresses |
| 19 | ADetailer downloaded its 10 detector models (about 240 MB from Hugging Face, GitHub and Google Storage) at startup, with no user action | medium | listed at startup, downloaded when a unit first runs with one; the startup test runs without `--ad-no-huggingface` |
| 20 | JSON bodies (jobs, `/sdapi` base64 images) had no size cap | medium | twice the upload limit, enforced while streaming (`remi/server/limits.py`) |
| 21 | `{"username": 1e999}` to the public login (and any body route) answered 500 with a logged traceback per request: the validation error could not be encoded | low | 422 through a non-finite-safe handler, debug log; `/api/v2` models refuse non-finite numbers |
| 22 | `/assets/` and `/assets/<missing>` served the SPA shell without a login | low | 404 for anything under `/assets` that is not a build file |
| 23 | An unreadable upload's 415 named the incoming file's path in the data folder (also the network preview, PNG Info and reveal errors could carry OS error text) | low | fixed messages; the details are logged |
| 24 | `/docs` and `/redoc` were FastAPI's defaults: Swagger UI / ReDoc scripts from `cdn.jsdelivr.net` (floating `@5` / `@2`, no SRI), Google Fonts and a `fastapi.tiangolo.com` favicon on every visit, third-party code on the app origin with the session, no CSP | medium | own pages with the build's pinned copies and the shell's CSP (`remi/server/api_docs.py`) |

Checked and found sound: session signing and revocation, cookie flags,
login timing, Origin/rebinding checks for writes and WebSockets, file serving
by id, upload streaming limits and content sniffing, SPA static path
resolution, network write paths, config-state ids, the booru grabber's URL
checks, the translation key handling, and the absence of `innerHTML` in the
SPA.
