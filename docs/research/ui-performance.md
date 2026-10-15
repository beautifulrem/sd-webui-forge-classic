# Remi WebUI — UI performance

Measured 2026-10-15 against the scorecard's performance budgets
(`docs/remi-webui/scorecard.md`, dimension 5). The numbers come from the real
server with every built-in extension loaded and the real Anima Turbo model,
driven by Chromium through Playwright and the DevTools protocol. The tools
are in `tools/perf/` and `webui/tools/perf/`; the raw JSON results are listed
under each section and kept in `/tmp/remi-perf-out` on the measuring machine.

## Result

| budget | measured (final) | result |
|---|---|---|
| Initial JS ≤ 300 KB gzip | **179.8 KB** gzip (4 files; 156.3 KB brotli, as served) | pass |
| No route chunk > 150 KB gzip, except justified lazy canvas / editor chunks | largest page chunk 28.2 KB (`generate`); largest lazy chunks: canvas editor 66.0 KB, CodeMirror 62.4 KB; largest initial chunk 86.3 KB | pass |
| Generate view ≤ 3,000 DOM nodes, all built-in extensions loaded | **1,174** elements (1,489 DOM nodes incl. text, CDP `Nodes`) on txt2img, 1,245 / 1,602 on img2img; heaviest single section opened: 1,529 | pass |
| INP p95 ≤ 50 ms typing a 300-token prompt, autocomplete on | **p95 24 ms**, max 40 ms (INP 24 ms); at 4× CPU throttling p95 24 ms, max 56 ms | pass |
| 5,000 network cards scroll at 60 fps | **60.0 fps**, p95 frame 16.7 ms, 0 of 3,257 frames over 20 ms (2,400 px/s, whole list) | pass |
| 10,000 gallery items scroll at 60 fps | **60.0 fps**, p95 frame 16.8 ms, 0 of 5,699 frames over 20 ms (2,400 px/s, all 10,000 loaded) | pass |
| Preview frame latency p95 ≤ 150 ms | **p95 20.7 ms** (sampler → painted); was 177 ms before this work | pass (fixed) |
| UI overhead on generation speed ≤ 1 % it/s | **+0.48 %** at the baseline 768×768 (95 % CI ±1.66), **−0.27 %** at 512×512 (±2.53); pooled −0.1 % ± 1.5 % over 27 interleaved window pairs; was 4.5 % before this work | pass (point estimates; the 95 % interval reaches +1.4 %) |
| Server startup no slower than before the rewrite | UI mode **4.37 s** vs 12.02 s (Gradio); API only 4.19 s vs 6.58 s (medians of 5) | pass |

Four things missed their budget or were found wasteful and were changed (see
[Optimisations](#optimisations)): the preview encoder polled, so frames waited
up to 1/6 s before leaving; endless CSS animations on the running-job
indicators and two progress-bar transitions kept the browser redrawing
during generation; SPA assets were
gzipped on every request; and the initial vendor chunk was just over 150 KB.

## Environment

| | |
|---|---|
| Machine | MacBook Air M4 (`Mac16,13`, fanless), 10 cores (4 P + 6 E), 24 GB unified memory |
| OS | macOS 27.0.1 |
| Load | a normal desktop session ran alongside (mail, chat, other browsers, other agent sessions): 22 GB of 24 GB in use and about 7 GB of swap during the runs |
| Server | `/tmp/remi-nogradio-env` (Python 3.13.13, torch 2.14.1 on MPS, no Gradio), `launch.py --api` on port 7861 through `tools/perf/serve_traced.py`, every built-in extension enabled (`config.json` of the repo, `disabled_extensions` empty) |
| Model | `anima-turbo-v1.1.safetensors` + `qwen_image_vae` + `qwen_3_06b_base`; Euler / Simple, CFG 1, Shift 3.5, seed 12345; live previews on (RGB method, every step, JPEG, 6 fps cap) |
| Browser | Chromium 156.0.8078.4 (Playwright 1.64.0, new headless), 1920×1080, DPR 1 |
| Tooling | Node 26.10.0, pnpm 12.6.0, Vite 8 production build (`webui/dist`) |
| Old tree | git ref `remi` (`d5a71cad`) in the worktree `/tmp/remi-baseline`, run with `/tmp/remi-testenv` (same Python and torch, Gradio 4.40.0) |

The machine is fanless and was shared: sampling speed drifted from 1.6 s to
3.3 s per step at 512×512, and from 3.6 s to 7.9 s per step at 768×768, over
the session. Everything that compares speed below interleaves its conditions
for that reason.

## Method and tools

| tool | what it does |
|---|---|
| `tools/perf/fixtures.py create /tmp/remi-perf` | 5,000 LoRA files (valid metadata-only safetensors in 50 folders, each with a 96×128 PNG preview) and a `--data-dir` with 10,000 384×384 PNG outputs carrying an infotext; `remove` deletes everything (done after the runs) |
| `tools/perf/fixtures.py prompt …/prompt300.txt` | the typing test's prompt: 103 common Danbooru general tags, exactly 300 CLIP tokens, 1,203 characters |
| `tools/perf/serve_traced.py` | runs `launch.py` with probes added at import time (no server change): sampler step times, preview image assignment, preview frame publication, to a JSONL file (`time.time()`); a mute file makes the WebSocket hub act as if no client were connected |
| `webui/tools/perf/bundle.mjs` | initial JS (scripts in `index.html` and their static imports) and every lazy entry, from the Vite manifest, gzip -6 and brotli -11 |
| `webui/tools/perf/browser.mjs` | Playwright + CDP scenarios: `load`, `inp`, `networks`, `gallery`, `preview`, `overhead`, `interleaved` (below) |
| `tools/perf/startup.py` | starts old and new trees five times each per mode (after an unrecorded warm-up start) and times them |

Commands (server: `REMI_PERF_TRACE=/tmp/remi-perf-trace.jsonl python
tools/perf/serve_traced.py --skip-prepare-environment --skip-install --api
--port 7861 --data-dir /tmp/remi-perf/data --model-ref $PWD/models
--lora-dirs /tmp/remi-perf/Lora`):

```sh
python tools/perf/fixtures.py create /tmp/remi-perf
python tools/perf/fixtures.py prompt /tmp/remi-perf-out/prompt300.txt
cd webui
pnpm exec vite build --manifest --outDir /tmp/remi-perf-dist --emptyOutDir && node tools/perf/bundle.mjs /tmp/remi-perf-dist
node tools/perf/browser.mjs --only load,inp,networks,gallery
node tools/perf/browser.mjs --only inp --cpu-throttle 4
node tools/perf/browser.mjs --only preview --steps 16 --jobs 4
node tools/perf/browser.mjs --only interleaved --steps 100 --size 768x768 --window 30
node tools/perf/browser.mjs --only interleaved --steps 150 --size 512x512 --window 15
cd .. && python tools/perf/startup.py --old /tmp/remi-baseline --old-python /tmp/remi-testenv/bin/python --new-python /tmp/remi-nogradio-env/bin/python --runs 5
python tools/perf/fixtures.py remove /tmp/remi-perf
```

## 1. Bundle

`bundle.mjs` on a `--manifest` build of the final sources
(`bundle-final.json`):

| | gzip | brotli | raw |
|---|---|---|---|
| Initial JS (`rolldown-runtime`, `react`, `vendor`, `shell`) | **179.8 KB** | 156.3 KB | 553.2 KB |
| — `vendor` (libraries except React) | 86.3 KB | | |
| — `react` (react, react-dom, scheduler) | 66.0 KB | | |
| — `shell` (app shell, router, i18n, WebSocket) | 27.1 KB | | |
| Initial CSS | 31.6 KB | | |
| All JS (226 files) | 918.5 KB | | |

Before the vendor split the same JS was three files with a 152.5 KB `vendor`
chunk (total 180.0 KB): over the 150 KB chunk limit, though not a route
chunk. React is now its own initial chunk.

Page chunks (own chunk / everything a first visit adds on top of the initial
JS, gzip KB): generate 28.2 / 226.3, queue 19.8 / 104.6, gallery 18.1 /
95.6, settings 13.1 / 164.4, extensions 10.7 / 69.5, png-info 7.3 / 29.9,
extras 6.3 / 153.0, extension pages 1.5 / 122.7, networks 1.2 / 147.3.
The added amounts are shared chunks (form renderer 25.3 KB, Base UI parts,
CodeMirror) that later pages reuse.

Largest lazy chunks: the canvas editor (`Editor`, Konva and the editor UI)
66.0 KB, never loaded by txt2img; img2img preloads it after its first render
(`preloadImageEditor()`) so the sketch / inpaint editor opens at once; CodeMirror's view and state (`dist-*`) 62.4 KB, loaded
by the prompt editor; the networks browser (`TargetSwitch`) 32.8 KB; zh-CN
chrome strings 22.4 KB (only for that locale); the command palette 18.4 KB
(on first open). The two over 60 KB are the editor surfaces the budget
exempts; both are already separate from every route.

What the Generate view really loads (`load` scenario, `browser-load-inp-networks-gallery.json`):
138 scripts, 548 KB gzip-equivalent (1,547 KB raw), **488 KB on the wire**
(brotli, precompressed), plus 8 API calls, 80 KB on the wire (568 KB of JSON;
already gzipped by the server's middleware). Before this work the scripts
were gzipped per request at level 9 (547 KB on the wire and 37 ms of
compression on the server's event loop per page load). First contentful
paint 100 ms; img2img: 147 scripts, 548 KB on the wire.

## 2. Generate view DOM

`load` scenario, after the view is idle, garbage collected:

| | elements | DOM nodes (CDP `Nodes`) | JS heap |
|---|---|---|---|
| txt2img, default state | **1,174** | 1,489 | 15.9 MB |
| img2img, default state | 1,245 | 1,602 | 16.5 MB |
| txt2img, heaviest single section opened (Anima Regional Conditioning) | 1,529 | | |
| txt2img, every collapsible section opened at once (73 toggles, nested ones included) | 6,724 | | |

All 33 scripts of the txt2img schema (core, `scripts/`, every built-in extension) are on the page. Collapsed sections do not render
their controls, so the default view stays at 1,174 elements; the next
heaviest sections are FreeFuse (+277), Detail Daemon (+225), ControlNet
(+215) and Spectrum (+182). Opening every section at once is not a state the
UI produces by itself; it is listed as the upper bound.

## 3. INP while typing

`inp` scenario: the 300-token prompt typed into the empty txt2img prompt
editor, one key every 100 ms, about 10 characters per second (at 50 ms per
key the autocomplete debounce let almost no query through). Tag autocomplete is on
(`tac_active`, `danbooru.csv`); the run opened the completion popup 103
times and sent 858 completion requests. INP is computed from Event Timing
(`PerformanceObserver` type `event`, `durationThreshold: 16`): one value per
`interactionId` (the longest of its events), keys whose events were all under
16 ms count as 16 ms; web-vitals INP is the worst interaction after
skipping one per 50.

| | interactions | > 16 ms | p50 | p95 | p99 | max | INP | long animation frames |
|---|---|---|---|---|---|---|---|---|
| unthrottled | 1,203 | 262 | ≤ 16 | **24** | 24 | 40 | 24 | 0 |
| 4× CPU throttling | 1,203 | 1,196 | 16 | **24** | 24 | 56 | 24 | 1 (79 ms) |

(Event Timing durations are rounded to 8 ms.) The typed text matched the
prompt exactly in both runs. A first run before the optimisations gave the
same p95 (24 ms, max 32 ms).

## 4. Scrolling 5,000 network cards and 10,000 gallery items

`networks` and `gallery` scenarios. The grid's scroll container is moved by a
fixed distance in a `requestAnimationFrame` loop from the top to the end
(40 px per frame = 2,400 px/s, and a fling-like 160 px per frame), which
records every frame interval; a CDP trace of each run counts the
compositor's frames (`PipelineReporter` states). The gallery is paged by the
SPA (100 per page); the scenario first scrolls to the end until all pages are
loaded (16.6 s; the gallery then showed 10,037 images: the fixtures plus the
jobs of this session). A second pass at 40 px repeats the run with every
thumbnail cached ("warm").

| list | speed | frames | duration | fps | frame p95 | frames > 20 ms | compositor | DOM after |
|---|---|---|---|---|---|---|---|---|
| networks, 5,000 LoRA cards | 2,400 px/s | 3,258 | 54.3 s (whole list) | **60.0** | 16.7 ms | 0 | 3,258 presented | 995 |
| | 9,600 px/s | 815 | 13.6 s | 60.0 | 16.7 ms | 0 | 813 presented, 811 partial, 2 dropped | 2,367 |
| | 2,400 px/s, warm | 3,258 | 54.3 s | 60.0 | 16.8 ms | 0 | 3,256 presented, 3,254 partial, 2 dropped | 2,051 |
| gallery, 10,037 items | 2,400 px/s | 5,700 | 95.0 s (to 228,000 px) | **60.0** | 16.8 ms | 0 | 5,700 presented | 1,055 |
| | 9,600 px/s | 1,425 | 23.7 s (whole list) | 60.0 | 16.7 ms | 0 | 1,425 presented | 1,128 |
| | 2,400 px/s, warm | 5,700 | 95.0 s | 60.0 | 16.7 ms | 0 | 5,700 presented | 1,055 |

Idle rAF rate was 60 fps on both pages. The compositor column counts
`PipelineReporter` reports: in two networks runs each frame was reported
twice, once fully presented and once "partial" (presented without that
frame's main-thread update); 2 of about 3,260 / 815 frames were dropped. The virtualized grids keep the
DOM at 1,000–2,400 elements whatever the list length. Traces:
`trace-networks-*.json`, `trace-gallery-*.json`.

## 5. Preview frame latency

`preview` scenario: txt2img jobs (768×768) started from the
Generate page (the baseline infotext pasted with "Read parameters", Generate
clicked), so the page shows its own job's live previews. Per frame:

- **image**: the sampler assigned the preview (`State.assign_current_image`,
  server probe, `time.time()`);
- **publish**: the encoder handed the JPEG to the WebSocket hub (probe);
- **receive**: the page's socket got the binary frame (a listener installed
  before the app's, `performance.timeOrigin + performance.now()`);
- **paint**: the `<img>` that shows the frame's object URL was decoded and the
  next animation frame ran (`MutationObserver` on `src`, `img.decode()`, then
  `requestAnimationFrame`).

Server and browser run on the same host and read the same wall clock; the
page's `Date.now()` and `timeOrigin + now()` differed by −0.4 ms (recorded
per run as `clockOffsetMs`). Frames are matched by job sequence and step
(the newest frame of that step published before it was received).

| | frames | total p50 | **total p95** | image → publish p95 | publish → receive p95 | receive → paint p95 |
|---|---|---|---|---|---|---|
| before (polling encoder; 3 jobs × 24 steps) | 69 | 104.3 ms | **177.0 ms** | 165.3 ms | 1.2 ms | 17.8 ms |
| after (4 jobs × 16 steps) | 60 | 13.4 ms | **20.7 ms** | 0.5 ms | 2.0 ms | 20.1 ms |

Before, the encoder thread woke every 1/6 s and encoded the newest preview,
so a frame waited 0–167 ms before it even left. The p99 (4.5 s) comes from
the last preview of two jobs (step 22 of 24), which left 4.0 and 5.5 s after
it was made: apparently the polling thread did not get to run while the
job's last step and decoding held the interpreter; event-driven, the frame is
encoded before the sampler moves on (after: max 25.6 ms).
Now `shared.state` wakes the encoder when a preview is assigned; the fps
setting only throttles (see Optimisations). Encoding takes 0.2 ms (median).
Raw rows: `preview-rows.json`, `before-preview-rows.json`.

## 6. UI overhead on generation speed

Sampling speed comes from the server's step probe (`it/s` = steps / time
between the first and last step callback; the first step is excluded).

### Finding and fixing the overhead

The first measurement (`overhead` scenario, 768×768, 24 steps, five pairs of
jobs, one with the Generate page connected and showing previews, one sent
through the API with no client, alternating order) gave **4.5 %** lower it/s
with the browser (0.1491 vs 0.1561 it/s; every browser job was slower than
the API job next to it). The server was idle after a page load (0.19
s of CPU, then nothing), so the cost was in the browser while it showed the
job:

- the Queue icon's "running" dot (`animate-pulse`, NavRail) animated for the
  whole job: Chromium processes used **10–11 % of a core** during a short
  job, **2.6–4.3 %** with the page's animations switched off by injected CSS
  (`ps` over every Chromium process on the machine, so other browsers add a
  little to both). An endless animation makes the compositor
  draw at display rate; on this machine the browser shares the GPU (and the
  fanless chassis' heat budget) with MPS sampling;
- after removing it, a 20 s trace during sampling (512×512, about 2.7 s per
  step) still counted 168 presented frames, about **23 per step**: the
  progress bars animated `width` (150 ms, layout every frame) and the header
  bar `transform` (300 ms) on every step;
- with both transitions removed, the page draws **4.1 frames per step**
  (preview, progress, step text; `preview` scenario, `framesPerStep`, from
  the second step to the last of a 16-step 768×768 job).

### Measuring what is left

Between-job comparisons are dominated by this machine's drift (it/s of
neighbouring identical jobs differed by up to 8 %), so the final number comes
from the `interleaved` scenario: one long job started from the Generate page
(the page owns it and paints its previews), and the server is switched every
`--window` seconds between serving the page (previews encoded, sent,
painted, progress events) and behaving as if no client were connected
(`serve_traced.py`'s mute file: nothing encoded or sent; the page stays open
and idle). Windows run in on, off, off, on order; every step interval inside
one window counts (the first one after a switch is dropped); each adjacent
on/off window pair gives one relative difference.

| run (`interleaved`) | steps | on: ms/step (n) | off: ms/step (n) | overall | window pairs: mean ± 95 % CI | previews painted | Chromium CPU |
|---|---|---|---|---|---|---|---|
| after removing the pulse, before removing the transitions, 768×768 | 100 | 7,542 (37) | 7,476 (37) | +0.88 % | +0.69 ± 1.90 % (12) | 56 | 1.1 % of a core |
| same, 512×512 | 150 | 2,702 (60) | 2,661 (62) | +1.53 % | +2.26 ± 2.87 % (13) | 78 | 2.9 % |
| **final, 768×768** | 100 | 7,358 (37) | 7,323 (37) | **+0.48 %** | **+0.49 ± 1.66 %** (12) | 56 | 0.4 % |
| **final, 512×512** | 150 | 3,099 (60) | 3,107 (58) | **−0.27 %** | **−0.59 ± 2.53 %** (15) | 82 | 0.7 % |
| final, both pooled | | | | | −0.11 ± 1.51 % (27; median +0.43 %, one-sided 95 % bound +1.14 %) | | |

(Positive = slower with the page served. 512×512 is the stricter case: the
per-step UI work is the same while a step is 2.4× shorter. The steps are
capped at 150 by the form.)

The final point estimates are under 1 % at both sizes, and the per-step UI
work explains no more: one 0.2 ms JPEG encode on the encoder thread, a send
on the server's event loop, and about four browser frames per step. The
measurement's resolution on this machine is about ±1.5 % (95 %), so an
overhead between 1 % and 1.4 % cannot be excluded by it; the earlier 4.5 %
and the per-frame savings above are well outside that noise. Raw:
`browser-overhead.json` (before), `browser-preview-overhead.json`
(between-job pairs after removing the pulse: +2.1 ± 3.8 %, six pairs, too
noisy to use), `il768/`, `il512/`, `final-il768/`, `final-il512/`
(`browser-interleaved.json` with every window's step times).

Resource accounting: during the between-job runs (768×768, after removing
the pulse, before removing the transitions) Chromium used 2.4 % of one core,
and the server process 3.3 % of a core with the page connected vs 2.8 %
without (preview encoding and WebSocket sends: +0.5 % of a core, on threads
other than the sampler). In the final interleaved runs Chromium used 0.4 %
(768×768) and 0.7 % (512×512) of one core averaged over the whole job (the
page was served half of the time).

## 7. Server startup

`tools/perf/startup.py`: fresh `--data-dir` with a copy of `config.json` per
start, models from `models/`, five recorded starts per tree and mode after a
warm-up start; time from spawning `launch.py --skip-prepare-environment
--skip-install` until the server answers (`GET /` with the UI,
`GET /sdapi/v1/sd-models` with `--nowebui`), and the resident memory of the
process tree at that moment. Neither tree loads the checkpoint at startup.

| tree | mode | ready, median (min–max) | `Startup time` line | RSS |
|---|---|---|---|---|
| old (`remi`, Gradio) | UI | 12.02 s (11.59–13.97) | printed after the UI is served | 1,160 MB |
| **new** | UI | **4.37 s** (4.23–4.77) | 4.3 s | 643 MB |
| old | `--nowebui` | 6.58 s (6.56–6.73) | 6.5 s | 703 MB |
| **new** | `--nowebui` | **4.19 s** (4.17–4.30) | 4.1 s | 647 MB |

The UI is ready 2.8× sooner and uses 45 % less memory; the new UI mode costs
0.2 s over API-only (the SPA is static files). Raw: `startup.json`.

With the 5,000 synthetic LoRAs of the other measurements the new server
reported 7.7–8.0 s (scanning the LoRA folder is part of "load scripts").

## Optimisations

| change | files | effect |
|---|---|---|
| The preview encoder is woken by `shared.state` when a preview is assigned (`State.live_preview_listeners`) instead of polling every 1/`live_preview_fps` s; the fps setting only throttles (a newer preview during the throttle wait replaces the older one) | `remi/realtime.py`, `modules/shared_state.py`, `remi/server/testing.py` (`FakeState`), tests in `tests/test_remi_server_ws.py` | preview latency p95 177 → 20.7 ms |
| No endless animation on the running-job indicators (NavRail dot and queue badge, queue card placeholder); documented as a frontend convention | `webui/src/app/NavRail.tsx`, `webui/src/features/queue/RunningCard.tsx`, `docs/remi-webui/frontend.md` | Chromium CPU during a job 10–11 % → 2.6–4.3 % of a core (same method); with the next change 0.4–0.7 % averaged over the interleaved jobs (served half of the time) |
| Progress bars without transitions | `webui/src/components/ui/progress.tsx`, `webui/src/app/Header.tsx` | ~23 → 4.1 frames drawn per step |
| The build writes `.br` / `.gz` copies of scripts, styles and SVGs; `/assets` serves them by `Accept-Encoding` (`Vary`, the existing gzip middleware passes them through) | `webui/vite.config.ts`, `remi/server/spa.py`, test in `tests/test_remi_server_app.py` | Generate view JS 547 → 488 KB on the wire, no per-request compression (37 ms of event-loop CPU per page load) |
| React in its own initial chunk | `webui/vite.config.ts` | no chunk over 150 KB gzip (largest 86.3 KB) |

Docs updated: `architecture.md` (preview encoder, precompressed assets),
`api.md` (preview timing), `upstream-absorption.md` (the `shared_state.py`
hook), `frontend.md` (no endless animations while a job runs).

## Caveats

- Headless Chromium. Frame pacing was 60 fps idle and the compositor traces
  show presented frames, but a headed browser on a high-refresh display, or
  a slower machine, will differ. The 4× CPU-throttled INP run is the margin
  check for slower machines.
- One machine, shared with a desktop session and memory-pressured; the speed
  comparisons are interleaved and reported with confidence intervals for that
  reason. Absolute it/s values are not comparable between sections.
- The overhead measurement's "off" condition keeps an idle page open (the
  page that owns the job). That page has no animation and receives nothing
  while muted, so it should cost nothing, but this was not measured on its
  own; the between-job comparison (no page at all) was too noisy to show it.
- The preview "paint" time is the start of the first animation frame after
  the image decoded, so it can be early by up to one frame (16.7 ms) relative
  to the frame reaching the screen.
