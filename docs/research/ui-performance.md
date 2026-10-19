# Remi WebUI — UI performance

Measured 2026-10-15 against the scorecard's performance budgets
(`docs/remi-webui/scorecard.md`, dimension 5), and revisited 2026-10-19 after
an independent review reproduced the measurements ([Review of
2026-10-19](#review-of-2026-10-19)). The numbers come from the real server
with every built-in extension loaded and the real Anima Turbo model (the
2026-10-19 runs of the UI's per-step cost: a synthetic GPU sampler, no
model), driven by Chromium through Playwright and the DevTools protocol. The
tools are in `tools/perf/` and `webui/tools/perf/`; the raw results (JSON and
logs, without the large CDP traces) are committed under
[`perf-data/`](perf-data/) ([Raw data](#raw-data)).

## Result

| budget | measured (final) | result |
|---|---|---|
| Initial JS ≤ 300 KB gzip | **179.8 KB** gzip (4 files; 156.3 KB brotli, as served) | pass |
| No route chunk > 150 KB gzip, except justified lazy canvas / editor chunks | largest page chunk 28.2 KB (`generate`); largest lazy chunks: canvas editor 66.0 KB, CodeMirror 62.4 KB; largest initial chunk 86.3 KB | pass |
| Generate view ≤ 3,000 DOM nodes, all built-in extensions loaded | **1,174** elements (1,489 DOM nodes incl. text, CDP `Nodes`) on txt2img, 1,245 / 1,602 on img2img; heaviest single section opened: 1,529 | pass |
| INP p95 ≤ 50 ms typing a 300-token prompt, autocomplete on | **p95 24 ms** in all three recorded runs (idle machine: max 40 ms, INP 24 ms; heavily loaded machine: max 152 ms, INP 24 ms). The 4× CPU-throttled margin check: p95 24 / 40 / 48 ms, max 56 / 88 / 144 ms, INP 24 / 48 / 56 ms (idle / review / loaded) | pass (unthrottled); the throttled margin shrinks under load |
| 5,000 network cards scroll at 60 fps | **59.8–60.0 fps**, p95 frame 16.7 ms at 2,400 px/s over the whole list; frames over 20 ms: 0 (idle), 6, 9 of 3,257 (loaded); over 33 ms: 0 / 2; worst single frame 250 ms (warm pass, loaded) | pass (occasional hitches under load, below) |
| 10,000 gallery items scroll at 60 fps | **59.9–60.0 fps**, p95 frame 16.7–16.8 ms at 2,400 px/s, all 10,000 loaded; frames over 20 ms: 0 (idle), 6 of 5,678 (loaded); over 33 ms: 0 / 4 | pass (occasional hitches under load, below) |
| Preview frame latency p95 ≤ 150 ms | **p95 20.7 ms** (sampler → painted, idle machine); was 177 ms before this work. Under heavy load the server-side part grows (below) | pass |
| UI overhead on generation speed ≤ 1 % it/s | 2026-10-15: +0.48 % (768×768) and −0.27 % (512×512), intervals reaching +1.5 %; the review's 512×512 runs: +2.0 % and +3.8 %. After the per-step work was cut to **one React commit per step** ([below](#per-step-cost-of-the-page)), the real model, 2026-10-19 ([verify runs](#verify-runs-of-2026-10-19)): **768×768 −1.53 %, one-sided upper 95 % bound −0.05 %** (pass); **512×512 +0.28 % and +0.30 %** overall, window pairs +1.47 ± 2.17 % and +0.53 ± 1.02 %, **upper bounds 3.29 % and 1.39 %** (load average 7–26 from other work on the machine) | **pass at 768×768; not demonstrated at 512×512** (point estimates +0.3 %, the bound stays above 1 % on this loaded machine) |
| Server startup no slower than before the rewrite | UI mode **4.37 s** vs 12.02 s (Gradio); API only 4.19 s vs 6.58 s (medians of 5, idle machine). On a loaded machine everything is slower (the review: 9.6 s for the new UI mode; 2026-10-19 at load average 13–16: UI 23.9 s vs 68.0 s, API 29.8 s vs 33.6 s, alternating runs) | pass (the ratio; absolute times depend on load) |

Four things missed their budget or were found wasteful and were changed (see
[Optimisations](#optimisations)): the preview encoder polled, so frames waited
up to 1/6 s before leaving; endless CSS animations on the running-job
indicators and two progress-bar transitions kept the browser redrawing
during generation; SPA assets were
gzipped on every request; and the initial vendor chunk was just over 150 KB.
After the review, the page's per-step work was cut further (one paint per
step, no backdrop blur over the changing preview) and the measurement tools
were made reproducible from the repository.

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
| Old tree | git ref `remi` (`d5a71cad`) in a separate worktree, run with a Python that adds Gradio 4.40.0 to the same packages ([Recreating the environments](#recreating-the-environments)) |
| 2026-10-19 runs | same machine (macOS 27.0.1), Python 3.13 venv without Gradio (torch 2.14.1), Chromium of Playwright 1.64 (headless); several other agent sessions ran builds and test suites alongside: load average 13–16 on the 10 cores, server startup 30–150 s instead of 8 s |

The machine is fanless and was shared: sampling speed drifted from 1.6 s to
3.3 s per step at 512×512, and from 3.6 s to 7.9 s per step at 768×768, over
the session. Everything that compares speed below interleaves its conditions
for that reason.

## Method and tools

| tool | what it does |
|---|---|
| `tools/perf/fixtures.py create $PERF` | 5,000 LoRA files (valid metadata-only safetensors in 50 folders, each with a 96×128 PNG preview) and a `--data-dir` with 10,000 384×384 PNG outputs carrying an infotext; `remove` deletes everything (done after the runs) |
| `tools/perf/fixtures.py prompt …/prompt300.txt` | the typing test's prompt: 103 common Danbooru general tags, exactly 300 CLIP tokens, 1,203 characters |
| `tools/perf/serve_traced.py` | runs `launch.py` with probes added at import time (no server change): sampler step times, preview image assignment, preview frame publication, to a JSONL file (`time.time()`); a mute file makes the WebSocket hub act as if no client were connected; `REMI_PERF_FAKE_SAMPLER=<blocks>` replaces the processing with a synthetic GPU sampler (no model: DiT-like blocks on MPS, about 3 s per step with 28 blocks, a live preview per step as the real callback makes it); `REMI_PERF_DIST` serves another SPA build |
| `webui/tools/perf/bundle.mjs` | initial JS (scripts in `index.html` and their static imports) and every lazy entry, from the Vite manifest, gzip -6 and brotli -11 |
| `webui/tools/perf/browser.mjs` | Playwright + CDP scenarios: `load`, `inp`, `networks`, `gallery`, `preview` (latency, frames drawn per step, the page's CPU per step), `overhead`, `interleaved` (below); `--headed` runs a visible browser |
| `tools/perf/startup.py` | starts old and new trees five times each per mode, alternating (after an unrecorded warm-up start each), and times them; proxies are removed from the environment; the readiness probe waits for a slow answer instead of retrying every 2 s (on a busy machine the old Gradio page takes seconds per request, and abandoned probes kept it from ever answering) |

Commands, from the repository root. `$PY` is a Python with the WebUI's
requirements and no Gradio, `$PERF` a scratch directory (fixtures, traces,
results; not under `/tmp` if the runs span days: macOS purges it), and the
server is started once with the fixtures:

```sh
export PERF=$HOME/remi-perf OUT=$HOME/remi-perf/out
$PY tools/perf/fixtures.py create $PERF            # 5,000 LoRAs, 10,000 gallery images
$PY tools/perf/fixtures.py prompt $OUT/prompt300.txt
REMI_PERF_TRACE=$PERF/trace.jsonl REMI_PERF_MUTE=$PERF/mute $PY tools/perf/serve_traced.py \
  --skip-prepare-environment --skip-install --api --port 7861 \
  --data-dir $PERF/data --model-ref $PWD/models --lora-dirs $PERF/Lora &
cd webui
B="--base http://127.0.0.1:7861 --out $OUT --trace $PERF/trace.jsonl --mute $PERF/mute --prompt $OUT/prompt300.txt"
pnpm exec vite build --manifest --outDir $PERF/dist --emptyOutDir && node tools/perf/bundle.mjs $PERF/dist
node tools/perf/browser.mjs $B --only load,inp,networks,gallery
node tools/perf/browser.mjs $B --only inp --cpu-throttle 4
node tools/perf/browser.mjs $B --only networks,gallery --headed     # a visible window, real compositor
node tools/perf/browser.mjs $B --only preview --steps 16 --jobs 4
node tools/perf/browser.mjs $B --only interleaved --steps 150 --size 512x512 --window 15 --batch-count 4
node tools/perf/browser.mjs $B --only interleaved --steps 100 --size 768x768 --window 30 --batch-count 2
cd .. && $PY tools/perf/startup.py --old $OLD_TREE --old-python $OLD_PY --new-python $PY --runs 5 --out $OUT/startup.json
$PY tools/perf/fixtures.py remove $PERF
```

The UI's per-step cost without the model (any machine with MPS or CUDA):
start the server with `REMI_PERF_FAKE_SAMPLER=28` (and `REMI_PERF_DIST=<a
build>` to compare two frontends against the same server), then `node
tools/perf/browser.mjs $B --only preview --steps 16 --jobs 2 --size 512x512`.

### Recreating the environments

- The new tree: a Python 3.13 venv with `requirements.txt` (no Gradio:
  `tools/check.sh` fails its `env:gradio-free` step when Gradio is
  importable).
- The old tree, for `startup.py`: a worktree of the pre-rewrite code and a
  venv that adds Gradio to the same packages (Gradio 4.40.0 is what the old
  launcher installed, `GRADIO_PACKAGE` in its `modules/launch_utils.py`):

  ```sh
  git worktree add --detach ../remi-baseline d5a71cad   # git ref "remi"
  export OLD_TREE=$PWD/../remi-baseline OLD_PY=$PWD/../remi-oldenv/bin/python
  uv venv --python 3.13 ../remi-oldenv
  # Gradio 4.40 with the old tree's pins of the packages it shares (newer huggingface-hub lacks HfFolder)
  uv pip install --python $OLD_PY gradio==4.40.0 gradio_rangeslider==0.0.8 huggingface-hub==0.36.2 fastapi==0.127.1 pydantic==2.10.6
  # the old tree's other requirements are the new one's: reuse $PY's packages
  $PY -c "import site; print(site.getsitepackages()[0])" > "$($OLD_PY -c 'import site; print(site.getsitepackages()[0])')/_base.pth"
  $OLD_PY -c "import gradio, torch; print(gradio.__version__)"   # 4.40.0
  ```

  Keep both outside `/tmp`. A SOCKS proxy in the environment makes Gradio's
  `httpx` client fail at import (no `socksio`); `startup.py` drops proxy
  variables for that reason.

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

| run | interactions | > 16 ms | p50 | p95 | p99 | max | INP | long animation frames |
|---|---|---|---|---|---|---|---|---|
| 2026-10-15, idle machine, unthrottled | 1,203 | 262 | ≤ 16 | **24** | 24 | 40 | 24 | 0 |
| 2026-10-15, idle machine, 4× CPU throttling | 1,203 | 1,196 | 16 | **24** | 24 | 56 | 24 | 1 (79 ms) |
| 2026-10-19, the review (loaded machine), 4× CPU throttling | | | | **40** | | 88 | 48 | |
| 2026-10-19, load average 13–16, unthrottled | 1,203 | 356 | 16 | **24** | 32 | 152 | 24 | 1 (144 ms) |
| 2026-10-19, load average 13–16, 4× CPU throttling | 1,203 | 1,203 | 32 | **48** | 64 | 144 | 56 | 5 (51–132 ms) |

(Event Timing durations are rounded to 8 ms.) The typed text matched the
prompt exactly in every run. A first run before the optimisations gave the
same p95 (24 ms, max 32 ms). The budget (p95 ≤ 50 ms, unthrottled) holds in
every run. The 4× throttled run is the margin for slower machines: on an idle
machine it matched the unthrottled one; on a machine already running other
heavy work (the review, and the 2026-10-19 rerun with other builds and test
suites alongside) its p95 rose to 40–48 ms and web-vitals INP to 48–56 ms,
i.e. a slow machine that is also busy is at the edge of the budget. The
longest frames are React renders with CodeMirror updates
(`long-animation-frame` scripts: `react`, CodeMirror's `dist-*`; the worst
one 4.5 s into typing), not steady per-key work: p50 stays 16–32 ms.

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
`trace-networks-*.json`, `trace-gallery-*.json` (not committed: 40–250 MB each).

The same runs on 2026-10-19 with the machine under heavy load (load average
13–16 on 10 cores: other agent sessions building and testing), the same
fixtures and headless Chromium:

| list | speed | frames | fps | frame p95 / p99 / max | frames > 20 ms | > 33 ms | compositor: dropped of presented |
|---|---|---|---|---|---|---|---|
| networks | 2,400 px/s | 3,258 | **59.8** | 16.7 / 16.8 / 33.4 ms | 9 | 2 | 5 of 3,261 |
| | 9,600 px/s | 815 | 57.3 | 16.8 / 33.4 / 83.3 ms | 25 | 8 | 31 of 817 |
| | 2,400 px/s, warm | 3,258 | 59.3 | 16.7 / 16.8 / 250 ms | 14 | 7 | 24 of 3,283 |
| gallery | 2,400 px/s | 5,679 | **59.9** | 16.7 / 16.8 / 66.7 ms | 6 | 4 | 11 of 5,679 |
| | 9,600 px/s | 1,420 | 58.3 | 16.8 / 33.3 / 83.2 ms | 36 | 6 | 43 of 1,423 |
| | 2,400 px/s, warm | 5,679 | 59.7 | 16.7 / 16.8 / 100 ms | 11 | 6 | 28 of 5,679 |

The review's run (also loaded) found 6 of 3,258 frames over 20 ms on the
networks grid at 40 px per frame, and 5 over 20 ms plus 3 over 33 ms at
120 px per frame. So the grids hold 60 fps at the steady 2,400 px/s on an
idle machine (no frame over 20 ms) and 59.3–59.9 fps on a loaded one, where
a few frames in a thousand miss (0.2–0.4 % over 20 ms, single hitches up to
250 ms when another process takes the CPU); at fling speed the loaded
machine drops 2–3 % of frames. The p95 frame stays 16.7 ms throughout. These
are headless runs: the compositor's own reports (the `PipelineReporter`
column) agree with the rAF intervals, but a headed browser on a real display
(`browser.mjs --only networks,gallery --headed`) has not been measured.

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

On 2026-10-19 the same scenario ran 24 times against the synthetic sampler
(512×512, 64×64 previews) on the heavily loaded machine. Where nothing else
peaked, p95 was 19–66 ms before and 35–42 ms after this revision's changes
(receive → paint about 10–15 ms longer: the image is decoded first and drawn in
the next animation frame). In runs that coincided with test suites or
builds, the server-side parts grew for both builds alike, up to a total p95
of 490 ms (image → publish up to 141 ms: the encoder thread waiting to run;
publish → receive up to 299 ms: the event loop), so the 150 ms budget holds
on a machine with CPU to spare, not on one that is saturated by other work.

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

The final point estimates were under 1 % at both sizes, with one 0.2 ms
JPEG encode on the encoder thread, a send on the server's event loop, and
about four browser frames per step left. The measurement's resolution on
this machine was about ±1.5 % (95 %), so an overhead up to 1.5–1.9 % (the
one-sided upper bounds, [below](#review-of-2026-10-19)) was not excluded; the
earlier 4.5 % and the per-frame savings above are well outside that noise.
The review showed that the "pass" drawn from these point estimates did not
hold. Raw:
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

### Review of 2026-10-19

The review reran `interleaved` with the same model, sizes and windows on the
same machine (a loaded one). The table also gives each run's one-sided 95 %
upper bound, the result with the first step of every window dropped (the
first step after a switch was often slow), and the ratio of median step
times; the 2026-10-15 runs are reanalysed the same way (`browser.mjs` now
reports these as `pairStats`, `trimmedPairStats` and `medianOverheadPct`).

| run | window pairs | overall | pairs: mean ± 95 % CI | one-sided upper 95 % | first step dropped | medians |
|---|---|---|---|---|---|---|
| 2026-10-15, final 768×768, 30 s windows | 12 | +0.48 % | +0.49 ± 1.66 % | +1.85 % | −0.38 ± 2.04 % | +0.82 % |
| 2026-10-15, final 512×512, 15 s | 15 | −0.27 % | −0.59 ± 2.53 % | +1.49 % | −0.88 ± 2.73 % | +0.41 % |
| review, 512×512, 15 s | 16 | +2.05 % | +2.35 ± 2.45 % | +4.36 % | +1.04 ± 2.37 % | +1.51 % |
| review, 512×512, 15 s, again | 16 | +3.75 % | **+4.24 ± 1.84 %** | +5.76 % | +4.01 ± 2.37 % | +3.38 % |
| review, 768×768, 30 s | 13 | −2.61 % | −2.92 ± 4.37 % | +0.65 % | −4.03 ± 4.77 % | −0.52 % |

The second 512×512 run excludes 1 % (2.4–6.1 %), with or without the first
step of each window; the 768×768 runs are consistent with no overhead, and
512×512 is the stricter size (the same per-step UI work against a 2.4×
shorter step). The overhead budget is therefore **not demonstrated**, and
the result table says so. Raw: `perf-data/2026-10-19-review/`.

### Verify runs of 2026-10-19

`interleaved` with the real model after the per-step changes (the
documented commands: 512×512 150 steps, 15 s windows, `--batch-count 4`;
768×768 100 steps, 30 s windows, `--batch-count 2`), all 45 built-in
extensions on, headless Chromium, on the same machine with other heavy work
running (load average 13–26 during the first 512×512 run, 4–10 during the
others):

| run | window pairs | overall | pairs: mean ± 95 % CI | one-sided upper 95 % | first step dropped | medians | Hodges–Lehmann (upper 95 %) |
|---|---|---|---|---|---|---|---|
| 512×512 (load 13–26) | 65 | +0.28 % | +1.47 ± 2.17 % | +3.29 % | +0.66 ± 1.92 % | +0.17 % | +0.12 % (+1.19 %) |
| 512×512 (load 5–10) | 64 | +0.30 % | +0.53 ± 1.02 % | +1.39 % | +0.86 ± 1.42 % | +0.46 % | +0.57 % (+1.35 %) |
| **768×768** (load 4–8) | 24 | −1.53 % | −1.44 ± 1.68 % | **−0.05 %** | −1.83 ± 2.28 % | −0.93 % | −0.79 % (+0.07 %) |

At 768×768 the budget is met by the criterion set above (the one-sided
upper bound of the window pairs, with and without each window's first step,
≤ 1 %). At 512×512 every point estimate is +0.1 to +0.9 % (overall +0.28 %
and +0.30 %, i.e. about 9 ms of a 3.2 s step), but individual window pairs
swing by ±5–40 % when the other work on the machine spikes, and the bound
stays at 1.4 % even in the quieter run; the budget is **not demonstrated**
at 512×512 on this machine. Rerun the 512×512 command on an otherwise idle
machine to settle it. Raw: `perf-data/2026-10-19-verify/` (`il512/`,
`il512b/`, `il768/`).

### Per-step cost of the page

A trace of the Generate page during a job (2026-10-19, synthetic sampler,
below) showed what is left per step: the binary preview arrives and the
stage swaps its `<img>` (a frame, and often one more once the new image has
decoded); 60–200 ms later the step's `job.progress` arrives (the server
publishes it when the step counter changes, i.e. when the sampler has
dispatched the next step) and the header's progress line, the stage's
overlay and the Generate bar render again (a layout and another frame); the
stage overlay's `backdrop-blur` sat over the changing preview, so its blur
was redrawn with every frame. 3.3–4.2 frames drawn per step.

Changes (frontend only; the server sends the same messages):

- `webui/src/api/progress-gate.ts`: while previews arrive for every step of
  the running job, a `job.progress` waits for the next preview frame (at
  most two step intervals after the last frame, never over 10 s) and is drawn
  with it, so the counter, percentage and ETA change together with the image;
  previews every N steps, throttled previews or none: no waiting.
- `webui/src/api/paint-batch.ts`: the step's store updates (progress,
  preview) run in one animation frame, one React commit; the preview is
  decoded first (`img.decode()` on a detached image, at most 100 ms), so the
  stage swaps to an already decoded image in that same frame.
- No `backdrop-filter` on the stage's progress overlay (it sits on the
  preview) and on the header (only the page background is behind it; it
  carries the progress line).

A server variant that also published `job.progress` the moment a preview
was assigned was tried and dropped: the progress monitor deduplicates by
step, so it published nothing new, and it added a thread wake-up at the
moment the sampler thread dispatches the next step.

Measured with the synthetic sampler (`REMI_PERF_FAKE_SAMPLER=28`: about
3.5 s per step on the loaded machine, a 64×64 preview per step, 512×512),
the `preview` scenario (16 steps, two jobs; frames from a CDP trace of the
first job, CPU from the untraced second job), headless Chromium 1920×1080,
the same server serving a build from before and one from after the changes,
alternating, three runs each:

| build | frames drawn per step | dropped frames (first job) | renderer main-thread tasks per step | Chromium CPU per step (all processes) | preview latency p95 (receive → paint) |
|---|---|---|---|---|---|
| before | 3.29, 3.29, 3.93 | 9, 12, 15 | 6.6, 7.9, 7.4 ms | 35, 43, 41 ms | 22 (21), 19 (19), 38 (25) ms |
| after | **2.14, 1.64, 1.93** | 4, 2, 5 | 6.0, 5.7, 6.0 ms | **24, 25, 21 ms** | 37 (34), 42 (34), 39 (36) ms |

Frames are distinct compositor frames (`PipelineReporter` presented, by
frame sequence) over the traced steps. They include, once per job, about a
dozen frames of CSS transitions as the job starts (`transitionrun` events
in the trace) and, per step, the measuring
script's own animation-frame callback; in the steady state of the trace
(`perf-data/2026-10-19/preview-trace/`), a step is one main-thread commit
(one `FireAnimationFrame` with the progress, the image swap and a 0.5 ms
layout) where before it was two or three. The earlier set of runs of the
first version of the hold (the batch re-requested animation frames while a
preview decoded; `preview-ab/pv4-*`) gave the same picture (before 3.4–4.2,
after 1.6–1.9 frames; Chromium CPU 31–37 → 19–22 ms where nothing else ran).
CPU figures are noisy on a loaded machine; the frame count and the
direction of the CPU change are the robust part. The step's preview reaches
the screen about 10–15 ms later than before (decoded first, then drawn in
the next animation frame), well inside the 150 ms budget. Raw:
`perf-data/2026-10-19/preview-ab/` (`pv5-*` is the final code).

What this does to it/s was measured afterwards with the real model
([verify runs](#verify-runs-of-2026-10-19)). When this section was written the model was not loaded for
this revision, and the machine was too busy for the `interleaved` scenario
to resolve 1 %: a synthetic run before the changes (150 steps, 17 window
pairs) gave −0.87 ± 6.36 %, step times varying ±20 % with the other work on
the machine (in that run the model bar's loading spinner also turned in
both conditions, the harness had not yet marked the absent model as loaded;
it shows the noise, not the overhead). To mark the row pass: run `interleaved` with the real model on
an otherwise idle machine at 512×512 with `--batch-count 4` (600 steps in
the one job the page owns, about 40 window pairs instead of 16) and at
768×768, and require the one-sided upper 95 % bound (`pairStats.upper95`,
also `trimmedPairStats.upper95`) to be ≤ 1 %.

## 7. Server startup

`tools/perf/startup.py`: fresh `--data-dir` with a copy of `config.json` per
start, models from `models/`, five recorded starts per tree and mode after a
warm-up start (since 2026-10-19 the two trees alternate, so a load that
changes during the runs slows both alike); time from spawning `launch.py --skip-prepare-environment
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

These times are from an idle machine (2026-10-15). Startup is CPU- and
disk-bound import work, so it stretches with whatever else runs: the review
measured a 9.6 s median for the new UI mode on a loaded machine, and on
2026-10-19, with a load average of 13–16, single starts of the new server
took 26–150 s ("forge init" alone 7–35 s). Compare the two trees only in
the same conditions, alternating (as `startup.py` now does); the claim is
the ratio, not the absolute time.

Rerun under that load on 2026-10-19 (`startup.py --runs 3`, the trees
alternating, the old tree with the environment built as in [Recreating the
environments](#recreating-the-environments)):

| tree | mode | ready, median (min–max) | RSS |
|---|---|---|---|
| old (Gradio) | UI | 67.95 s (66.27–73.22) | 1,136 MB |
| **new** | UI | **23.93 s** (22.05–24.09) | 646 MB |
| old | `--nowebui` | 33.64 s (33.54–90.85) | 707 MB |
| **new** | `--nowebui` | **29.76 s** (20.45–46.97) | 649 MB |

Everything takes five to six times as long as on the idle machine, but the
new UI mode is still 2.8× faster than the old one (2.8× idle) and lighter;
API-only stays ahead (1.1×, 1.6× idle). Raw: `perf-data/2026-10-19/startup.json`.

## Optimisations

| change | files | effect |
|---|---|---|
| The preview encoder is woken by `shared.state` when a preview is assigned (`State.live_preview_listeners`) instead of polling every 1/`live_preview_fps` s; the fps setting only throttles (a newer preview during the throttle wait replaces the older one) | `remi/realtime.py`, `modules/shared_state.py`, `remi/server/testing.py` (`FakeState`), tests in `tests/test_remi_server_ws.py` | preview latency p95 177 → 20.7 ms |
| No endless animation on the running-job indicators (NavRail dot and queue badge, queue card placeholder); documented as a frontend convention | `webui/src/app/NavRail.tsx`, `webui/src/features/queue/RunningCard.tsx`, `docs/remi-webui/frontend.md` | Chromium CPU during a job 10–11 % → 2.6–4.3 % of a core (same method); with the next change 0.4–0.7 % averaged over the interleaved jobs (served half of the time) |
| Progress bars without transitions | `webui/src/components/ui/progress.tsx`, `webui/src/app/Header.tsx` | ~23 → 4.1 frames drawn per step |
| The build writes `.br` / `.gz` copies of scripts, styles and SVGs; `/assets` serves them by `Accept-Encoding` (`Vary`, the existing gzip middleware passes them through) | `webui/vite.config.ts`, `remi/server/spa.py`, test in `tests/test_remi_server_app.py` | Generate view JS 547 → 488 KB on the wire, no per-request compression (37 ms of event-loop CPU per page load) |
| React in its own initial chunk | `webui/vite.config.ts` | no chunk over 150 KB gzip (largest 86.3 KB) |
| 2026-10-19: a step's `job.progress` waits for its preview frame while previews come every step; the step's updates are applied in one animation frame, the preview decoded before it is swapped in | `webui/src/api/progress-gate.ts`, `webui/src/api/paint-batch.ts`, `webui/src/api/ws.ts`, `webui/src/app/ws-bridge.ts`, tests in `paint-batch.test.ts`, `progress-gate.test.ts`, `ws.test.ts` | frames drawn per step and the page's CPU per step: [Per-step cost of the page](#per-step-cost-of-the-page) |
| 2026-10-19: no `backdrop-filter` on surfaces that change every step | `webui/src/pages/generate/Stage.tsx` (progress overlay), `webui/src/app/Header.tsx` | the blur is no longer redrawn with each preview / progress update |
| 2026-10-19: measurement tools: synthetic sampler and SPA override in `serve_traced.py`; per-step page cost, distinct frame count and `--save-trace` in `preview`; `--batch-count`, consecutive steps only, one-sided bound, trimmed and median estimates in `interleaved`; alternating, proxy-free `startup.py` whose probe waits for slow answers | `tools/perf/serve_traced.py`, `webui/tools/perf/browser.mjs`, `tools/perf/startup.py`, tests in `tests/test_remi_perf_tools.py` | reproducible from the repository; the review's analysis is part of the output |

Docs updated: `architecture.md` (preview encoder, precompressed assets, one
paint per step), `api.md` (preview and progress timing),
`upstream-absorption.md` (the `shared_state.py` hook), `frontend.md` (no
endless animations while a job runs; one commit and one frame per step, no
backdrop blur on per-step surfaces).

## Raw data

[`perf-data/`](perf-data/) holds every result file of these measurements
except the CDP traces (40–250 MB each; `browser.mjs` writes them next to the
results):

| folder | what |
|---|---|
| `2026-10-15/` | the original runs: `bundle-*.json`, `browser-load*.json` (load, INP, scrolling), `browser-inp*.json`, `browser-networks-gallery.json`, `before-preview*.json` and `final/` (preview latency before / after), `browser-overhead.json`, `browser-preview-overhead.json`, `il512/`, `il768/`, `final-il512/`, `final-il768/` (interleaved windows with every step time), `o512/`, `startup.json`, the console logs and `prompt300.txt` (the typing test's prompt) |
| `2026-10-19-review/` | the review's interleaved runs (`il512/`, `il512b/`, `il768/`) and preview runs (`out1/`, `out2/`) |
| `2026-10-19-verify/` | the verify stage's interleaved runs with the real model (`il512/`, `il512b/`, `il768/`) |
| `2026-10-19/` | the reruns of this revision: `fx-out/` (load, INP, INP at 4× throttling, scrolling, under load), `preview-ab/` (the before / after runs of the per-step cost, synthetic sampler), `interleaved-synthetic/` (the synthetic interleaved run), `startup.json` / `startup.log` (startup under load), `preview-trace/` (a compressed CDP trace of the final build during a job) |

## Caveats

- Headless Chromium. Frame pacing was 60 fps idle and the compositor traces
  show presented and dropped frames, but a headed browser on a high-refresh
  display, or a slower machine, will differ; no headed run is recorded yet
  (`--headed`). The 4× CPU-throttled INP run is the margin check for slower
  machines.
- One machine, shared with a desktop session and memory-pressured (and on
  2026-10-19 with several other heavy jobs); the speed comparisons are
  interleaved and reported with confidence intervals for that reason.
  Absolute it/s values and startup times are not comparable between
  sections or dates.
- The synthetic sampler loads the GPU like a DiT of about Anima's step time
  but has none of the real pipeline's Python work per step; it measures what
  the page costs, not the overhead on the real model.
- The overhead measurement's "off" condition keeps an idle page open (the
  page that owns the job). That page has no animation and receives nothing
  while muted, so it should cost nothing, but this was not measured on its
  own; the between-job comparison (no page at all) was too noisy to show it.
- The preview "paint" time is the start of the first animation frame after
  the image decoded, so it can be early by up to one frame (16.7 ms) relative
  to the frame reaching the screen.
