# Prompt Queue (for sd-webui-forge-classic / neo)

Queue up prompts on **txt2img** and **img2img** and run them back-to-back,
hands-free. Up to **100** prompts can be pending at once.

## Install

Extract this folder into your webui's `extensions` directory so you have:

```
extensions/
└── sd-webui-prompt-queue/
    ├── scripts/prompt_queue.py
    ├── javascript/prompt_queue.js
    ├── style.css
    └── README.md
```

Then restart the webui (a full restart, not just "Reload UI", so the API
routes get registered).

## How it works

* An **"📋 Add to Queue"** button appears directly under **Generate** on both
  txt2img and img2img. It snapshots the current **prompt + negative prompt**
  and adds them to the queue. All other settings (model, sampler, steps,
  size, seed, img2img source image, etc.) are taken from whatever is set on
  that tab *at the moment the item runs*.
* A **Queue** tab appears in the top bar, right next to img2img, with a live
  badge showing how many prompts are pending. There you can reorder (▲/▼),
  cancel (✕), interrupt a running item (■), re-queue finished ones (↻), and
  pause/resume the whole queue.
* When the webui finishes a generation (including one you started manually),
  the queue automatically fills in the next prompt and presses Generate for
  you. It never double-dispatches: it waits until the webui is fully idle.
* The queue is saved to `queue.json` inside the extension folder, so it
  survives restarts. After a restart the queue starts **paused** if it
  restored unfinished items — press **▶ Run queue** to continue.

## Notes & limitations

* A browser tab with the webui open must stay open for the queue to advance
  (the runner lives in the page, which keeps it compatible with any settings
  or extensions you use — it presses the real Generate button).
* When an item dispatches, the prompt boxes of the target tab are overwritten
  with the queued prompt.
* Don't combine with right-click "Generate forever" — the two will fight
  over the Generate button.
