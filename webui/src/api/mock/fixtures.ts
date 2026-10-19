/**
 * Mock fixtures (VITE_MOCK=1, and the renderer tests). The txt2img schema
 * uses every node kind and condition operator, including a realistic
 * Anima-like section, so the renderer can be built without the backend. Its
 * shape follows the real `tab_schema()` (see `contract/*.json`, dumped from
 * the backend by `tools/dump_schemas.py`): sampler and seed are scripts placed
 * by slots, Hires. fix is a core section whose toggle is a conflict param.
 */
import type { Bootstrap, Models, PageSchema, PromptStyle } from "@/api/types"
import type { ChoiceOption, Node, ScriptSchema, TabSchema } from "@/ui/schema"

const options = (values: string[]): ChoiceOption[] => values.map((value) => ({ value, label: value }))

export const SAMPLERS = options([
  "Euler a",
  "Euler",
  "DPM++ 2M",
  "DPM++ 2M SDE",
  "DPM++ SDE",
  "DPM++ 3M SDE",
  "DDIM",
  "UniPC",
  "LCM",
  "Heun",
  "DPM2",
  "LMS",
  "Restart",
  "ER SDE",
])

export const SCHEDULERS = options(["Automatic", "Simple", "Karras", "Exponential", "SGM Uniform", "Beta"])

// ---------------------------------------------------------------- core form

/** img2img modes of the real form (remi.core.forms.IMG2IMG_MODES, without Batch). */
const IMG2IMG_MODES = ["img2img", "Sketch", "Inpaint", "Inpaint sketch", "Inpaint upload"]

function coreLayout(tab: "txt2img" | "img2img"): Node {
  // txt2img recomputes the Hires. fix output line when the size changes.
  const hires = tab === "txt2img" ? { on_change: "hr_resolution" } : {}
  const img2img: Node[] =
    tab === "img2img"
      ? [
          // Like the real form: the mode strip selects which layers the canvas edits over `init_img`
          // (the stage's image editor); the inpaint options show in the inpaint modes.
          {
            type: "group",
            children: [
              { type: "choice", name: "mode", label: "Mode", choices: options(IMG2IMG_MODES), style: "radio", hidden: true },
              { type: "tabs", bind: "mode", children: IMG2IMG_MODES.map((mode) => ({ type: "tab", title: mode, value: mode, children: [] })) },
              { type: "image", name: "init_img", label: "Image for img2img", kind: "image", mode: "RGBA" },
              { type: "image", name: "sketch", label: "Sketch", kind: "sketch", source: "init_img", mode: "RGBA", visible_if: { param: "mode", in: ["Sketch", "Inpaint sketch"] } },
              { type: "image", name: "mask", label: "Mask", kind: "mask", source: "init_img", visible_if: { param: "mode", eq: "Inpaint" } },
              { type: "image", name: "mask_upload", label: "Mask", kind: "mask", visible_if: { param: "mode", eq: "Inpaint upload" } },
            ],
          },
          {
            type: "group",
            visible_if: { param: "mode", in: ["Inpaint", "Inpaint sketch", "Inpaint upload"] },
            children: [
              { type: "slider", name: "mask_blur", label: "Mask blur", minimum: 0, maximum: 64, step: 1, integer: true },
              {
                type: "row",
                children: [
                  { type: "choice", name: "inpaint_full_res", label: "Inpaint area", choices: options(["Whole picture", "Only masked"]), style: "radio" },
                  { type: "slider", name: "inpaint_full_res_padding", label: "Only masked padding, pixels", minimum: 0, maximum: 256, step: 4, integer: true },
                ],
              },
            ],
          },
          {
            type: "slider",
            name: "denoising_strength",
            label: "Denoising strength",
            minimum: 0,
            maximum: 1,
            step: 0.01,
            info: "How much the source image is changed; 0 keeps it, 1 replaces it.",
          },
        ]
      : []

  return {
    type: "column",
    children: [
      { type: "text", name: "prompt", label: "Prompt", lines: 3, max_lines: 12, prompt: true, placeholder: "Prompt\n({{generate}} to Generate ; {{skip}} to Skip ; {{interrupt}} to Interrupt)" },
      { type: "text", name: "negative_prompt", label: "Negative prompt", lines: 2, max_lines: 8, prompt: true, placeholder: "Negative prompt" },
      { type: "multichoice", name: "styles", label: "Styles", choices: styles.map((style) => ({ value: style.name, label: style.name })) },
      ...img2img,
      { type: "slot", section: "sampler" },
      {
        type: "row",
        children: [
          {
            type: "column",
            width: "fill",
            children: [
              { type: "slider", name: "width", label: "Width", minimum: 64, maximum: 2048, step: 8, integer: true, infotext: "Size-1", ...hires },
              { type: "slider", name: "height", label: "Height", minimum: 64, maximum: 2048, step: 8, integer: true, infotext: "Size-2", ...hires },
            ],
          },
          { type: "button", label: "Swap", action: "swap_dimensions", variant: "tool" },
          {
            type: "column",
            width: 4,
            children: [
              { type: "slider", name: "batch_count", label: "Batch count", minimum: 1, maximum: 100, step: 1, integer: true },
              { type: "slider", name: "batch_size", label: "Batch size", minimum: 1, maximum: 8, step: 1, integer: true },
            ],
          },
        ],
      },
      {
        type: "slider",
        name: "cfg_scale",
        label: "CFG Scale",
        minimum: 1,
        maximum: 30,
        step: 0.5,
        info: "How strongly the image follows the prompt.",
      },
      {
        type: "markdown",
        text: "CFG **1** disables the negative prompt.",
        visible_if: { param: "cfg_scale", lte: 1 },
      },
      { type: "slot", section: "seed" },
      ...(tab === "txt2img"
        ? [
            {
              type: "section",
              title: "Hires. fix",
              open: false,
              enable: {
                type: "toggle",
                name: "enable_hr",
                label: "Hires. fix",
                exclusive: { group: "pid_hires", member: "hires" },
                on_change: "hr_resolution",
              },
              children: [
                { type: "output", name: "hr_final_resolution", kind: "text" },
                {
                  type: "row",
                  children: [
                    { type: "choice", name: "hr_upscaler", label: "Upscaler", choices: options(["Latent", "Latent (bicubic)", "ESRGAN_4x", "R-ESRGAN 4x+", "SwinIR 4x"]), width: 6 },
                    { type: "slider", name: "hr_second_pass_steps", label: "Hires steps", minimum: 0, maximum: 150, step: 1, integer: true, width: 6 },
                  ],
                },
                {
                  type: "row",
                  children: [
                    { type: "slider", name: "denoising_strength", label: "Denoising strength", minimum: 0, maximum: 1, step: 0.01, width: 6 },
                    { type: "slider", name: "hr_scale", label: "Upscale by", minimum: 1, maximum: 4, step: 0.05, width: 6, on_change: "hr_resolution" },
                  ],
                },
                {
                  type: "markdown",
                  text: "> Upscaling beyond **2×** needs a lot of VRAM.",
                  visible_if: { param: "hr_scale", gt: 2 },
                },
                {
                  type: "group",
                  title: "Hires prompts",
                  open: false,
                  children: [
                    { type: "text", name: "hr_prompt", label: "Hires prompt", lines: 2, prompt: true, placeholder: "Leave empty to reuse the prompt" },
                    { type: "text", name: "hr_negative_prompt", label: "Hires negative prompt", lines: 2, prompt: true },
                  ],
                },
              ],
            } satisfies Node,
          ]
        : []),
      { type: "slot", section: "accordions" },
      { type: "separator" },
      {
        type: "group",
        children: [
          {
            type: "choice",
            name: "script",
            label: "Script",
            choices: options(["None", "X/Y/Z plot", "Prompt matrix"]),
            info: "Selectable scripts change how the batch is produced.",
          },
          { type: "slot", section: "scripts" },
        ],
      },
    ],
  }
}

function coreDefaults(tab: "txt2img" | "img2img") {
  return {
    prompt: "1girl, solo, (masterpiece:1.2), cherry blossoms, <lora:watercolor_pastel_v1_0:0.8>",
    negative_prompt: "lowres, bad anatomy, watermark",
    width: 1024,
    height: 1024,
    batch_count: 1,
    batch_size: 1,
    cfg_scale: 5,
    styles: [],
    script: "None",
    ...(tab === "txt2img"
      ? { enable_hr: false, hr_upscaler: "Latent", hr_second_pass_steps: 0, denoising_strength: 0.7, hr_scale: 2, hr_prompt: "", hr_negative_prompt: "" }
      : {
          mode: "img2img",
          init_img: null,
          sketch: null,
          mask: null,
          mask_upload: null,
          mask_blur: 4,
          inpaint_full_res: "Only masked",
          inpaint_full_res_padding: 32,
          denoising_strength: 0.75,
        }),
  }
}

// ---------------------------------------------------------------- scripts

const ANIMA: ScriptSchema = {
  id: "anima",
  title: "Anima",
  kind: "alwayson",
  section: null,
  order: 0,
  layout: {
    type: "section",
    title: "Anima",
    open: true,
    enable: { type: "toggle", name: "enable", label: "Enable Anima" },
    children: [
      {
        type: "tabs",
        children: [
          {
            type: "tab",
            title: "Preset",
            children: [
              {
                type: "choice",
                name: "preset",
                label: "Preset",
                choices: [
                  { value: "balanced", label: "Balanced" },
                  { value: "detail", label: "Detail" },
                  { value: "speed", label: "Speed" },
                ],
                style: "radio",
                preset: "preset",
              },
              {
                type: "row",
                children: [
                  { type: "button", label: "Apply preset", action: "apply_preset", variant: "primary" },
                  { type: "button", label: "Reset to recommended", action: "recommended", variant: "ghost" },
                ],
              },
              {
                type: "markdown",
                text: "Presets set **sampler**, **steps** and **CFG** together. See the [Anima guide](https://example.com/anima) for details.\n\n- *Balanced*: quality and speed\n- *Detail*: more steps, lower CFG\n- *Speed*: fewest steps",
              },
              { type: "output", name: "status", kind: "text", label: "Status" },
            ],
          },
          {
            type: "tab",
            title: "Sampling",
            children: [
              {
                type: "group",
                enabled_if: { param: "enable" },
                children: [
                  {
                    type: "row",
                    children: [
                      { type: "number", name: "shift", label: "Timestep shift", minimum: 0, maximum: 10, step: 0.1, precision: 2, infotext: "Anima shift" },
                      { type: "slider", name: "rescale", label: "Guidance rescale", minimum: 0, maximum: 1, step: 0.05 },
                    ],
                  },
                  {
                    type: "multichoice",
                    name: "quality_tags",
                    label: "Quality tags",
                    choices: options(["masterpiece", "best quality", "very aesthetic", "absurdres", "newest", "highres"]),
                    allow_custom: true,
                    info: "Appended to the prompt; type to add your own.",
                  },
                  {
                    type: "row",
                    children: [
                      { type: "toggle", name: "lock_steps", label: "Lock steps", exclusive: { group: "anima-lock", member: "steps" } },
                      { type: "toggle", name: "lock_cfg", label: "Lock CFG", exclusive: { group: "anima-lock", member: "cfg" } },
                    ],
                  },
                  { type: "color", name: "tint", label: "Color tint", visible_if: { any: [{ param: "preset", eq: "detail" }, { param: "core.enable_hr" }] } },
                ],
              },
              {
                type: "markdown",
                text: "Enable Anima to edit these parameters.",
                visible_if: { not: { param: "enable" } },
              },
            ],
          },
          {
            type: "tab",
            title: "Blocks",
            children: [
              {
                type: "custom",
                name: "block_weights",
                label: "Block weights",
                widget: "block-weights",
                props: {
                  blocks: ["IN0", "IN1", "IN2", "IN3", "MID", "OUT0", "OUT1", "OUT2", "OUT3"],
                  minimum: 0,
                  maximum: 1.5,
                  step: 0.05,
                  presets: [
                    { label: "All 1", values: [1, 1, 1, 1, 1, 1, 1, 1, 1] },
                    { label: "Style only", values: [0, 0, 0, 0, 0, 1, 1, 1, 1] },
                  ],
                },
              },
              { type: "json", name: "advanced", label: "Advanced (JSON)", info: "Raw options passed to the sampler patch." },
              { type: "state", name: "internal_state", label: "" },
            ],
          },
        ],
      },
    ],
  },
  defaults: {
    enable: false,
    preset: "balanced",
    shift: 3,
    rescale: 0,
    quality_tags: ["masterpiece", "best quality"],
    lock_steps: false,
    lock_cfg: false,
    tint: "#ff9ec7",
    block_weights: [1, 1, 1, 1, 1, 1, 1, 1, 1],
    advanced: { sigma_max: 14.6, rho: 7 },
    internal_state: { version: 2 },
  },
  args: ["enable", "preset", "shift", "rescale", "quality_tags", "lock_steps", "lock_cfg", "tint", "block_weights", "advanced", "internal_state"],
}

const CONTROLNET: ScriptSchema = {
  id: "controlnet",
  title: "ControlNet",
  kind: "alwayson",
  section: null,
  order: 10,
  layout: {
    type: "section",
    title: "ControlNet Unit 0",
    enable: { type: "toggle", name: "enabled", label: "Enable" },
    children: [
      {
        type: "row",
        children: [
          { type: "image", name: "image", label: "Control image", kind: "image", width: 6 },
          { type: "image", name: "mask", label: "Control mask", kind: "mask", source: "image", width: 6 },
        ],
      },
      {
        type: "choice",
        name: "module",
        label: "Preprocessor",
        style: "radio",
        choices: options(["none", "canny", "depth_anything_v2", "openpose_full", "lineart_anime", "tile_resample"]),
      },
      {
        type: "choice",
        name: "model",
        label: "Model",
        allow_custom: true,
        choices: options(["None", "control_v11p_sd15_canny", "noobai_xl_depth", "illustrious_lineart"]),
        on_change: "model_changed",
      },
      { type: "slider", name: "weight", label: "Control weight", minimum: 0, maximum: 2, step: 0.05 },
      { type: "file", name: "custom_model", label: "Custom model file", accept: ".safetensors,.pth" },
      {
        type: "markdown",
        text: "Anima is enabled: control weights are rescaled automatically.",
        visible_if: { all: [{ param: "anima.enable" }, { param: "enabled" }] },
      },
      {
        type: "html",
        text: '<p>Need models? See the <a href="https://huggingface.co/lllyasviel">model list</a>.<script>alert("xss")</script><img src="x" onerror="alert(1)"></p>',
      },
    ],
  },
  defaults: { enabled: false, image: null, mask: null, module: "none", model: "None", weight: 1, custom_model: null },
  args: ["enabled", "image", "mask", "module", "model", "weight", "custom_model"],
}

const BATCH: ScriptSchema = {
  id: "batch-tools",
  title: "Batch tools",
  kind: "alwayson",
  section: null,
  order: 20,
  layout: {
    type: "section",
    title: "Batch tools",
    children: [
      { type: "image_list", name: "references", label: "Reference images" },
      {
        type: "multichoice",
        name: "steps_to_save",
        label: "Save intermediate",
        style: "checkboxes",
        choices: options(["Before hires", "Mask", "Grid"]),
      },
      { type: "button", label: "Analyze references", action: "analyze", variant: "secondary" },
      { type: "gallery", name: "analysis", label: "Analysis" },
      { type: "output", name: "report", kind: "json", label: "Report" },
      { type: "output", name: "summary", kind: "html" },
    ],
  },
  defaults: { references: [], steps_to_save: [] },
  args: ["references", "steps_to_save"],
}

/** Mirrors the real `seed` script: an untitled group in the `seed` slot. */
const SEED: ScriptSchema = {
  id: "seed",
  title: "Seed",
  kind: "alwayson",
  section: "seed",
  order: 0,
  layout: {
    type: "group",
    open: true,
    children: [
      {
        type: "row",
        children: [
          { type: "seed", name: "seed", label: "Seed", infotext: "Seed", width: "fill" },
          { type: "toggle", name: "subseed_show", label: "Extra" },
        ],
      },
      {
        type: "row",
        visible_if: { param: "subseed_show" },
        children: [
          { type: "seed", name: "subseed", label: "Variation seed", infotext: "Variation seed", width: "fill" },
          { type: "slider", name: "subseed_strength", label: "Variation strength", minimum: 0, maximum: 1, step: 0.01, width: 6 },
        ],
      },
    ],
  },
  defaults: { seed: -1, subseed_show: false, subseed: -1, subseed_strength: 0 },
  args: ["seed", "subseed_show", "subseed", "subseed_strength"],
}

/** Mirrors the real `sampler` script: a row in the `sampler` slot; its choices are conflict params. */
const SAMPLER: ScriptSchema = {
  id: "sampler",
  title: "Sampler",
  kind: "alwayson",
  section: "sampler",
  order: 0,
  layout: {
    type: "row",
    children: [
      { type: "choice", name: "sampler_name", label: "Sampling method", choices: SAMPLERS, width: 5, infotext: "Sampler" },
      { type: "choice", name: "scheduler", label: "Schedule type", choices: SCHEDULERS, width: 4, infotext: "Schedule type" },
      { type: "slider", name: "steps", label: "Sampling steps", minimum: 1, maximum: 150, step: 1, integer: true, width: "fill", infotext: "Steps" },
    ],
  },
  defaults: { steps: 28, sampler_name: "Euler a", scheduler: "Automatic" },
  args: ["steps", "sampler_name", "scheduler"],
}

/** Mirrors the real `refiner` script: a section with an enable toggle in the `accordions` slot. */
const REFINER: ScriptSchema = {
  id: "refiner",
  title: "Refiner",
  kind: "alwayson",
  section: "accordions",
  order: 0,
  layout: {
    type: "section",
    title: "Refiner",
    open: false,
    enable: { type: "toggle", name: "enable", label: "Refiner" },
    children: [
      {
        type: "row",
        children: [
          {
            type: "choice",
            name: "checkpoint",
            label: "Checkpoint",
            choices: options(["", "animagine-xl-4.0.safetensors", "noobai-xl-vpred-1.0.safetensors"]),
            width: "fill",
          },
          { type: "slider", name: "switch_at", label: "Switch at", minimum: 0.01, maximum: 1, step: 0.01, width: 6 },
        ],
      },
    ],
  },
  defaults: { enable: false, checkpoint: "", switch_at: 0.8 },
  args: ["enable", "checkpoint", "switch_at"],
}

const XYZ: ScriptSchema = {
  id: "xyz-plot",
  title: "X/Y/Z plot",
  kind: "selectable",
  section: "scripts",
  order: 0,
  visible_if: { param: "core.script", eq: "X/Y/Z plot" },
  layout: {
    type: "column",
    children: [
      {
        type: "row",
        children: [
          { type: "choice", name: "x_type", label: "X type", choices: options(["Nothing", "Seed", "Steps", "CFG Scale", "Sampler"]), width: 4 },
          { type: "text", name: "x_values", label: "X values", placeholder: "1-5, 10", width: "fill" },
        ],
      },
      { type: "toggle", name: "draw_legend", label: "Draw legend" },
    ],
  },
  defaults: { x_type: "Nothing", x_values: "", draw_legend: true },
  args: ["x_type", "x_values", "draw_legend"],
}

const PROMPT_MATRIX: ScriptSchema = {
  id: "prompt-matrix",
  title: "Prompt matrix",
  kind: "selectable",
  section: "scripts",
  order: 1,
  visible_if: { param: "core.script", eq: "Prompt matrix" },
  layout: {
    type: "column",
    children: [
      { type: "toggle", name: "put_at_start", label: "Put variable parts at start of prompt" },
      { type: "number", name: "margin", label: "Grid margins (px)", minimum: 0, maximum: 500, step: 2, integer: true },
    ],
  },
  defaults: { put_at_start: false, margin: 0 },
}

// An extension asking for a slot this tab does not have, with JSON params.
const MISC: ScriptSchema = {
  id: "misc-ext",
  title: "Misc extension",
  kind: "alwayson",
  section: "unknown-slot",
  order: 99,
  layout: {
    type: "group",
    open: true,
    children: [
      { type: "json", name: "flag", label: "Flag" },
      { type: "json", name: "pair", label: "Pair" },
    ],
  },
  defaults: { flag: true, pair: [0.5, "auto"] },
}

export function tabSchema(tab: "txt2img" | "img2img"): TabSchema {
  return {
    version: 1,
    tab,
    core: { owner: "core", layout: coreLayout(tab), defaults: coreDefaults(tab) },
    scripts: [SEED, SAMPLER, REFINER, ANIMA, CONTROLNET, BATCH, XYZ, PROMPT_MATRIX, MISC],
    conflict_params: [
      ...(tab === "txt2img" ? ["core.enable_hr"] : []),
      "anima.lock_cfg",
      "anima.lock_steps",
      "refiner.checkpoint",
      "refiner.enable",
      "sampler.sampler_name",
      "sampler.scheduler",
    ],
  }
}

export const extrasSchema: TabSchema = {
  version: 1,
  tab: "extras",
  core: {
    owner: "core",
    layout: {
      type: "column",
      children: [
        { type: "image", name: "image", label: "Source" },
        { type: "choice", name: "upscaler_1", label: "Upscaler", choices: options(["None", "Lanczos", "Nearest", "R-ESRGAN 4x+"]) },
        { type: "slider", name: "upscaling_resize", label: "Scale by", minimum: 1, maximum: 8, step: 0.05 },
      ],
    },
    defaults: { image: null, upscaler_1: "Lanczos", upscaling_resize: 2 },
  },
  scripts: [],
}

export const toolboxPage: PageSchema = {
  version: 1,
  id: "toolbox",
  title: "Toolbox",
  owner: "page:toolbox",
  layout: {
    type: "column",
    children: [
      { type: "markdown", text: "## Toolbox\nA schema-rendered extension page." },
      { type: "text", name: "text", label: "Text to transform", lines: 3 },
      { type: "choice", name: "transform", label: "Transform", style: "radio", choices: options(["upper", "lower", "reverse"]) },
      { type: "button", label: "Run", action: "run", variant: "primary" },
      { type: "output", name: "result", kind: "text", label: "Result" },
      // Like Prompt Workshop: an action row mixing buttons and send widgets, two columns of fields, a status line.
      {
        type: "section",
        title: "Prompt kit",
        open: true,
        children: [
          { type: "text", name: "kit_prompt", label: "Consolidated prompt", lines: 2 },
          {
            type: "row",
            children: [
              { type: "button", label: "Build prompt", action: "kit_build", variant: "primary" },
              { type: "custom", name: "kit_send", widget: "send-button", label: "", persist: false, submit: false, props: { to: "prompt", source: "kit_prompt", label: "Send to txt2img Prompt" } },
              { type: "custom", name: "kit_replace", widget: "send-button", label: "", persist: false, submit: false, props: { to: "prompt", source: "kit_prompt", mode: "replace", label: "Replace txt2img Prompt" } },
              { type: "button", label: "Clear all fields", action: "kit_clear", variant: "danger" },
            ],
          },
          {
            type: "row",
            children: [
              { type: "column", children: [{ type: "text", name: "kit_quality", label: "Quality" }, { type: "text", name: "kit_subject", label: "Subject" }] },
              { type: "column", children: [{ type: "text", name: "kit_series", label: "Series" }, { type: "text", name: "kit_general", label: "General tags" }] },
            ],
          },
          { type: "output", name: "kit_status", kind: "text" },
        ],
      },
    ],
  },
  defaults: { text: "Hello Remi", transform: "upper", kit_prompt: "", kit_send: null, kit_replace: null, kit_quality: "masterpiece", kit_subject: "1girl", kit_series: "", kit_general: "smile" },
}

// ---------------------------------------------------------------- app

export const bootstrap: Bootstrap = {
  version: "remi-mock",
  features: ["jobs", "uploads", "preview.binary", "folders.open"],
  tabs: ["txt2img", "img2img", "extras", "png-info", "gallery", "networks", "queue", "settings", "extensions"],
  pages: [{ id: "toolbox", title: "Toolbox", icon: "wrench" }],
  locales: [{ id: "zh_CN", name: "简体中文" }],
  locale: null,
  options: {
    sd_model_checkpoint: "animagine-xl-4.0.safetensors [a1b2c3]",
    forge_preset: "xl",
    forge_additional_modules: [],
    forge_unet_storage_dtype: "Automatic",
    keyedit_precision_attention: 0.1,
    keyedit_precision_extra: 0.05,
    keyedit_move: true,
    ctrl_enter_interrupt: false,
    show_progress_in_title: true,
    prevent_screen_sleep_during_generation: true,
    notification_volume: 60,
    js_modal_lightbox: true,
    js_modal_lightbox_initially_zoomed: false,
    js_live_preview_in_modal_lightbox: true,
    send_seed: true,
    send_size: true,
  },
  models: { checkpoint: "animagine-xl-4.0.safetensors [a1b2c3]", vae: "Automatic", text_encoders: [], modules: [] },
  not_ported: [],
  auth: { required: false },
  queue: { size: 0, paused: false, running: null },
  user_assets: [],
}

const GB = 1024 ** 3

export const models: Models = {
  checkpoints: (
    [
      ["animagine-xl-4.0.safetensors [a1b2c3]", "xl", 6.46],
      ["noobai-xl-vpred-1.0.safetensors [d4e5f6]", "xl", 6.62],
      ["illustrious-xl-v2.safetensors [0718aa]", "xl", 6.46],
      ["anima-preview.safetensors [99c0de]", "anima", 3.9],
      ["flux1-dev-fp8.safetensors [5e1f00]", "flux", 11.1],
    ] as const
  ).map(([title, arch, size]) => ({ name: title.replace(/ \[.*\]$/, ""), title, arch, size: Math.round(size * GB) })),
  vae: [{ name: "Automatic" }, { name: "sdxl_vae.safetensors" }],
  text_encoders: [{ name: "clip_l.safetensors" }, { name: "t5xxl_fp8.safetensors" }],
  modules: [
    { name: "sdxl_vae.safetensors", size: Math.round(0.31 * GB) },
    { name: "ae.safetensors", size: Math.round(0.31 * GB) },
    { name: "qwen_image_vae.safetensors", size: Math.round(0.24 * GB) },
    { name: "clip_l.safetensors", size: Math.round(0.23 * GB) },
    { name: "t5xxl_fp8_e4m3fn.safetensors", size: Math.round(4.6 * GB) },
    { name: "qwen_3_06b_base.safetensors", size: Math.round(1.1 * GB) },
  ],
  unet_dtypes: [
    "Automatic",
    "Automatic (fp16 LoRA)",
    "float8-e4m3fn",
    "float8-e4m3fn (fp16 LoRA)",
    "float8-e5m2",
    "float8-e5m2 (fp16 LoRA)",
    "float8-e4m3fn (Anima scaled)",
    "int8 (Anima ConvRot)",
  ],
  presets: ["sd", "xl", "flux", "klein", "qwen", "lumina", "zit", "wan", "anima", "ernie", "pid", "krea", "qwen21"],
  samplers: SAMPLERS.map((s) => ({ name: String(s.value), aliases: [] })),
  schedulers: SCHEDULERS.map((s) => ({ name: String(s.value) })),
  upscalers: [{ name: "Latent" }, { name: "R-ESRGAN 4x+" }],
  styles: [],
}

/** styles.csv in mock mode. */
export const styles: PromptStyle[] = [
  { name: "Anime quality", prompt: "masterpiece, best quality, {prompt}, very aesthetic", negative_prompt: "lowres, worst quality, bad anatomy" },
  { name: "Watercolor", prompt: "watercolor (medium), soft colors", negative_prompt: "photorealistic" },
  { name: "Cinematic", prompt: "cinematic lighting, depth of field, film grain", negative_prompt: "" },
]

/** A sample of the server catalog for zh_CN (schema labels). */
export const serverCatalogZh: Record<string, string> = {
  Prompt: "提示词",
  "Prompt\n({{generate}} to Generate ; {{skip}} to Skip ; {{interrupt}} to Interrupt)": "提示词\n（{{generate}} 生成；{{skip}} 跳过；{{interrupt}} 中断）",
  "Negative prompt": "反向提示词",
  "Sampling method": "采样方法",
  "Schedule type": "调度类型",
  "Sampling steps": "迭代步数",
  Width: "宽度",
  Height: "高度",
  "Batch count": "批次数",
  "Batch size": "单批数量",
  "CFG Scale": "提示词引导系数 (CFG)",
  Seed: "随机数种子",
  "Hires. fix": "高分辨率修复",
  Upscaler: "放大算法",
  "Hires steps": "高分迭代步数",
  "Denoising strength": "重绘幅度",
  "Upscale by": "放大倍数",
  Script: "脚本",
  None: "无",
  "Enable Anima": "启用 Anima",
  Preset: "预设",
  Balanced: "均衡",
  Detail: "细节",
  Speed: "速度",
  "Variation seed": "差异随机种子",
  Extra: "更多",
  Refiner: "精炼器",
  Checkpoint: "模型",
  "Switch at": "切换时机",
  "Variation strength": "差异强度",
  Swap: "交换",
}
