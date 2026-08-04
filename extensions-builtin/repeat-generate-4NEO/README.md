<p align="center">
🌍
<strong>English</strong> |
<a href="./README_JP.md">日本語</a> |
</p>

---

<div align="center">

# Repeat Generate for NEO

---

</div>

Repeat image generation automatically in [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic). (It can potentially work with other webUIs, but I am not promising to maintain them for anything other than Forge Neo)

Unlike “Generate forever”, minimizing the web browser does not cause any issues.

This is a lightweight Forge extension that continuously requeues image generation for:

* txt2img

* img2img

It tracks generation completion using Forge/Gradio queue APIs and automatically starts the next generation when the current task finishes.

## 💕 Features

* Automatic repeated generation

* Queue-based execution (Forge native)

* Continues to run even if minimizing the web browser

* Supports both txt2img and img2img

* One button operation to toggle repeat START and STOP

* “Interrupt” button to stop immediately

## 🖥️ UI

* ↻ Start Repeat / ■ Stop Repeat / ⌛ Stopping: Transitions between the following three states.

  * ↻ Start Repeat: Start repeated generation.<BR>
    ![Start](/images/start.webp)

  * ■ Stop Repeat: Stop repeated generation. (It'll wait until the current generation is complete.)<BR>
    ![Stop](/images/stop.webp)

  * ⌛ Stopping: Waiting until the current generation is complete. (Click to re-enable repeated generation.)<BR>
    ![Stopping](/images/stopping.webp)

* Interrupt: Stop repeated generation & interrupt current generation immediately. (Sends interrupt.)

* Status: Shows current state.

## ⚠️ Known Limitations

Continue generating using the same parameters as at the start.

* Even if ControlNet is enabled, it will be ignored. There may be other settings that are also ignored.
* When image generation is complete, the first image will be temporarily displayed in the preview.
* Changing parameters during repeated generation is not being accepted.
* If you want to change the parameters, stop repeated generation and then start.
* You don't need to wait until “↻ Start Repeat”, click while “⌛ Stopping”.

## 🛠️ Installation

1. Open your Stable Diffusion WebUI Forge - Neo.
2. Navigate to the Extensions tab → Install from URL.
3. Paste: <https://github.com/hirorohi03/repeat-generate-4NEO.git>
4. Click Install and restart the WebUI.

## 🚗 How It Works

The extension:

1. Hooks Forge submit functions
2. Captures generation arguments
3. Sends requests directly to:
   /queue/join
4. Monitors completion using:
   requestProgress()
5. Automatically submits the next generation

This avoids timer polling and works with Forge's native queue system.

## 🔍 Internal States

The extension internally separates:

| State     | Meaning                                |
| --------- | -------------------------------------- |
| `running` | Repeat mode enabled                    |
| `busy`    | A generation task is currently running |

This allows correct behavior such as:

* stopping after current task

* preventing double queue submission

* proper UI status display

## ⚖️ License

This repository is based on a fork of an upstream repository that did not specify a license.

The developer **Mr. akirau-ai** ([@akirau-ai](https://github.com/akirau-ai)) of the upstream repository has stated on [note.com](https://note.com/akirau338/n/n3fbfeaa6184d?scrollpos=comment) that he does not design no specific license has been and that anyone is free to use.

My original modifications in this repository are released under the MIT License.
