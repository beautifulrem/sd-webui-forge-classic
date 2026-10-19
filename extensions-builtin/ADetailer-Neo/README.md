# ADetailer Neo
The good ol' [ADetailer](https://github.com/Bing-su/adetailer) extension, which performs automatic masking and inpainting, completely rewritten for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), now ***lighter, faster, and prettier*** :tm:

- **Download ZIP:** `68 KB` &rarr; `38 KB`

> [!Caution]
> If you have installed **any** other version of `ADetailer` before, please review the **Settings** and **UI Defaults** related to `Adetailer`, as some older values are incompatible<br>
> *(or just delete `config.json` and `ui-config.json` to reset **everything**)*

## Features
After a generation, use **YOLO** or **MediaPipe** model(s) to detect certain regions *(**e.g.** face)* to perform **Inpainting**, improving fine details automatically

> [!Note]
> Not taking feature requests ; this repo is only meant as the "official" fork that is guaranteed to work with **Forge Neo** ; if you need other features, consider [this fork](https://github.com/abzaloff/aadetailer-neoforge) instead

## Models
The following **10** detector models are listed from the start; each one is downloaded into the `models/adetailer/` folder the first time a generation uses it (nothing is downloaded at launch). Launch with `--ad-no-huggingface` to turn these downloads off: then only the models already in the folders are listed

<table>
	<tr>
		<th>Filename</th>
		<th>Source</th>
	</tr>
	<tr>
		<td>face_yolov8n.pt</td>
		<td rowspan="6">
			<a href="https://huggingface.co/Bingsu/adetailer">Bingsu/adetailer</a>
		</td>
	</tr>
	<tr>
		<td>face_yolov8s.pt</td>
	</tr>
	<tr>
		<td>hand_yolov8n.pt</td>
	</tr>
	<tr>
		<td>hand_yolov8s.pt</td>
	</tr>
	<tr>
		<td>person_yolov8n-seg.pt</td>
	</tr>
	<tr>
		<td>person_yolov8s-seg.pt</td>
	</tr>
	<tr>
		<td>yolov8x-worldv2.pt</td>
		<td>
			<a href="https://github.com/ultralytics/assets">ultralytics/assets</a>
		</td>
	</tr>
	<tr>
		<td>mediapipe_face_short.tflite</td>
		<td rowspan="2">
			<a href="https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector">mediapipe/face_detector</a>
		</td>
	</tr>
	<tr>
		<td>mediapipe_face_full.tflite</td>
	</tr>
	<tr>
		<td>face_landmarker.task</td>
		<td>
			<a href="https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker">mediapipe/face_landmarker</a>
		</td>
	</tr>
</table>

> [!Tip]
> You can put additional [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO models into the `models/adetailer/` folder

> [!Important]
> The **YOLO** models must be in `.pt` format

The **Refresh models** button next to a tab's detector rescans the model folders (without downloading)

The `/adetailer/v1/ad_model` list also contains the stock models that are not downloaded yet

## API
- **Remi WebUI** (`/api/v2`): named params `ad_enable`, `ad_skip_img2img`, then each tab's [`ADetailerArgs`](lib_adetailer/args.py) fields, suffixed from the 2nd tab on (`ad_model`, `ad_model_2nd`, `ad_model_3rd`, ...), then one UI-only flag per tab (`ad_model_is_world`, `ad_model_is_world_2nd`, ...; shows "Detector Classes", never processed)
- **A1111** (`/sdapi/v1`): unchanged, one dict per model in `alwayson_scripts`, e.g. `{"ADetailer": {"args": [true, false, {"ad_model": "face_yolov8n.pt"}]}}`
  - `/sdapi/v1/script-info` now describes the form's scalar params in that order (`2 + 40 × tabs`, labels without the tab ordinal) instead of `[enable, skip, dict per tab]`; sending dicts works as before
- `GET /adetailer/v1/version`, `/adetailer/v1/schema` (the dict's JSON schema), `/adetailer/v1/ad_model`

<br>
<hr>
<br>

<pre align="center">
Copyright (C) 2026 Bing-su
Copyright (C) 2026 Haoming02

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
</pre>
