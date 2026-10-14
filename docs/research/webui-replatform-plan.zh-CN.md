# Forge Neo Remi：替换 Gradio 的最终方案

*(2026-10-14，分支 `remi`)*

---

## 1. 结论摘要

建议不升级到 Gradio 5/6，也不改用 Python 原生 UI 框架（NiceGUI、Reflex、Streamlit），而是做下面这件事：

- 在 Python 端定义参数 Schema（`ParamSpec`），由现有的 `SCRIPT_ARGS` 表直接扩展得到。
- 由一个预构建的 React SPA 统一渲染这些 Schema。
- 新 UI 与现有 FastAPI 进程放在一起，挂在 `/app`，并配一套 v2 API（具名参数、SQLite 队列、WebSocket 二进制预览）。
- Gradio 4.40 冻结，作为 `/legacy`（经典 UI）继续存在。它承接第三方扩展，并让上游 Forge Neo 的合并成本不变。
- 最后一个阶段用 Remi 自有的 `gr` 外观模块（facade）把 Gradio 降为"可选安装"。这样默认运行时不再加载 Gradio，上游文件也不需要改。

这个方案主体来自评审胜出的 "python-native" 路线，同时吸收了 strangler 路线的兼容报告、安全清单和 opaque_args 导入器，以及 rewrite 路线的 schema_version 迁移、CodeMirror 吸收 tagcomplete、模块边界测试。评审指出的事实错误和致命缺陷都已修正，见第 9 节。

**最重要的一点：P0 和 P1 本身就有价值，是保底交付。** P0 立刻缓解 #1302 的输入卡顿；P1 交付 v2 API 和实时预览。即使后面的 SPA 停工，这两步的收益也保得住。

### 推荐技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 前端框架 | React 19.3 + React Compiler 1.0，TypeScript strict | 生态最大，AI 辅助写代码最强；与 SD.Next Enso、InvokeAI 同构 |
| 构建 | Vite 8（Rolldown）+ pnpm；CI 产出 `dist.zip` | 终端用户不需要安装 Node |
| 组件 / 样式 | Tailwind 4 + shadcn/ui（Base UI 底座，组件代码拷入仓库） | 代码归我们所有，避免被 PrimeVue/MUI 锁死 |
| 状态 | Zustand 5（每个 tab 的参数草稿、画布）+ TanStack Query 5（服务端数据） | InvokeAI webv2 已弃用 Redux |
| 路由 / 命令面板 | TanStack Router（代码路由，无 SSR）+ cmdk | |
| 虚拟化 | TanStack Virtual（`lanes` 实现瀑布流） | 用于图库和 LoRA 浏览器 |
| 画布 | Konva + react-konva，统一走 `canvas/api.ts` 接口层 | 以后可以整体换成 PixiJS |
| 提示词编辑器 | CodeMirror 6 | 吸收 edit-attention、tagcomplete、bracket-checker 的功能 |
| API 客户端 | `@hey-api/openapi-ts`（精确锁版本；它仍是 0.x）+ CI `codegen:check` | 备选方案：openapi-typescript + openapi-fetch |
| i18n | react-i18next；Schema 标签本身支持 `{en, zh-CN}` | |
| 后端 | 现有 FastAPI 0.127.1 / Pydantic 2.10.6、标准库 `sqlite3`（WAL、FTS5）、Starlette WebSocket、Pillow | 不引入任何新的 Python 服务 |
| 测试 | pytest、Vitest、Playwright + CDP、size-limit、axe | 性能预算不达标则 CI 失败 |

**版本核对（2026-10-15，查询 npm registry、PyPI、GitHub API 和 shadcn 官网）：** 下列说法均已确认。

- 前端：react / react-dom 19.3.1；vite 8.3.4；babel-plugin-react-compiler 1.0.0；@hey-api/openapi-ts 0.99.0，仍是 0.x，需要精确锁版本。
- 样式与状态：tailwindcss 4.3.3；zustand 5.0.15；@tanstack/react-query 5.104.2、react-router 1.170.44、react-virtual 3.14.19。
- 画布与编辑器：konva 10.8.0；react-konva 19.3.0；@codemirror/view 6.43.14；备选 openapi-typescript 7.13.0。
- shadcn/ui 官网 changelog「July 2026 - Base UI as the Default」确认已默认改用 Base UI。
- Python 侧（对照用）：gradio 6.30.0，nicegui 3.18.0，reflex 0.10.0。fastapi 最新 0.143.1，仓库当前锁定 0.127.1，本方案不要求升级。
- Enso：Apache-2.0，1 star，2026-03 创建，仍在活跃提交。ComfyUI_frontend 为 GPL-3.0，约 2.1k star。

---

## 2. 现状诊断（Gradio 耦合面与痛点）

### 耦合面

**版本与引用规模**
- Gradio 版本锁在 `modules/launch_utils.py:306`（`gradio==4.40.0 gradio_rangeslider==0.0.8`），启动时安装（l.422-423）。
- 核心有 43 个文件、743 处 `gr.*`；扩展有 56 个文件、1,214 处 `gr.*`。

**事件绑定**
- 约 155 个 `.click`、约 147 个 `.change` 类事件、约 70 个 `_js` 钩子、约 600 个 `elem_id`。

**模块级 `import gradio as gr`（已用 grep 核实）**
- `modules/shared.py:5`、`options.py:5`、`shared_options.py:3`、`scripts.py:8`、`infotext_utils.py:10`。
- 所以哪怕是 `--nowebui`，也必然加载 gradio（包括 `gradio.blocks`）。

**参数位置由 UI 构建决定**
- `scripts.py:688-731` 中的 `create_script_ui_inner` 按 `ui()` 返回控件的顺序生成 `args_from`/`args_to` 和 `api_info`。
- `api.py:266-267` 在 runner 为空时调用完整的 `ui.create_ui()`。

**infotext 注册表是 UI 构建的副作用**
- `Api.apply_infotext`（`api.py:369` 起）读取 `infotext_utils.paste_fields[tabname]["fields"]`。
- `anima_support.register_table_fields`（l.800）在 `ui()` 中顺带注册 preset 和 paste 字段。

**设置项绑定 Gradio 类**
- 423 个 `OptionInfo` 各自声明一个 Gradio 组件类。

**JS 依赖 DOM 结构**
- 279 次 `gradioApp()`、471 次 `querySelector`、29 次 `updateInput`。
- 12 个 `MutationObserver`。其中 `script.js:121-136` 是一个全树级观察者，每次 DOM 变化都会跑一遍全部 `onUiUpdate`。

### 性能痛点

- 估计有 3k–5k 个 Gradio 组件挂在同一个 SPA DOM 里，隐藏 tab 也被挂载；`/config` JSON 体积庞大。
- 每个 `ui()` 构建两次（txt2img 和 img2img）；"Reload UI" 会整树重建。
- 页面注入 62 个未打包的 `<script>`，约 2.43 MB。其中 agent-scheduler 1.33 MB、prompt-all-in-one 708 KB。CSS 首屏约 380 KB，之后再懒加载约 400 KB。
- 进度每 500 ms 轮询一次 `POST /internal/progress`，预览以 base64 塞在 JSON 里（约 33% 膨胀），并在请求线程里重新编码。
- token 计数走 `wrap_queued_call`，会排在生成任务后面。上游 issue #1302 "Extremely lagged prompt typing"，Haoming02 本人回复 "Gradio being Gradio"。

### 美观痛点

- 整体是 Gradio 默认风格，再叠加 46 KB 手写 `style.css` 和各扩展 CSS 的补丁，视觉不统一。
- 没有设计 token，没有密度控制。
- 中英双语靠 DOM 文本节点遍历来实现。

### 已有的有利接缝

- `processing.py` 管线本身不依赖 Gradio。
- 冲突解析器是纯 Python（`anima_feature_conflicts.py:138-307`）。
- `parse_generation_parameters` 是纯函数。
- Anima 扩展已经有声明式的 `SCRIPT_ARGS` 表（guidance 51 行、FreeFuse 15 行、Regional 7 行），以及 `script_arguments()` 和 `PasteKey`。
- `webui.py` 的 `api_only_worker` 已经会调用 `app_started_callback(None, app)`。

---

## 3. 备选方案对比

评分为 1-5 分，越高越好。

| 路线 | 工作量 | 美观 | 性能上限 | 扩展兼容 | 上游同步成本 | 日常开发仍以 Python 为主 | 结论 |
|---|---|---|---|---|---|---|---|
| A. 升级 Gradio 6.30 并换主题 | 2 | 2 | 3 | 1 | 1（上游停在 4.40） | 5 | 不选：所有扩展第三次破坏性迁移，组件树成本不变，外观仍是 Gradio |
| B. NiceGUI 3.18 作为主 UI | 3 | 3 | 2（每个事件都要 socket 往返，大页面卡顿） | 2 | 4 | 4 | 不选：画布和图库仍需手写 Vue |
| C. Reflex / Streamlit / Flet | 2 | 3-4 | 1 | 1 | 4 | 3 | 不选：每个事件约 40 ms 且串行，或整页重跑，或 Flutter-web 不是 DOM |
| D1 "strangler"：渐进替换 + 最终退役 Gradio | 3 | 4 | 4 | 4 | 3 | 3 | 评审 42 分。扩展和安全设计最完整，但"无 Gradio 启动"的承诺与"只改极少上游文件"自相矛盾 |
| D2 "rewrite"：干净重写，放弃第三方扩展 | 1-2 | 5 | 5 | 2 | 4 | 2 | 评审 37 分。会让 `/sdapi/v1` 的第三方脚本槽位错位，冲突规则在前端写第二份，单人很可能半途停滞 |
| **D3 "python-native"：Python Schema + 预构建 SPA（选用并合并）** | 3 | 5 | 5 | 3-4 | 5（新增目录为主） | 4 | 评审 43 分，胜出。P0/P1 可独立交付，以后新增 Remi 功能只写 Python |

**关于原方案里的 NiceGUI 决策闸门：** 由一周 spike 改为 2 天的"纸面核查"。用 P0 的实测基线核对 #4434/#2876 中列出的 DOM 和往返成本，不通过就直接定 React，避免拖延工具链决策。

---

## 4. 目标架构

```
浏览器
 ├─ /app  预构建 SPA（extensions-builtin/remi-webui/dist，按路由懒加载）
 │   ├─ shell：布局（参数 | 舞台 | 图库/队列 三栏可调）、cmdk、主题/密度、i18n
 │   ├─ ParamRenderer(schema)：约 15 种控件（滑块+数值、seed、带刷新的下拉、
 │   │    开关、单选、CodeMirror 提示词、LoRA 选择、图像输入（上传暂存）、多选、颜色、高级 JSON）
 │   ├─ 路由：generate · edit(react-konva) · extras · png-info · library ·
 │   │    gallery · queue · settings · ext/<id>
 │   ├─ stores：Zustand（按参数路径存草稿，支持撤销/重做）+ TanStack Query
 │   ├─ realtime：单条 WS（JSON 事件 + 二进制预览帧）
 │   └─ plugin host：import map 共享 react/@remi/sdk/@remi/ui → import(扩展 ESM)
 │        插槽：txt2img.prompt.after / generate.toolbar / gallery.actions / sidebar / tab
 └─ /（默认，Gradio 4.40 经典 UI，冻结）
     可通过选项 remi_default_ui 让 / 重定向到 /app，经典 UI 走 /?classic=1

FastAPI 进程（webui.py → demo.launch 不变 → app）
 ├─ /sdapi/v1/*、/internal/progress、扩展路由（保持不变，作兼容）
 ├─ /remi/api/v2/  （由 remi-webui 扩展在 on_app_started(demo|None, app) 中挂载，
 │   │               --nowebui 下同样可用）
 │   ├─ GET  schema?tab=           核心参数 + 各脚本 ScriptSchema（JSON Schema + x-ui）
 │   ├─ POST rules/derive          {changed, values} → {patch, locked:{id:reason}, notes}
 │   ├─ POST jobs / GET jobs / DELETE jobs/{id} / POST jobs/{id}/interrupt|skip
 │   ├─ POST uploads → upload_id；GET files/{id}、thumbs/{id}?w=（ETag，WebP）
 │   ├─ GET  library/{kind}?q=&cursor=；POST library/rescan
 │   ├─ GET/PATCH options；GET presets/{name}
 │   ├─ POST infotext/parse|apply → 具名参数 patch
 │   └─ GET  i18n/{locale}
 ├─ WS /remi/ws  （鉴权 + Origin 校验；feature_flags 握手）
 └─ 执行器：JobQueue(SQLite) → 具名参数 → ArgLayout.to_positional()
        → 投递到 modules_forge/main_thread，持有 call_queue.queue_lock
        → modules.txt2img/img2img → processing.py（不改动）

modules_remi/  （Remi 自有、纯 Python，模块级不 import gradio）
 ├─ ui_schema/ spec.py · layout.py(ArgLayout) · from_table.py · introspect.py ·
 │             gradio_render.py · options.py · registry.py(paste/preset 注册表)
 ├─ server/    v2 路由、ProgressHub、预览编码线程、鉴权依赖
 ├─ library/   networks.db / outputs.db（FTS5）
 ├─ jobs/      queue.db、schema_version 迁移、agent-scheduler 导入器
 └─ grshim/    （P6）gradio 外观模块与 meta_path finder
```

**上游文件改动白名单。** 下面这些点逐一写进 `docs/remi-hook-points.md`，每次同步都要核对：

1. `scripts.py` 的 `create_script_ui_inner` 增加一个 `PARAMS` 分支，约 30 行。
2. `api.py:266` 的 `create_ui()` 改为"先尝试 schema 构建，失败再回退"，约 10 行。
3. `launch.py`/`webui.py` 早期挂 grshim finder，P6 才做，约 5 行。
4. 可选：`script.js:121-136` 的 rAF 批处理。

上游的 `ui*.py`、`shared_options.py`、`infotext_utils.py` 都不动。

---

## 5. 关键子系统设计

### 5.1 script_args 与 infotext

**ParamSpec 字段**

`id`、`type`、`label`（`str | {en, zh-CN}`）、`help`、`default`、`min`/`max`/`step`、`choices | choices_source`、`group`、`widget`、`visible_if`、`infotext_key`、`paste`（可调用）、`preset_key`、`exclusive_group`。

- Anima 扩展的写法：`PARAMS = from_table(SCRIPT_ARGS, overrides=…)`，`ui()` 改为 `return gradio_render.build(self, PARAMS)`。
- `gradio_render` 内部继续调用 `register_table_fields`，所以经典 UI 的行为、控件顺序和 `tests/test_anima_presets.py` 都保持不变（测试最多小改）。

**ArgLayout 是唯一的槽位来源**

- Schema 原生脚本贡献 `len(PARAMS)` 个槽，不需要构建 Gradio。
- 传统脚本用 introspect 得到槽位：在一个不 launch 的 `gr.Blocks()` 里只跑该脚本的 `ui()`，读取控件类型、label、min/max/choices、visible，以及是否存在 `.change`/`.then`。
- 无法映射的控件成为 opaque 槽：保留默认值、不渲染，前端显示徽章。注意这不是"`--nowebui` 现在就这样做"，现状是 `create_ui()` 会构建完整树，这是新增的轻量路径。
- 黄金测试：对每个内置脚本、两个 tab，断言 `ArgLayout` 与 Gradio 构建出的 `self.inputs` 长度和顺序完全一致。
- 新增"上游核心输入 diff 测试"：比较 `ui.py:340-384` 的固定 txt2img/img2img 输入列表与 v2 `CoreParams`。上游新增参数时测试会失败，并给出提示。

**修复 `apply_infotext` 依赖 UI 副作用的问题**

- `modules_remi/ui_schema/registry.py` 在不构建 UI 的情况下，从 PARAMS 和 introspect 结果重建 `paste_fields[tab]["fields"]` 与 preset 注册表。
- 写法是"填充上游已有的字典"，不改上游代码。
- 在 `api.py` 的白名单补丁里，先跑 registry 再判断是否需要 `create_ui()`。
- 契约测试：`/sdapi/v1/txt2img` 携带 `infotext`（含 `Anima …` 键和 `Anima conflict resolution`）时，结果必须与改动前逐字节一致。

**infotext 统一入口**

- `POST infotext/apply` = `parse_generation_parameters` + 各 spec 的 `paste` 函数（即现在的 `PasteKey` 函数，原样复用），返回具名 patch。
- UI 粘贴、"Send to"、PNG Info、图库 remix、API 回放都走这一条路。SPA 不解析 infotext。

### 5.2 设置

- `ui_schema/options.py` 把 423 个 `OptionInfo` 映射为 ParamSpec：按 `component` 类名加 `component_args` 推断控件，可调用参数在每次请求时于服务端求值。
- `PATCH options` 走 `opts.set`，`onchange`（例如模型重载）照常在服务端触发，进度通过 WS 推送。
- quicksettings 是用户可钉选的子集；设置页由前端搜索并虚拟化。
- `ui-config.json` 由一次性导入器按"tab/label → spec id"映射到 `user_defaults.json`（具名路径），对不上的项记录日志。

### 5.3 额外网络

- 复用各 `ExtraNetworksPage` 的条目枚举逻辑，把 name、alias、tags、sd_version、path、mtime、sha、preview 写入 `networks.db`（FTS5）。
- 刷新时机：按需、`refresh-*`、以及后台按 mtime 增量扫描。
- `GET library/{kind}?q=&tags=&cursor=` 分页，前端 TanStack Virtual 只挂约 30 张卡片。anapnoe 的实测：监听器减少约 50%，堆内存减少约 43%。
- 缩略图复用 `/sd_extra_networks/thumb`，另生成一次 WebP 缓存。
- 插入 LoRA 用 CodeMirror transaction 写入 `<lora:name:w>`。Anima LoRA 工具（layer-weight、lora_schedule、`anima_lora_ui.py`）合成一个详情抽屉。

### 5.4 进度与实时预览

- 在 `progress.py` 的任务注册表（`record_progress`、`current_task`）和 `shared.state` 上挂进程内事件总线，由 `/remi/ws` 扇出。
- JSON 事件：`job.queued|started|progress{step,steps,eta}|completed{file_ids,infotext}|failed`、`queue.changed`、`options.changed`。
- 二进制预览帧沿用 ComfyUI 风格：`[u8 type][u8 fmt][u32 job_seq][u32 step][u64 server_ts]` 后接 WebP/JPEG 字节。
- 编码放在独立线程，只保留最新帧、旧帧丢弃，不占用采样线程。每个客户端限帧，默认 8 fps，可在 4–15 之间调整。
- `feature_flags` 握手协商格式。中断和跳过同时提供 REST 与 WS 指令。
- `/internal/progress` 保留，由同一注册表供数，并能看到 v2 任务。

### 5.5 画布与局部重绘

**图层**
- 底图、蒙版、草图、Regional 蒙版（直接绑定 `sd_forge_anima_regional` 的参数）、ControlNet 输入。
- 每个蒙版图层画在一块离屏 canvas 上（`globalCompositeOperation`），不要每一笔建一个 `Line` 节点。

**工具**
- 画笔、橡皮、填充、套索、缩放、平移、撤销。
- 蒙版模糊和 padding、soft-inpainting 都作为普通 spec。

**上传**
- 只上传变化过的图层（PNG blob，按 hash 缓存）到 `/uploads`，参数里引用 `upload:<id>`。
- 新 UI 不再使用 `LogicalImage(gr.Textbox)` 传 base64；`forge_canvas` 仅留给经典 UI。

所有画布调用都必须经过 `canvas/api.ts`，并有模块边界测试守住。

### 5.6 图库

- `on_image_saved` 回调加后台扫描，增量写入 `outputs.db`：path、w、h、seed、infotext、job_id、created。
- `GET outputs?cursor=` 与 `files/{id}` 只按 id 查找，不接受原始路径。
- 瀑布流的行高用存储的宽高精确计算。附带灯箱（缩放/平移）、对比滑块、infotext 面板、remix、"送到编辑"、星标、删除和 `<video>` 卡片。

### 5.7 i18n

- 界面固定文案放在 `locales/{en,zh-CN}.json`。Schema 标签直接携带双语，`anima_lora_ui.py` 的 `intro_block`/`register_language_option` 文案用脚本一次性抽成单一数据源，经典 UI 通过小型 loader 继续读同一份。
- 第三方 introspect 来的标签，用 `localizations/*.json`（A1111 格式，以英文文本为键）做回退查找。
- 切换语言即时生效，不需要重载。CI 对 schema 原生扩展检查缺失的翻译键，并测试中文标签宽度下的布局。
- `sd-webui-language-diffusion` 的提示改为 zh-CN 资源包。

### 5.8 队列（取代 agent-scheduler）

- `queue.db`：id、tab、`schema_version`、具名参数 JSON、`layout_hash`、uploads、priority、status、结果 file_ids。
- 功能：暂停、重排、重试、重复 N 次、收藏、历史、空闲时运行。吸收 agent-scheduler、prompt-queue、repeat-generate 的功能。
- **只有一个 worker，只把任务投递到 `modules_forge/main_thread` 并持有 `queue_lock`**，任何地方都不在别的线程碰 torch。经典 UI 的提交和 v2 任务天然串行。
- agent-scheduler 导入器：用 `task_helpers.map_*_args` 反向加上导入时的 `ArgLayout` 快照，把位置参数转成具名参数。对不上的尾部参数放进 `opaque_args`，**拒绝重放**而不是猜测。报告写日志。
- 迁移完成前，agent-scheduler 在经典 UI 中照常工作，它的 34 个路由保留到 P6。

### 5.9 Anima 冲突锁与预设

- `anima_feature_conflicts` 的解析器仍是唯一权威，处理阶段的语义不变：API 保留主选项，丢弃冲突的辅助开关。
- Schema 只发布 `exclusive_group` 元数据，用于前端乐观翻转（最后触碰者生效）。随后以 80–150 ms 防抖调用 `rules/derive`，由服务端确认并返回锁定原因。这个端点**不排在 `queue_lock` 后面**。前端**不另写一套规则语言**（即删去 rewrite 方案的 `disabled_when` DSL）。
- `main_entry.on_preset_change` 里构造约 25 个 update 的字典逻辑抽成纯函数，`GET presets/{name}` 与经典 UI 共用。
- 一致性测试：每个解析器用例同时跑经典 `.change` 路径和 `rules/derive`，结果必须相同。
- 前端为每个控件显示"偏离默认/预设"小圆点和逐项重置按钮；被锁定的控件显示锁图标和原因。

---

## 6. 扩展策略

### Remi 内置扩展迁移顺序（按日常价值排列）

1. **P1：** `sd_forge_anima_guidance`（51 行）、`_regional`（7 行），以及 `processing_scripts/*`（seed、sampler、refiner、mahiro、rescale_cfg）。
2. **P2：** `_freefuse`、`_hires_guard`、`_presets`、AnimaPatchPanel 组（dave、xattn_boost）、prompt-rescale、dynamic-shift、resolutions、sd-forge-negpip、sd_forge_spectrum、detail-daemon。
3. **P3-P4：** 自定义控件类。`anima-lora-stage-scheduler`（阶段时间线）、`anima-lora-layer-weight`（块权重网格）、`anima-artist-scheduled-mixer`（混合表）、soft-inpainting、image_stitch、postprocess-suite。
4. **P5：** sd_forge_controlnet（单元数组 spec、预处理器预览端点、openpose 以 `mount(el)` 插件形式接入；`forge_legacy_preprocessors` 无 UI，不动），ADetailer-Neo（单元数组），XYZ plot。
5. **功能吸收：**
   - tagcomplete 改成 CodeMirror 补全源，复用 `/tacapi/v1`。
   - bracket-checker 改成 lint，edit-attention 改成快捷键。
   - agent-scheduler、prompt-queue、repeat-generate 并入核心队列。
   - prompt-all-in-one 不移植它的 Vue 包，常用的翻译、收藏、历史功能做成核心小功能。
   - mobile 由响应式布局取代。
6. **只留在经典 UI，按需再做：** joycaption、prompt-workshop、prompt-anchor（以后可做成 `render(el)` 插件）。

每个已有 `SCRIPT_ARGS` 表的扩展，在 AI 辅助下约需 0.3–1 人周。

### 第三方 A1111/Forge 扩展：分级与淘汰节奏

| 级别 | 条件 | `/app` 中的表现 |
|---|---|---|
| native | 声明了 `PARAMS` | 完整支持 |
| captured | 纯标准控件、无 JS | 自动渲染 |
| partial | 有 `.change` 链或部分 opaque 槽 | 渲染可映射部分并显示徽章；作者补一个 `ui_spec()` 即可升级 |
| legacy-only | 有 DOM/JS 钩子、`on_after_component` 注入或 `on_ui_tabs` | 显示"在经典 UI 打开"链接 |

- `/app/extensions` 是兼容性报告页，把每个扩展的级别都列出来，不让破坏悄悄发生。
- `/sdapi/v1` 的 `alwayson_scripts` 与 `script-info` 必须覆盖所有级别，包括 introspect 来的 legacy 脚本。这样外部客户端（Krita、脚本）的槽位不会漂移，也就修正了 rewrite 方案的那个矛盾。
- `on_before_component`/`on_after_component` 只在经典 UI 中生效；新 UI 用具名插槽。
- 发布 `docs/remi-webui-extension-guide.md`，内容覆盖 `PARAMS`、`rules`、`mount(el)` 和插槽，以及"`SCRIPT_ARGS` → PARAMS，`ui()` → schema，JS → ESM 插件"的移植指南。
- 淘汰节奏：
  - P5 前，经典 UI 是默认，`/app` 需主动打开。
  - P5 后，`/` 重定向到 `/app`，经典 UI 仍随时可进。
  - P6 后，Gradio 不再默认安装，`--classic-ui` 时才懒安装并挂载。
  - 第三方 legacy-only 扩展**永不强制移植**。

---

## 7. 分阶段路线图

单人加 AI 代理。"人周"指全职等价。

| 阶段 | 目标 | 退出标准 | 估算 |
|---|---|---|---|
| **P0 基线与经典 UI 速效优化** | Playwright + CDP 测量工具；在经典 UI 中把 `script.js:121-136` 改成 rAF 批量派发；agent-scheduler 和 prompt-all-in-one 的包延迟到首次打开时再加载；token 计数移出 `wrap_queued_call`；写 ADR；初始化 `webui-next/` 工作区和 CI（打包 dist.zip） | `docs/research/ui-baseline.md` 提交；经典 UI 输入提示词 INP < 100 ms；上游文件 diff < 50 行；`/app` 能提供空壳页面 | 2 |
| **P1 Schema 核心 + v2 API + WS（与渲染器无关）** | `modules_remi/ui_schema` 全套（含 registry 重建 paste_fields）；ArgLayout；introspect；options 映射；rules；JobQueue；uploads/files；`/remi/ws` 二进制预览；guidance 和 regional 改为 schema 原生；统一鉴权依赖 | 所有内置脚本两个 tab 的黄金布局测试全绿；`/remi/api/v2/jobs` 跑 guidance+FreeFuse 的 txt2img，infotext 与像素和经典 UI 逐字节一致；`/sdapi/v1` 带 infotext 的契约测试全绿；现有全部测试（约 1,659 个）通过；`--nowebui` 行为不变（不承诺不导入 Gradio） | 6 |
| **P2 `/app` 的 txt2img 日常可用** | 设计系统、三栏布局、ParamRenderer、CodeMirror 提示词、模型栏（checkpoint/VAE/TE/dtype/预设）、高分修复和 refiner、图库和灯箱、队列面板、粘贴与发送、zh-CN；完成 P2 那批 Anima 扩展 | 主人日常 Anima txt2img 全流程不用打开经典 UI；第 8 节的 txt2img 预算达标；冲突锁一致性测试全绿 | 7 |
| **P3 编辑视图** | Konva 画布（局部重绘、草图、Regional 蒙版）、上传暂存、img2img 各模式、extras、PNG Info、图像拼接 | 局部重绘和 Regional 的黄金图片与经典 UI 一致；4096² 画布在 M 系列 Mac 上涂抹保持 60 fps；任何请求中都不出现 base64 | 7 |
| **P4 资源库、设置、队列** | networks.db 与虚拟网格、设置页和 quicksettings、`ui-config.json` 导入、队列 UI、agent-scheduler 导入器、P3-P4 自定义控件扩展 | 5k 个 LoRA 搜索 < 50 ms、60 fps 滚动；423 个选项都能编辑且 `onchange` 触发；导入的任务重放结果相同 | 6 |
| **P5 长尾与默认切换** | ControlNet、ADetailer、XYZ、插件插槽、兼容报告、扩展指南；把 `/` 重定向到 `/app`（选项控制）；验证 `--subpath`/TLS/`--share`/`allowed_paths` | 连续两周日常使用不回经典 UI 做核心工作；每个内置扩展都已归入 native/captured/partial/legacy-only 之一 | 7 |
| **P6 Gradio 降为可选** | grshim（见第 9 节）；`launch_utils` 只在 `--classic-ui` 时安装 Gradio；`gradio_extensions.py`、`ui_gradio_extensions.py` 只在经典 UI 模式下加载 | 干净安装、不装 Gradio 也能启动并完成全部 A-C 级功能；`--classic-ui` 照常启动；grshim 导入测试覆盖上游所有 `gr.*` 名称 | 4 |

合计约 39 人周。**另加 25–50% 缓冲，按 45–60 人周规划，兼职约 10–14 个月。** ControlNet、ADetailer、画布和 Regional 加起来就可能占 10–15 人周。P0+P1（约 8 人周）是保底承诺。

---

## 8. 性能目标与度量方法

所有指标都对照 P0 基线测量。参考机器为主人的 Apple Silicon Mac 和一台 Windows/CUDA 机器，都用 Chrome stable，开启全部内置扩展。`tests/perf/` 在每个 PR 上运行，回退则 CI 失败。

| 指标 | `/app` 目标 | 测量方法 |
|---|---|---|
| 服务端 UI 构建 | schema 与布局 ≤ 300 ms（经典 UI 改为懒构建后，冷启动省下整棵 Blocks 树的时间） | `startup_timer` 子阶段、`python -X importtime` |
| 首屏 JS / CSS（壳 + generate，gzip） | ≤ 300 KB / ≤ 50 KB；每个路由分块 ≤ 150 KB；扩展面板懒加载 | size-limit、rollup visualizer |
| 冷/热加载（localhost） | LCP ≤ 1.0 s，TTI ≤ 1.2 s；热加载 ≤ 0.5 s | Playwright + web-vitals + CDP `Performance.getMetrics`，5 次取中位数 |
| Generate 视图 DOM 节点数 | ≤ 3,000；隐藏 tab 不挂载 | `document.querySelectorAll('*').length` |
| 输入延迟 | 300 token 提示词、开启补全时 INP p95 ≤ 50 ms；拖动滑块 0 次网络请求 | Event Timing API + 网络断言（#1302 场景） |
| 规则/计数往返 | p95 ≤ 30 ms，永不排在生成之后 | 服务端中间件计时 |
| 预览延迟 | p95 ≤ 150 ms，单帧 ≤ 60 KB，带宽比 base64 方案低 ≥ 30% | 帧头 `server_ts` 对比 rAF 绘制时间（先用 ping 校准时钟偏移）；WS 字节计数 |
| 对生成速度的影响 | 比关闭 UI 时慢 ≤ 1% it/s | 基准任务 A/B |
| 浏览器内存 | 浏览 500 张图 + 5k LoRA 后 JS 堆 ≤ 150 MB；1 小时浸泡增长 ≤ 10% | CDP HeapProfiler |
| 服务端开销 | WS 空闲时 CPU ≈ 0；RSS 比 `--nowebui` 多 ≤ 100 MB | psutil 采样 |
| 图库/网络 | 10k 张图 + 5k 个 LoRA 滚动 60 fps，挂载卡片 ≤ 60 | Playwright trace 帧时间 |

---

## 9. 风险与缓解，以及明确不做的事

### 已修正的评审问题

**"无 Gradio 启动"不可能只靠小补丁做到**

上游 5 个核心模块在模块级 `import gradio`，且 `OptionInfo` 引用 `gr.Slider` 等类。所以 P1 和 P5 都**不承诺** `--nowebui` 时不导入 Gradio。真正的退役放到 P6，做法是 grshim：

- 在 `modules_remi/grshim/` 中提供一个与 `gradio` 同名的外观包。它用纯数据类实现 `Slider/Radio/Dropdown/Checkbox/Number/Textbox/ColorPicker/HTML/Blocks/Row/Column/Group/Accordion/Tab/State/Image/update/skip/Request/Info/Warning` 等名称，只记录构造参数。
- 启动早期由 meta_path finder 注入，只在未安装真 Gradio 且未带 `--classic-ui` 时生效。
- 上游文件一行都不用改。每次同步时，CI 用 AST 扫描上游新增的 `gr.*` 名称，缺了就失败。
- 这是一个持续维护的成本：上游每增加一种控件，shim 就要补一个类。

**`apply_infotext` 依赖 UI 副作用**

由 `ui_schema/registry.py` 从 PARAMS 和 introspect 结果重建 `paste_fields` 与 preset 注册表，并有契约测试兜底。

**原文中的事实错误，已更正**

- 现状下 `--nowebui` 运行的是完整的 `ui.create_ui()`，不是轻量的 headless Blocks。轻量 introspect 是本方案新增的路径。
- Enso 的许可证是 **Apache-2.0**，不是 "MIT-style"。它与 AGPL-3.0 的 Forge 兼容，但复制代码时必须保留 NOTICE 和署名。
- Enso 只有 1 star，是单人维护的早期项目。我们借鉴它的**模式**（`on_app_started` 挂载、按 commit 下载 dist.zip、codegen:check），不要把它当作成熟度背书。

### 上游同步的矛盾与对策

上游 Forge Neo 会长期停留在 Gradio 4.40，并持续修改 `ui*.py`。对策如下：

- 新代码只放在 `modules_remi/`、`extensions-builtin/remi-webui/`、`webui-next/`。
- 上游补丁只限第 4 节的白名单。每次 `docs/upstream-sync-*.md` 都附带三项检查：
  - 黄金布局测试
  - 核心输入 diff 测试
  - grshim 名称扫描
- 上游对经典 UI 的修复会继续自动流入 `/legacy`。上游新增的参数会被 diff 测试捕获，再手动补进 CoreParams 或 spec。
- **结论：Gradio 的代码不会从仓库删除。** 删除会导致每次同步都冲突。它只从默认运行时和默认安装里退出。这正好符合"不再用 Gradio"的实际诉求：新 UI 完全不依赖它。

### 主要风险

| 风险 | 缓解 |
|---|---|
| 单人范围失控、倦怠（InvokeAI webv2 约 50.8 万行，还要开 fork） | P0+P1 作为保底；每个阶段都可用；P2 退出后经典 UI 不再加新功能；只把小粒度、有契约测试的任务交给 AI；长尾留在 legacy |
| 参数槽位漂移（`/sdapi`、存档队列、infotext） | ArgLayout 唯一来源；黄金测试；每个任务存 `layout_hash` 和 `schema_version`；保留 `script_arguments()` 默认值填充和 NaN/Inf 拒绝；v2 全部用具名参数 |
| 两套 UI 行为不一致 | 共用 spec、preset 纯函数和解析器；黄金图片测试和冲突一致性测试；经典 UI 冻结 |
| introspect 漏掉动态行为 | 标为 opaque 或 partial，从不猜测、从不静默丢值；兼容报告页 |
| 线程与 GPU | 队列 worker 只投递到 `main_thread`；预览编码线程用有界队列并丢帧 |
| 安全回退 | 统一鉴权依赖（`--api-auth`/`--gradio-auth`）；WS 校验 Origin 和 token；v2 写路由复用 `extension_route_guard` 语义；上传限制大小和类型；`files` 只按 id 查找，并有路径穿越测试；`--nowebui` 鉴权模式也要测；参照 `test_extension_route_guard.py` |
| 工具链变动（hey-api 0.x、Vite 大版本） | 精确锁版本并提交生成的客户端；准备 openapi-typescript 作为后路；每月统一升级一次 |
| 终端用户需要 Node | `remi-webui/install.py` 下载与 `git rev-parse HEAD` 匹配的 `dist.zip`，版本不符时告警；本地 pnpm 构建只作开发后备；`dist/` 不进 git 历史 |
| 维护者调试 TS 困难 | 渲染器做薄，逻辑放在 Python rules 里；Playwright e2e；模块边界测试（画布只能 import `canvas/api.ts`，插件只能 import `@remi/sdk`、`@remi/ui`） |

### 明确不做

- 不升级到 Gradio 5 或 6，经典 UI 永久锁定 4.40。
- 不用 NiceGUI、Reflex、Streamlit、Flet、Mesop 做主 UI，也不给 Gradio 换皮（Lobe Theme、ModernUI 那条路）。
- 不一次性替换，不开长期 fork，不改写上游 `ui*.py`。
- 新 API、队列和存储里不出现位置参数；SPA 不读取参数索引。
- 新 UI 不用 base64 传图，不靠轮询拿进度，蒙版不按笔画建 Konva 节点。
- 前端不复制冲突或预设逻辑，也不设计规则 DSL。
- 新扩展 API 不按 DOM id 查询节点，没有 `on_*_component` 的等价物。
- 不用 Next.js、SSR 或 Module Federation；不用 tldraw（需商业授权）；不用重型组件库；不用 RJSF 或 JSON Forms。
- 不移植 prompt-all-in-one 和 agent-scheduler 的前端包。
- 不改动 `processing.py`、采样器、Anima 数学，以及 `/sdapi/v1` 的响应模型。

---

## 10. 需要你拍板的决策点

1. **前端框架用 React 还是 Vue 3.5？** 推荐 React 19：有 Enso 和 InvokeAI 的先例，react-konva 更成熟，AI 辅助效果最好。如果你明显更能读懂 Vue 模板，就换 Vue 3.5 + reka-ui（ComfyUI 的路线），其余设计不变。
2. **第三方 A1111 扩展的承诺范围？** 推荐：只做 introspect 自动渲染、兼容报告和经典 UI 兜底，不主动移植。
3. **"去掉 Gradio"做到哪一步？** 推荐分两步：P5 时新 UI 成为默认、Gradio 仍安装；P6 用 grshim 把 Gradio 变成可选安装。仓库里的上游经典 UI 代码保留，以免同步冲突。
4. **prompt-all-in-one 和 agent-scheduler 是吸收进核心，还是作为插件保留？** 推荐吸收：核心队列加 CodeMirror 提示词工具，经典 UI 里原扩展继续可用。
5. **URL 布局？** 推荐保持 `demo.launch` 不变，`/app` 由 `on_app_started` 挂载，`/` 是否重定向由 `remi_default_ui` 选项控制。不推荐用 `gr.mount_gradio_app` 把经典 UI 重新挂到子路径，因为容易连带破坏 `/file=`、`ui_tempdir` 和 `root_path`。
6. **投入节奏与验收方式？** 推荐先只承诺 P0+P1（约 8 人周），完成后用 P1 的实测数据重新评估 P2 以后是否继续。暂不做桌面打包（Tauri/Electron）。

---

## 11. 参考资料

**Gradio**
- https://pypi.org/project/gradio/
- https://gradio.app/changelog
- https://gradio.app/main/guides/gradio-6-migration-guide
- https://huggingface.co/blog/gradio-5
- https://github.com/gradio-app/gradio/issues/4841
- https://github.com/gradio-app/gradio/issues/8570
- https://github.com/gradio-app/gradio/issues/12891
- https://github.com/gradio-app/gradio/issues/12943

**Forge / Neo / A1111**
- https://github.com/Haoming02/sd-webui-forge-classic
- https://github.com/Haoming02/sd-webui-forge-classic/issues/1302
- https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/801
- https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/853
- https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/888
- https://github.com/lllyasviel/stable-diffusion-webui-forge/issues/1088
- https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/10532
- https://github.com/Haoming02/sd-forge-couple/issues/149

**SD.Next / Enso**
- https://github.com/vladmandic/sdnext/discussions/4723
- https://github.com/vladmandic/sdnext/wiki/Enso
- https://github.com/CalamitousFelicitousness/enso
- https://github.com/vladmandic/sdnext/blob/master/installer.py

**ComfyUI**
- https://github.com/Comfy-Org/ComfyUI_frontend
- https://docs.comfy.org/custom-nodes/js/javascript_overview
- https://docs.comfy.org/custom-nodes/js/javascript_sidebar_tabs
- https://docs.comfy.org/custom-nodes/v3_migration
- https://github.com/comfyanonymous/ComfyUI/blob/master/script_examples/websockets_api_example.py
- https://blog.comfy.org/p/comfyui-node-2-0
- https://github.com/Comfy-Org/ComfyUI/issues/7025

**InvokeAI**
- https://invoke.ai/development/front-end/workflows/
- https://invoke.ai/releases/version/v5-0-0/
- https://github.com/invoke-ai/InvokeAI/pull/9613
- https://github.com/invoke-ai/InvokeAI-7/pull/375
- https://deepwiki.com/invoke-ai/InvokeAI/7-canvas-system

**SwarmUI**
- https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Making%20Extensions.md

**换皮与性能参考**
- https://github.com/anapnoe/stable-diffusion-webui-ux
- https://github.com/lobehub/sd-webui-lobe-theme/issues

**前端技术栈**
- https://react.dev/versions
- https://vite.dev/blog/announcing-vite8
- https://ui.shadcn.com/docs/changelog/2026-07-base-ui-default
- https://tanstack.com/virtual/latest/docs/api/virtualizer
- https://konvajs.org/docs/sandbox/Free_Drawing.html
- https://konvajs.org/docs/guides/best-canvas-library.html
- https://tldraw.dev/sdk-features/license-key
- https://dev.to/nyaomaru/which-openapi-codegen-should-you-choose-openapi-typescript-vs-hey-api-vs-orval-vs-kubb-100p

**Python UI 框架**
- https://github.com/zauberzeug/nicegui/discussions/4434
- https://github.com/zauberzeug/nicegui/discussions/2876
- https://reflex.dev/docs/advanced-onboarding/how-reflex-works/
- https://github.com/reflex-dev/reflex/issues/105
- https://github.com/mesop-dev/mesop

**本仓库关键文件**
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules/launch_utils.py`（l.306、422-423）
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules/scripts.py`（l.8、688-731）
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules/api/api.py`（l.266-267、369-382）
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules/shared.py`（l.5）
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules/options.py`（l.5）
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules/shared_options.py`（l.3）
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules/infotext_utils.py`（l.10）
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules/anima_support.py`（l.730、773、800）
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules/anima_feature_conflicts.py`
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules/progress.py`
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules/call_queue.py`
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules_forge/main_entry.py`
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules_forge/main_thread.py`
- `/Volumes/Remi/Anima/sd-webui-forge-classic/webui.py`（api_only_worker，l.50-66）
- `/Volumes/Remi/Anima/sd-webui-forge-classic/script.js`（l.121-136）

**拟新增目录**
- `/Volumes/Remi/Anima/sd-webui-forge-classic/modules_remi/`
- `/Volumes/Remi/Anima/sd-webui-forge-classic/extensions-builtin/remi-webui/`
- `/Volumes/Remi/Anima/sd-webui-forge-classic/webui-next/`