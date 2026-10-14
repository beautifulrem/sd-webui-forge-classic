# Remi upstream sync — 2026-10-02

The starting upstream baseline is `710f1e25fcac84d880cbccf27d11b2e3276e589e`.
This update ports the following reviewed changes while retaining Remi's Anima
attention hooks, script cleanup, FP8 fast path and offload stream handling.

| Upstream commit | Integration |
| --- | --- |
| [3a3dea7 — param](https://github.com/Haoming02/sd-webui-forge-classic/commit/3a3dea7ff781f525dd227c0f6f9ed2790ad00a35) | Ignore non-tensor checkpoint metadata when detecting weight dtype. |
| [ad88b6b — draw](https://github.com/Haoming02/sd-webui-forge-classic/commit/ad88b6b484bc28a3c8c993d7d1e97d7fa0bcf506) | Canvas single-click drawing and frequent-read context hint. |
| [21886f4 — Rewrite TextProcessingEngine](https://github.com/Haoming02/sd-webui-forge-classic/commit/21886f41e0515453e3d29a5f77621b0c58291a61) | Adopt the shared encoder core for Anima with Remi tokenizer and extension compatibility. Other model engines retain their existing implementation. |
| [ceff516 — ck](https://github.com/Haoming02/sd-webui-forge-classic/commit/ceff5168f77b88a575b8bdcc58038b606a7c0795) | W6A8 INT8 registry/loader, packed-width validation, unsupported INT8 backend fallback, and comfy-kitchen 0.2.36. |

The launcher also prefers Python 3.13 for new Unix installations, preserves
existing environments and explicit `PYTHON` settings, resolves configuration
relative to the repository, and selects native PyTorch wheels on macOS.
Native PyTorch version strings no longer trigger a fallback to an assumed CUDA
version. `TORCH_COMMAND` still overrides automatic package selection.

## Adapted Anima text engine

Anima now uses the upstream `SDClipModel` encoder core through `Qwen06Engine`.
Remi keeps its existing prompt parser, raw callable Qwen/T5 tokenizers,
`tokenize_line()` and `PromptChunk` fields so NegPiP masks and FreeFuse concept
positions remain aligned with weighted T5 tokens and EOS. The old
`AnimaTextProcessingEngine` import remains an alias. `Ignore` emphasis now
consumes prompt syntax while neutralizing all weights, including negatives.
Repeated prompts cache the completed T5 conditioning within each encoding call.
Qwen output stays on the encoder device to avoid an unnecessary CPU round trip.

The encoder selects `memory_management.text_encoder_device()` rather than the
embedding weight's storage device. This preserves Remi's existing device policy
when weights are offloaded. The follow-up `46365871` mentioned in
[issue #1506](https://github.com/Haoming02/sd-webui-forge-classic/issues/1506)
still needs review from its actual Git patch. The reported NeverOOM slowdown
was not reproduced locally. This selective adaptation uses reviewed GitHub
diffs and the published Neo encoder source; complete upstream Git objects and
the latest upstream head could not be fetched because Git network access was
unavailable. Non-Anima text-engine migration remains deferred.

## Validation

Launcher regression tests use isolated temporary environments and executable
stubs: they do not install packages or start a model server. Backend policy tests
exercise the MPS/XPU/DirectML fallback without a GPU. Anima compatibility tests
cover weighted tokens, extension interfaces, empty prompts, emphasis modes,
per-call caches and execution-device selection. These 53 dependency-free tests
passed, along with shell/JavaScript syntax and whitespace checks (preserving the
launcher's original CRLF line endings).

PyTorch-dependent tests cover metadata dtype selection, grouped INT8 packed
widths/scales, encoder masks, inserted embeddings and conditioning shapes.
They could not run because this checkout has no PyTorch/pytest installation,
and package downloads were unavailable.

Before treating this adaptation as model-validated, verify checkpoint generation
with NegPiP, FreeFuse, Regional, Artist Mixer, Hires, and NeverOOM, including
non-Anima models. No claim of full upstream history synchronization is made by
this selective port.
