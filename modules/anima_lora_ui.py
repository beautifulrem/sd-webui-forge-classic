"""Gradio UI helpers of the Anima LoRA panels (Layer Weight, Stage Scheduler,
Artist Mixer): a "zh" / "en" settings option, and a description with its own
English / 中文 switch."""

from __future__ import annotations

UI_LANGUAGES = ("zh", "en")


def ui_language(option: str) -> str:
    """The value of the UI language setting ``option``; "zh" when unset."""

    from modules import shared

    value = getattr(shared.opts, option, "zh")
    return value if value in UI_LANGUAGES else "zh"


def ui_text(strings: dict, option: str, key: str) -> str:
    """``strings[language][key]`` in the language ``option`` selects."""

    return strings.get(ui_language(option), strings["zh"]).get(key, key)


def register_language_option(option: str, label: str, section: tuple[str, str]) -> None:
    """Add the settings radio ``option`` (once; applied on Reload UI)."""

    import gradio as gr

    from modules import shared

    if option in shared.opts.data_labels:
        return
    shared.opts.add_option(
        option,
        shared.OptionInfo(
            "zh",
            label,
            gr.Radio,
            {"choices": UI_LANGUAGES},
            section=section,
            category_id="system",
        ).needs_reload_ui(),
    )


def intro_text(language: str, intro_en: str, intro_zh: str) -> str:
    return intro_zh if language == "中文" else intro_en


def intro_block(elem_id, intro_en: str, intro_zh: str, option: str) -> None:
    """The panel description with its English / 中文 switch, opening in the
    language of the panel's settings option (``register_language_option``).
    The switch only swaps the text for this session."""

    import gradio as gr

    language = "中文" if ui_language(option) == "zh" else "English"
    with gr.Row():
        intro_language = gr.Radio(
            choices=["English", "中文"],
            value=language,
            label="Description language / 说明语言",
            elem_id=elem_id("intro_language"),
        )
    intro = gr.Markdown(value=intro_text(language, intro_en, intro_zh), elem_id=elem_id("intro"))
    intro_language.change(fn=lambda language: intro_text(language, intro_en, intro_zh), inputs=[intro_language], outputs=[intro], queue=False, show_progress=False)
