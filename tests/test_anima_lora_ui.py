def test_ui_text_falls_back_to_chinese_and_to_the_key(monkeypatch):
    from modules import shared
    from modules.anima_lora_ui import intro_text, ui_text

    strings = {"zh": {"title": "标题"}, "en": {"title": "Title"}}
    monkeypatch.setattr(shared, "opts", type("Opts", (), {"lang": "en"})(), raising=False)
    assert ui_text(strings, "lang", "title") == "Title" and ui_text(strings, "lang", "other") == "other"
    assert ui_text(strings, "unset_option", "title") == "标题"
    assert intro_text("中文", "en", "zh") == "zh" and intro_text("English", "en", "zh") == "en"


def test_intro_block_opens_in_the_panel_language_and_only_swaps_the_text(monkeypatch):
    import gradio as gr

    from modules import shared
    from modules.anima_lora_ui import intro_block

    monkeypatch.setattr(shared, "opts", type("Opts", (), {"lang": "zh", "lang_en": "en"})(), raising=False)
    for option, language, text in (("lang_en", "English", "en"), ("lang", "中文", "zh")):
        with gr.Blocks() as demo:
            intro_block(lambda name: f"panel_{name}", "en", "zh", option)
        components = {block.elem_id: block for block in demo.blocks.values()}
        assert components["panel_intro_language"].value == language
        assert components["panel_intro"].value == text
        # The switch writes no setting: its one event only outputs the description
        (event,) = demo.fns.values()
        assert [block.elem_id for block in event.outputs] == ["panel_intro"]
