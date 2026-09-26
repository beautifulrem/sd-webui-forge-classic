def test_ui_text_falls_back_to_chinese_and_to_the_key(monkeypatch):
    from modules import shared
    from modules.anima_lora_ui import intro_text, ui_text

    strings = {"zh": {"title": "标题"}, "en": {"title": "Title"}}
    monkeypatch.setattr(shared, "opts", type("Opts", (), {"lang": "en"})(), raising=False)
    assert ui_text(strings, "lang", "title") == "Title" and ui_text(strings, "lang", "other") == "other"
    assert ui_text(strings, "unset_option", "title") == "标题"
    assert intro_text("中文", "en", "zh") == "zh" and intro_text("English", "en", "zh") == "en"
