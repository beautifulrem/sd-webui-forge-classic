"""Prompt tools: the tag index and its ranking, completion sources, the
history / favorites store, translation providers and the
``/api/v2/prompt-tools`` routes (remi/prompt_tools)."""

import json
import sqlite3
import time

import pytest

from remi import prompt_tools as pt
from remi.prompt_tools import tags as T
from remi.prompt_tools.sources import ChantIndex, scan_wildcards, wildcard_values
from remi.prompt_tools.store import PromptToolsStore, StoreError
from remi.prompt_tools.translate import ProviderConfig, TranslationError, dictionary_translate, translate

TAG_CSV = """1girl,0,6000000,"1girls,sole_female"
long_hair,0,4300000,"longhair,/lh"
breasts,0,3400000,"oppai,boobs"
hair_ornament,0,900000,
very_long_hair,0,800000,
school_uniform,0,500000,"seifuku"
uniform,0,100000,
detached_collar,0,90000,
hair_between_eyes,0,850000,
masterpiece,5,1000,
hatsune_miku,4,200000,"miku"
"""


def entries(text=TAG_CSV):
    rows = []
    for line in text.strip().splitlines():
        parts = line.split(",", 3)
        rows.append(T.TagEntry(parts[0], int(parts[1]), int(parts[2]), T._aliases(parts[3].strip('"') if len(parts) > 3 else "")))
    return rows


@pytest.fixture
def index():
    return T.TagIndex(entries(), [T.TagEntry("masterpiece", 5, 0, (), extra=True, meta="Quality tag")], {"long_hair": "长发", "1girl": "1女孩"})


def names(matches):
    return [match.entry.name for match in matches]


def search(index, query, limit=8, **kwargs):
    return T.rank(index.matches(query, limit) + index.extra_matches(query), limit=limit, **kwargs)


# region Tag index ranking


def test_exact_beats_prefix_beats_word_prefix(index):
    ranked = search(index, "hair")
    assert names(ranked)[:2] == ["hair_ornament", "hair_between_eyes"]  # prefix, by posts
    assert names(ranked)[2:4] == ["long_hair", "very_long_hair"]  # word prefixes, by posts
    assert names(search(index, "uniform"))[0] == "uniform"  # exact first, despite fewer posts


def test_spaces_and_escapes_are_normalized(index):
    assert names(search(index, "long ha"))[0] == "long_hair"
    assert T.normalize_query("  Long\\ Hair  ") == "long_hair"


def test_aliases_find_their_tag_and_report_the_alias(index):
    match = search(index, "oppai")[0]
    assert (match.entry.name, match.tier, match.alias) == ("breasts", T.Tier.ALIAS_EXACT, "oppai")
    assert "breasts" not in names(index.matches("oppai", 8, aliases=False))


def test_translations_are_searchable(index):
    assert names(search(index, "长发")) == ["long_hair"]
    assert index.translation("Long Hair") == "长发"
    assert names(index.matches("长发", 8, translations=False)) == []


def test_substring_and_abbreviation_fallbacks(index):
    assert names(search(index, "*hair")) == ["long_hair", "hair_ornament", "hair_between_eyes", "very_long_hair"]
    fuzzy = search(index, "detco")
    assert names(fuzzy) == ["detached_collar"] and fuzzy[0].tier == T.Tier.FUZZY
    assert T.abbreviation_pattern("lh").search("long_hair") and not T.abbreviation_pattern("lh").search("blonde")


def test_abbreviations_stay_within_one_tag():
    # each query's letters are spread over consecutive tags of the list, which
    # a match across line breaks of the newline-joined blob would accept
    tags = ["wooden_door", "lamp", "open_mouth", "parted_lips", "double_exposure", "tie", "comb", "detached_collar"]
    index = T.TagIndex([T.TagEntry(name, 0, 1000 - rank) for rank, name in enumerate(tags)])
    assert [m.entry.name for m in index.matches("wlop", 8) if m.tier == T.Tier.FUZZY] == []
    assert [m.entry.name for m in index.matches("detco", 8) if m.tier == T.Tier.FUZZY] == ["detached_collar"]
    assert not T.abbreviation_pattern("wl").search("wooden_door\nlamp")
    assert T.abbreviation_pattern("pl").search("parted_lips") and T.abbreviation_pattern("pl").search("parted lips")


def test_artist_prefix_is_not_searched(index):
    assert T.normalize_query("@Hatsune Mi") == "hatsune_mi"
    assert T.normalize_query("@_@") == "@_@"  # an emoticon tag, not a prefix
    assert names(search(index, "@hatsune"))[0] == "hatsune_miku"
    assert names(search(index, "@miku"))[0] == "hatsune_miku"  # by alias too


def test_extra_tags_come_first_or_last_and_replace_their_namesake(index):
    ranked = search(index, "mas")
    assert ranked[0].entry.extra and ranked[0].entry.meta == "Quality tag"
    after = T.rank(index.matches("ma", 8) + index.extra_matches("ma"), extras_first=False, limit=8)
    assert after[-1].entry.extra


def test_usage_promotes_within_limits(index):
    plain = names(search(index, "hair"))
    assert plain[0] == "hair_ornament"
    config = T.FrequencyConfig(enabled=True, function="Logarithmic (strong)", min_count=3, recommend_cap=10)
    promoted = names(search(index, "hair", uses={"hair_between_eyes": 50}, config=config))
    assert promoted[0] == "hair_between_eyes"
    # below the minimum count, or frequency sorting off: no change
    assert names(search(index, "hair", uses={"hair_between_eyes": 2}, config=config)) == plain
    assert names(search(index, "hair", uses={"hair_between_eyes": 50}, config=T.FrequencyConfig(enabled=False))) == plain
    # "Usage first" lists used tags before everything of their query
    first = T.FrequencyConfig(function="Usage first", min_count=1)
    assert names(search(index, "hair", uses={"very_long_hair": 1}, config=first))[0] == "very_long_hair"
    # the cap: only the most used candidates are promoted
    capped = T.FrequencyConfig(function="Usage first", min_count=1, recommend_cap=1)
    ranked = names(search(index, "hair", uses={"very_long_hair": 5, "long_hair": 4}, config=capped))
    assert ranked[0] == "very_long_hair" and ranked.index("long_hair") > ranked.index("hair_between_eyes")


def test_limit_and_lookup(index):
    assert len(search(index, "h", limit=2)) == 2
    assert index.lookup("Hatsune Miku").category == 4
    assert index.lookup("masterpiece") is not None and index.lookup("nope") is None


def test_files_and_formats(tmp_path):
    (tmp_path / "tags.csv").write_text(TAG_CSV, encoding="utf8")
    (tmp_path / "extra.csv").write_text("masterpiece,5,Quality tag,,杰作\nbest_quality,5,,,\n", encoding="utf8")
    (tmp_path / "old.csv").write_text("long_hair,0,长发\n", encoding="utf8")
    (tmp_path / "new.csv").write_text("long_hair,长头发\n1girl,\n", encoding="utf8")
    extras, translations = T.read_extra_file(str(tmp_path / "extra.csv"))
    assert [(e.name, e.meta) for e in extras] == [("masterpiece", "Quality tag"), ("best_quality", None)] and translations == {"masterpiece": "杰作"}
    assert T.read_translation_file(str(tmp_path / "old.csv")) == {"long_hair": "长发"}
    assert T.read_translation_file(str(tmp_path / "new.csv")) == {"long_hair": "长头发"}
    files = T.TagFiles(str(tmp_path / "tags.csv"), str(tmp_path / "extra.csv"), str(tmp_path / "new.csv"))
    cache = T.IndexCache()
    first = cache.get(files)
    assert len(first) == 11 and first.translation("masterpiece") == "杰作" and first.translation("long_hair") == "长头发"
    assert cache.get(files) is first
    (tmp_path / "tags.csv").write_text(TAG_CSV + "new_tag,0,1,\n", encoding="utf8")
    assert len(cache.get(files)) == 12


# endregion

# region Wildcards and chants


def test_wildcards_nested_yaml_and_values(tmp_path):
    root = tmp_path / "wildcards"
    (root / "hair" / "colors").mkdir(parents=True)
    (root / "hair" / "colors" / "light.txt").write_text("# comment\nblonde hair\n\nsilver hair\n", encoding="utf8")
    (root / "animals.txt").write_text("cat\ndog\n", encoding="utf8")
    (root / "skip").mkdir()
    (root / "skip" / "hidden.txt").write_text("x", encoding="utf8")
    (root / "collection.yaml").write_text("styles:\n  painting: [oil, watercolor]\n  bad: {nested: 1}\n", encoding="utf8")
    found = scan_wildcards([str(root)], ["skip"])
    assert [w.name for w in found] == ["animals", "hair/colors/light", "styles/painting"]
    assert wildcard_values(found[1]) == ["blonde hair", "silver hair"]
    assert wildcard_values(found[2]) == ["oil", "watercolor"]


def test_wildcard_scan_survives_symlink_loops_and_reuses_yaml(tmp_path, monkeypatch):
    from remi.prompt_tools import sources

    root = tmp_path / "wildcards"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.txt").write_text("one\n", encoding="utf8")
    (root / "c.yaml").write_text("c: [1, 2]\n", encoding="utf8")
    try:
        (root / "a" / "loop").symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("no symlinks here")
    parsed = []
    original = sources._yaml_wildcards
    monkeypatch.setattr(sources, "_yaml_wildcards", lambda path, source: parsed.append(path) or original(path, source))
    cache = {}
    found = scan_wildcards([str(root)], yaml_cache=cache)
    assert [w.name for w in found] == ["a/x", "c"]
    assert [w.name for w in scan_wildcards([str(root)], yaml_cache=cache)] == ["a/x", "c"] and len(parsed) == 1
    (root / "c.yaml").write_text("c: [1, 2, 3]\n", encoding="utf8")
    assert wildcard_values(scan_wildcards([str(root)], yaml_cache=cache)[1]) == ["1", "2", "3"] and len(parsed) == 2


def test_wildcard_cache_rescans_in_the_background(tmp_path):
    from remi.prompt_tools.sources import WildcardCache

    root = tmp_path / "wildcards"
    root.mkdir()
    (root / "a.txt").write_text("x\n", encoding="utf8")
    cache = WildcardCache(lambda: [str(root)], ttl=0.0, background=True)
    assert [w.name for w in cache.items()] == ["a"]
    (root / "b.txt").write_text("y\n", encoding="utf8")
    assert [w.name for w in cache.items()] == ["a"]  # stale list at once, rescan started
    deadline = time.monotonic() + 5
    while [w.name for w in cache.items()] != ["a", "b"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert [w.name for w in cache.items()] == ["a", "b"]
    (root / "c.txt").write_text("z\n", encoding="utf8")
    cache.refresh()
    assert [w.name for w in cache.items()] == ["a", "b", "c"]  # a refresh scans before answering


def test_chants_search_by_name_and_terms(tmp_path):
    path = tmp_path / "chants.json"
    path.write_text(json.dumps([{"name": "Basic-Start", "terms": "Basic, Start", "content": "(masterpiece), 1girl", "color": 3}, {"name": "Negative", "terms": "low quality", "content": "(worst quality)"}]), encoding="utf8")
    chants = ChantIndex()
    assert [c.name for c in chants.search(str(path), "basic", 5)] == ["Basic-Start"]
    assert [c.name for c in chants.search(str(path), "quality", 5)] == ["Negative"]
    assert len(chants.search(str(path), "", 5)) == 2


# endregion

# region Store


@pytest.fixture
def store(tmp_path):
    store = PromptToolsStore(tmp_path / "prompt-tools.db")
    yield store
    store.close()


def test_history_deduplicates_groups_and_searches(store):
    store.record_history("a cat", "blurry", "txt2img", when=1)
    store.record_history("a dog", "blurry", "txt2img", when=2)
    again = store.record_history("a cat", "blurry", "img2img", when=3)
    assert again["uses"] == 2 and again["tab"] == "img2img"
    items, cursor, total = store.history()
    assert [item["prompt"] for item in items] == ["a cat", "a dog"] and total == 2 and cursor is None
    negatives, _, total = store.history(field="negative")
    assert total == 1 and negatives[0]["negative_prompt"] == "blurry" and negatives[0]["uses"] == 3
    assert [item["prompt"] for item in store.history(query="DOG")[0]] == ["a dog"]
    assert store.history(query="a blurry")[2] == 2 and store.history(query="cat dog")[2] == 0
    assert store.record_history("", "  ") is None


def test_grouped_history_rename_and_delete_cover_the_group(store):
    first = store.record_history("a cat", "blurry", when=1)
    store.record_history("a cat", "lowres", when=2)
    store.record_history("a dog", "lowres", when=3)
    [cat, dog] = sorted(store.history(field="prompt")[0], key=lambda item: item["prompt"])
    assert cat["uses"] == 2
    store.rename_history(first["id"], "kitty", field="prompt")
    assert {item["name"] for item in store.history()[0] if item["prompt"] == "a cat"} == {"kitty"}
    assert store.delete_history(first["id"], field="prompt") == 2
    assert [item["prompt"] for item in store.history(field="prompt")[0]] == ["a dog"]
    assert store.delete_history(dog["id"]) == 1 and store.delete_history(dog["id"]) == 0


def test_history_paging_limit_names_and_removal(store):
    for number in range(5):
        store.record_history(f"prompt {number}", "", when=number)
    page, cursor, total = store.history(limit=2)
    assert [item["prompt"] for item in page] == ["prompt 4", "prompt 3"] and total == 5
    assert [item["prompt"] for item in store.history(limit=2, cursor=cursor)[0]] == ["prompt 2", "prompt 1"]
    oldest = store.history(limit=5)[0][-1]
    store.rename_history(oldest["id"], "keeper")
    store.record_history("newest", "", when=10, limit=2)
    kept = {item["prompt"] for item in store.history()[0]}
    assert kept == {"newest", "prompt 4", "prompt 0"}  # the named one survives the limit
    assert store.clear_history() == 2 and [item["name"] for item in store.history()[0]] == ["keeper"]
    with pytest.raises(StoreError):
        store.rename_history("missing", "x")


def test_favorites_collections_and_order(store):
    collection = store.create_collection("Portraits")
    with pytest.raises(StoreError):
        store.create_collection("Portraits")
    a = store.add_favorite("1girl, smile", name="smile", collection_id=collection["id"])
    b = store.add_favorite("1girl, frown", collection_id=collection["id"])
    loose = store.add_favorite("blurry", "negative")
    assert [f["id"] for f in store.favorites(collection_id=collection["id"])] == [b["id"], a["id"]]  # newest first
    store.update_favorite(b["id"], {"before": None})  # to the end
    assert [f["id"] for f in store.favorites(collection_id=collection["id"])] == [a["id"], b["id"]]
    store.update_favorite(b["id"], {"before": a["id"]})
    assert [f["id"] for f in store.favorites(collection_id=collection["id"])] == [b["id"], a["id"]]
    moved = store.update_favorite(loose["id"], {"collection_id": collection["id"], "name": "neg"})
    assert moved["collection_id"] == collection["id"] and moved["name"] == "neg"
    counts = {c["name"]: c["count"] for c in store.collections()}
    assert counts == {"": 0, "Portraits": 3}
    assert [f["text"] for f in store.favorites(all_collections=True, kind="negative")] == ["blurry"]
    exported = store.export_favorites()
    assert store.delete_collection(collection["id"]) and store.favorites(all_collections=True) == []
    assert store.import_favorites(exported) == 3 and store.import_favorites(exported) == 0
    with pytest.raises(StoreError):
        store.add_favorite("x", "nope")
    with pytest.raises(StoreError):
        store.update_favorite("missing", {"name": "x"})


def test_favorites_paging_kinds_and_full_export(store):
    for number in range(7):
        store.add_favorite(f"prompt {number}")
    store.add_favorite("blurry", "negative")
    store.add_favorite("smile", "tags")
    page, cursor, total = store.favorites_page(all_collections=True, kinds=["prompt", "tags"], limit=5)
    assert total == 8 and len(page) == 5 and cursor
    rest, cursor, _ = store.favorites_page(all_collections=True, kinds=["prompt", "tags"], limit=5, cursor=cursor)
    assert len(rest) == 3 and cursor is None and "blurry" not in {f["text"] for f in page + rest}
    for number in range(600):
        store.add_favorite(f"many {number}", collection_id=None)
    exported = store.export_favorites()["favorites"]
    assert len(exported) == 609  # nothing is cut off
    assert [f["text"] for f in exported[:2]] == ["many 599", "many 598"]  # in their order


def test_favorites_import_keeps_order_and_skips_bad_items(store):
    data = {"favorites": [{"text": "b", "collection": "C"}, {"text": "a", "collection": "C", "name": "n"}, {"text": 5}, "junk", {"text": "  "}, {"text": "b", "collection": "C"}, {"text": "x" * 70000}]}
    assert store.import_favorites(data) == 2
    [collection] = [c for c in store.collections() if c["id"]]
    assert [(f["text"], f["name"]) for f in store.favorites(collection_id=collection["id"])] == [("b", ""), ("a", "n")]
    with pytest.raises(StoreError):
        store.import_favorites({"nope": []})


def test_secrets_are_kept_apart(store):
    assert store.secret("translation_key") is None
    store.set_secret("translation_key", "abc")
    assert store.secret("translation_key") == "abc"
    store.set_secret("translation_key", None)
    assert store.secret("translation_key") is None


def test_usage_counts_age_and_reset(store):
    for _ in range(3):
        store.record_use("long_hair", "tag")
    store.record_use("long_hair", "tag", negative=True)
    store.record_use("old_tag", "tag", when=time.time() - 40 * 86400)
    assert store.uses(["long_hair", "old_tag", "x"], "tag") == {"long_hair": 3, "old_tag": 1}
    assert store.uses(["long_hair"], "tag", negative=True) == {"long_hair": 1}
    assert store.uses(["long_hair", "old_tag"], "tag", max_age_days=30) == {"long_hair": 3}
    assert store.uses(["long_hair"], "lora") == {}
    assert store.reset_use("long_hair", "tag") and store.uses(["long_hair"], "tag") == {}


def test_legacy_imports(store, tmp_path):
    legacy = tmp_path / "tag_frequency.db"
    with sqlite3.connect(legacy) as db:
        db.execute("CREATE TABLE tag_frequency (name TEXT, type INT, count_pos INT, count_neg INT, last_used TIMESTAMP)")
        db.execute(
            "INSERT INTO tag_frequency VALUES ('long_hair', 1, 7, 1, '2026-10-01 10:00:00'), ('mylora', 8, 2, 0, NULL), "
            "('masterpiece', 2, 3, 0, NULL), ('hair/colors', 5, 1, 0, NULL), ('blonde hair', 4, 2, 0, NULL), ('bad', 1, 'x', 0, NULL), ('style', 11, 1, 0, NULL)"
        )
    assert store.import_tag_frequency(legacy) == 5 and store.import_tag_frequency(legacy) == 0
    assert store.uses(["long_hair"], "tag") == {"long_hair": 7} and store.uses(["mylora"], "lora") == {"mylora": 2}
    assert store.uses(["masterpiece"], "tag") == {"masterpiece": 3}  # extra tags count as tags
    assert store.uses(["hair/colors"], "wildcard") == {"hair/colors": 1} and store.uses(["blonde hair"], "wildcard-value") == {"blonde hair": 2}
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "history.txt2img.json").write_text(json.dumps([{"prompt": "a cat", "time": 5, "name": "kept"}, {"prompt": "bad time", "time": "soon"}, {"prompt": "x" * 70000}, 7]), encoding="utf8")
    (storage / "favorite.txt2img_neg.json").write_text(json.dumps([{"prompt": "blurry", "name": "neg"}, {"prompt": "lowres"}, {"prompt": None}]), encoding="utf8")
    (storage / "history.img2img.json").write_text("{not json", encoding="utf8")
    assert store.import_prompt_all_in_one(storage) == 4 and store.import_prompt_all_in_one(storage) == 0
    history = {item["prompt"]: item for item in store.history()[0]}
    assert set(history) == {"a cat", "bad time"} and history["a cat"]["name"] == "kept" and history["a cat"]["last_used"] == 5
    assert [(f["kind"], f["text"], f["name"]) for f in store.favorites(all_collections=True)] == [("negative", "blurry", "neg"), ("negative", "lowres", "")]


def test_legacy_import_is_all_or_nothing(store, tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "favorite.txt2img.json").write_text(json.dumps([{"prompt": "a"}, {"prompt": "b"}]), encoding="utf8")
    calls = []
    original = PromptToolsStore._insert_favorite

    def failing(self, *args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise sqlite3.OperationalError("disk full")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PromptToolsStore, "_insert_favorite", failing)
    with pytest.raises(sqlite3.OperationalError):
        store.import_prompt_all_in_one(storage)
    assert store.favorites(all_collections=True) == [] and store.meta("imported:prompt_all_in_one") is None
    monkeypatch.setattr(PromptToolsStore, "_insert_favorite", original)
    assert store.import_prompt_all_in_one(storage) == 2  # the next start imports once, without duplicates
    assert [f["text"] for f in store.favorites(all_collections=True)] == ["a", "b"]


# endregion

def test_old_dotted_setting_keys_are_migrated():
    from remi.prompt_tools.options import migrate_values

    data = {"tac_extra.addMode": "Insert after", "tac_alias.searchByAlias": False, "tac_alias_searchByAlias": True, "tac_maxResults": 5}
    assert migrate_values(data) == ["tac_extra_addMode"]
    assert data == {"tac_extra_addMode": "Insert after", "tac_alias_searchByAlias": True, "tac_maxResults": 5}


# region Translation


def test_dictionary_translates_both_ways():
    table = {"long_hair": "长发", "1girl": "1女孩"}
    assert dictionary_translate(["long hair", "cat"], table, "zh_CN") == {"long hair": "长发"}
    assert dictionary_translate(["长发", "猫"], table, "en") == {"长发": "long hair"}


def test_no_provider_means_no_translation():
    with pytest.raises(TranslationError) as error:
        translate(["a"], "en", ProviderConfig("none"), {})
    assert error.value.status == 409
    assert not ProviderConfig("openai", url="http://x").enabled and not ProviderConfig("libretranslate").enabled


def test_network_providers_are_called_once_per_missing_text():
    calls, cache = [], {}

    def post(url, headers, body):
        calls.append((url, headers, body))
        return {"translatedText": [f"<{text}>" for text in body["q"]]}

    config = ProviderConfig("libretranslate", url="http://127.0.0.1:5000/", key="k")
    out = translate(["长发", "猫", "猫"], "en", config, {"long_hair": "长发"}, cached=lambda texts: {t: cache[t] for t in texts if t in cache}, store=cache.update, post=post)
    assert out == ["long hair", "<猫>", "<猫>"]
    assert calls[0][0] == "http://127.0.0.1:5000/translate" and calls[0][2]["q"] == ["猫"] and calls[0][2]["api_key"] == "k"
    translate(["猫"], "en", config, {}, cached=lambda texts: {t: cache[t] for t in texts if t in cache}, store=cache.update, post=post)
    assert len(calls) == 1  # cached


def test_openai_and_deepl_requests():
    def chat(url, headers, body):
        assert url == "http://127.0.0.1:11434/v1/chat/completions" and "Authorization" not in headers
        assert json.loads(body["messages"][1]["content"]) == ["猫"] and body["model"] == "qwen"
        return {"choices": [{"message": {"content": 'Sure: ["cat"]'}}]}

    assert translate(["猫"], "en", ProviderConfig("openai", url="http://127.0.0.1:11434/v1", model="qwen"), {}, post=chat) == ["cat"]

    def deepl(url, headers, body):
        assert url == "https://api-free.deepl.com/v2/translate" and headers["Authorization"] == "DeepL-Auth-Key abc:fx" and body["target_lang"] == "ZH-HANS"
        return {"translations": [{"text": "猫"}]}

    assert translate(["cat"], "zh_CN", ProviderConfig("deepl", key="abc:fx"), {}, post=deepl) == ["猫"]

    def broken(url, headers, body):
        return {"choices": [{"message": {"content": "no list"}}]}

    with pytest.raises(TranslationError):
        translate(["猫"], "en", ProviderConfig("openai", url="http://x/v1", model="m"), {}, post=broken)


# endregion

# region Routes


@pytest.fixture(scope="module")
def webui():
    from remi.testing import webui_state

    with webui_state() as state:
        yield state


@pytest.fixture
def data(tmp_path, monkeypatch):
    tags = tmp_path / "tags"
    tags.mkdir()
    (tags / "danbooru.csv").write_text(TAG_CSV, encoding="utf8")
    (tags / "extra-quality-tags.csv").write_text("masterpiece,5,Quality tag,,\n", encoding="utf8")
    (tags / "demo-chants.json").write_text(json.dumps([{"name": "Basic", "terms": "start", "content": "(masterpiece), 1girl"}]), encoding="utf8")
    groups = tmp_path / "group_tags"
    groups.mkdir()
    (groups / "zh_CN.yaml").write_text("- name: 人物\n  groups:\n    - name: 对象\n      color: red\n      tags:\n        1girl: 1女孩\n        long hair: 长发\n", encoding="utf8")
    (groups / "default.yaml").write_text("- name: Person\n  groups:\n    - name: Subject\n      tags:\n        1girl: 1girl\n", encoding="utf8")
    wildcards = tmp_path / "wildcards"
    wildcards.mkdir()
    (wildcards / "animals.txt").write_text("cat\ndog\n", encoding="utf8")
    monkeypatch.setattr(pt, "registry", pt.Registry([str(tags)], [str(groups)], [str(wildcards)]))
    monkeypatch.setattr(pt._groups, "directories", [str(groups)])
    pt.refresh()
    yield tmp_path
    pt.refresh()


@pytest.fixture
def server(webui, data, tmp_path):
    from remi.server.testing import server_app

    with server_app(tmp_path / "server") as server:
        yield server
    pt.forget(server.services)


def test_settings_export_cleanly(webui, data):
    from remi.prompt_tools import options
    from remi.ui.settings import settings_schema

    options.register_tag_options(pt.tag_dir, pt.refresh)
    options.register_tool_options()
    schema = settings_schema(webui.opts)
    sections = {section["id"]: section for category in schema["categories"] for section in category["sections"]}
    for section_id, keys in (("tac", options.TAG_OPTION_KEYS), ("prompt_tools", options.TOOL_OPTION_KEYS)):
        items = sections[section_id]["items"]
        assert [item["type"] for item in items if item["type"] == "html"] == []
        assert {item["name"] for item in items} == set(keys)
    tag_file = next(item for item in sections["tac"]["items"] if item["name"] == "tac_tagFile")
    assert [choice["value"] for choice in tag_file["choices"]] == ["None", "danbooru.csv", "extra-quality-tags.csv"]


def test_config_reflects_the_extensions_and_settings(server, webui, monkeypatch):
    config = server.client.get("/api/v2/prompt-tools/config").json()
    assert config["tags"]["enabled"] and config["tags"]["chants"] and config["tags"]["source"] == "danbooru"
    assert config["tools"] == {
        "enabled": True,
        "history": True,
        "groups": True,
        "translation": None,
        "translation_provider": "none",
        "translation_key": False,
        "language": "zh_CN",
        "comments": True,
        "toolbar": "All prompt fields",
    }
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_provider", "dictionary")
    monkeypatch.setitem(webui.opts.data, "tac_animaArtistPrefix", "Always")
    config = server.client.get("/api/v2/prompt-tools/config").json()
    assert config["tools"]["translation"] == "dictionary" and config["tags"]["artist_prefix"]
    monkeypatch.setattr(pt, "registry", pt.Registry())
    config = server.client.get("/api/v2/prompt-tools/config").json()
    assert not config["tags"]["enabled"] and not config["tools"]["enabled"]
    assert server.client.get("/api/v2/prompt-tools/complete", params={"q": "hair"}).json() == {"items": []}


def test_unloading_an_extension_turns_its_feature_off(data):
    tags, groups = pt.registry.tag_dirs[0], pt.registry.group_dirs[0]
    pt.unregister_tag_data(tags)
    assert not pt.registry.tags_enabled and pt.registry.tools_enabled
    pt.unregister_tool_data(groups)
    assert not pt.registry.tools_enabled and pt._groups.directories == []
    pt.register_tag_data(tags)
    pt.register_tool_data(groups)
    assert pt.registry.tags_enabled and pt.registry.tools_enabled


def test_translation_key_is_write_only(server, webui, monkeypatch):
    api = "/api/v2/prompt-tools"
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_provider", "deepl")
    config = server.client.get(f"{api}/config").json()["tools"]
    assert config["translation"] is None and config["translation_provider"] == "deepl" and not config["translation_key"]
    assert server.client.put(f"{api}/translation-key", json={"key": " secret:fx "}).json() == {"configured": True}
    config = server.client.get(f"{api}/config").json()["tools"]
    assert config["translation"] == "deepl" and config["translation_key"] is True
    assert "secret" not in json.dumps(server.client.get(f"{api}/config").json())
    tools = pt.tools_for(server.services)
    assert tools.translation_config().key == "secret:fx"
    assert server.client.put(f"{api}/translation-key", json={"key": ""}).json() == {"configured": False}
    assert tools.translation_config().key == ""


def test_a_key_in_the_settings_moves_to_the_store(server, webui, monkeypatch):
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_key", "old-key")
    monkeypatch.setattr(webui.opts, "save", lambda filename: None, raising=False)
    tools = pt.tools_for(server.services)
    assert tools.migrate_translation_key()
    assert "prompt_tools_translation_key" not in webui.opts.data and tools.translation_config().key == "old-key"
    assert not tools.migrate_translation_key()


def test_complete_tags_and_record_usage(server, webui, monkeypatch):
    def complete(**params):
        response = server.client.get("/api/v2/prompt-tools/complete", params=params)
        assert response.status_code == 200, response.text
        return response.json()["items"]

    items = complete(q="hair", limit=4)
    assert [item["text"] for item in items] == ["hair_ornament", "hair_between_eyes", "long_hair", "very_long_hair"]
    assert items[2]["count"] == 4300000 and items[0]["type"] == "tag"
    assert complete(q="mas")[0] == {**complete(q="mas")[0], "type": "extra", "text": "masterpiece", "meta": "Quality tag"}
    monkeypatch.setitem(webui.opts.data, "tac_frequencyMinCount", 1)
    monkeypatch.setitem(webui.opts.data, "tac_frequencyFunction", "Usage first")
    for _ in range(2):
        assert server.client.post("/api/v2/prompt-tools/usage", json={"name": "very_long_hair", "type": "tag"}).json()["ok"]
    items = complete(q="hair", limit=4)
    assert items[0]["text"] == "very_long_hair" and items[0]["uses"] == 2
    # counted per prompt: the negative prompt is unaffected
    assert complete(q="hair", limit=4, negative=True)[0]["text"] == "hair_ornament"
    assert server.client.delete("/api/v2/prompt-tools/usage", params={"name": "very_long_hair"}).json()["ok"]
    assert complete(q="hair", limit=1)[0]["text"] == "hair_ornament"
    assert complete(q="@hatsune")[0]["text"] == "hatsune_miku"  # Anima's artist prefix is not searched
    assert complete(mode="chant", q="bas")[0] == {**complete(mode="chant", q="bas")[0], "type": "chant", "text": "(masterpiece), 1girl", "label": "Basic"}
    assert [item["text"] for item in complete(mode="wildcard", q="ani")] == ["animals"]
    assert [item["text"] for item in complete(mode="wildcard-value", wildcard="animals", q="d")] == ["dog"]
    assert server.client.get("/api/v2/prompt-tools/complete", params={"mode": "nope"}).status_code == 422


def test_lookup_groups_and_dictionary_translation(server, webui, monkeypatch):
    found = server.client.post("/api/v2/prompt-tools/lookup", json={"tags": ["long hair", "1girl", "unknown thing"]}).json()["tags"]
    assert found["long hair"] == {"category": 0, "count": 4300000, "translation": "长发"} and "unknown thing" not in found
    categories = server.client.get("/api/v2/prompt-tools/groups", params={"locale": "zh-CN"}).json()["categories"]
    assert categories[0]["name"] == "人物" and categories[0]["groups"][0]["tags"][0] == {"tag": "1girl", "label": "1女孩"}
    assert server.client.get("/api/v2/prompt-tools/groups", params={"locale": "en"}).json()["categories"][0]["name"] == "Person"
    assert server.client.post("/api/v2/prompt-tools/translate", json={"texts": ["长发"]}).status_code == 409
    monkeypatch.setitem(webui.opts.data, "prompt_tools_translation_provider", "dictionary")
    answer = server.client.post("/api/v2/prompt-tools/translate", json={"texts": ["长发", "猫"], "target": "en"}).json()
    assert answer == {"provider": "dictionary", "translations": ["long hair", None]}


def test_network_completion_reads_the_network_index(server, tmp_path):
    from PIL import Image

    from remi.networks import NetworkEntry, NetworkIndex, lora_prompt

    class Static:
        def __init__(self, entries):
            self.entries = entries

        def refresh(self):
            pass

        def list(self):
            return list(self.entries)

    root = tmp_path / "Lora"
    (root / "style").mkdir(parents=True)
    lora = root / "style" / "ink.safetensors"
    lora.write_bytes(b"\0" * 16)
    (root / "style" / "ink.json").write_text(json.dumps({"preferred weight": 0.7, "activation text": "ink style", "sd version": "SDXL", "description": "Ink lines"}), encoding="utf8")
    Image.new("RGB", (8, 8)).save(root / "style" / "ink.png")
    plain = root / "plain.safetensors"
    plain.write_bytes(b"\0" * 16)
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    (embeddings / "bad_hands.pt").write_bytes(b"\0" * 16)

    def failing(user):
        raise RuntimeError("broken provider")

    entries = {
        "lora": [
            NetworkEntry("lora", "ink", str(lora), title="ink", root=str(root), alias="inkAlias", prompt=lambda user: lora_prompt("inkAlias", user)),
            NetworkEntry("lora", "plain", str(plain), title="plain", root=str(root)),
        ],
        "embeddings": [
            NetworkEntry("embeddings", "bad_hands", str(embeddings / "bad_hands.pt"), root=str(embeddings), prompt=lambda user: "bad_hands_v2"),
            NetworkEntry("embeddings", "bad_feet", str(embeddings / "bad_feet.pt"), root=str(embeddings), prompt=failing),
        ],
    }
    index = NetworkIndex(server.services.files, lambda: {kind: Static(items) for kind, items in entries.items()}, option=lambda name, default: default)
    server.services.networks = index
    pt.tools_for(server.services).networks = index

    def complete(**params):
        response = server.client.get("/api/v2/prompt-tools/complete", params=params)
        assert response.status_code == 200, response.text
        return response.json()["items"]

    [ink] = complete(mode="lora", q="ink")
    preview = ink.pop("preview")
    assert isinstance(preview, str) and preview
    assert {key: value for key, value in ink.items() if value is not None} == {"type": "lora", "text": "inkAlias", "label": "ink", "weight": 0.7, "keywords": "ink style", "meta": "SDXL", "description": "Ink lines"}
    [plain_item] = complete(mode="lora", q="plain")
    assert plain_item["weight"] == 1.0 and plain_item["keywords"] is None and plain_item["meta"] is None and plain_item["preview"] is None
    found = {item["label"]: item["text"] for item in complete(mode="embedding", q="bad")}
    assert found == {"bad_hands": "bad_hands_v2", "bad_feet": "bad_feet"}  # a failing prompt falls back to the name
    assert {item["type"] for item in complete(mode="network", q="")} == {"lora", "embedding", "chant"}


def test_history_api_and_recording_from_jobs(server):
    api = "/api/v2/prompt-tools/history"
    created = server.client.post(api, json={"prompt": "a cat", "negative_prompt": "blurry", "tab": "txt2img"})
    assert created.status_code == 201 and created.json()["uses"] == 1
    assert server.client.post(api, json={}).status_code == 422
    entry = created.json()
    assert server.client.patch(f"{api}/{entry['id']}", json={"name": "cat"}).json()["name"] == "cat"
    assert server.client.patch(f"{api}/missing", json={"name": "x"}).status_code == 404

    tools = pt.attach(server.services)
    # paused: the job stays queued (and is removed below, leaving no task behind)
    assert server.client.post("/api/v2/queue/pause").status_code == 200
    try:
        job = server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"prompt": "a dog", "negative_prompt": "lowres"}, "scripts": {}})
        assert job.status_code == 201, job.text
        assert tools._recorder.flush()
    finally:
        if "job" in locals() and job.status_code == 201:
            assert server.client.delete(f"/api/v2/jobs/{job.json()['job_id']}").status_code == 204
        server.client.post("/api/v2/queue/resume")
    page = server.client.get(api, params={"q": "dog"}).json()
    assert page["total"] == 1 and page["items"][0]["negative_prompt"] == "lowres" and page["items"][0]["tab"] == "txt2img"
    assert server.client.get(api, params={"field": "negative"}).json()["total"] == 2
    assert server.client.delete(f"{api}/{entry['id']}").json() == {"ok": True, "removed": 1}
    assert server.client.delete(f"{api}/{entry['id']}").status_code == 404
    assert server.client.delete(api).json() == {"removed": 1}
    assert server.client.get(api).json()["total"] == 0


def test_grouped_history_api(server):
    api = "/api/v2/prompt-tools/history"
    first = server.client.post(api, json={"prompt": "a cat", "negative_prompt": "blurry"}).json()
    server.client.post(api, json={"prompt": "a cat", "negative_prompt": "lowres"})
    renamed = server.client.patch(f"{api}/{first['id']}", json={"name": "cat", "field": "prompt"})
    assert renamed.status_code == 200
    assert {item["name"] for item in server.client.get(api).json()["items"]} == {"cat"}
    assert server.client.delete(f"{api}/{first['id']}", params={"field": "prompt"}).json() == {"ok": True, "removed": 2}
    assert server.client.get(api, params={"field": "prompt"}).json()["total"] == 0


def test_favorites_api(server):
    api = "/api/v2/prompt-tools"
    collection = server.client.post(f"{api}/collections", json={"name": "Faces"}).json()
    assert server.client.post(f"{api}/collections", json={"name": "Faces"}).status_code == 409
    favorite = server.client.post(f"{api}/favorites", json={"text": "smile", "kind": "tags", "collection_id": collection["id"]})
    assert favorite.status_code == 201
    loose = server.client.post(f"{api}/favorites", json={"text": "1girl, smile"}).json()
    server.client.post(f"{api}/favorites", json={"text": "blurry", "kind": "negative"})
    assert [f["text"] for f in server.client.get(f"{api}/favorites", params={"collection": ""}).json()["items"]] == ["blurry", "1girl, smile"]
    assert server.client.get(f"{api}/favorites").json()["total"] == 3
    # the prompt editor asks for its own kind and the tags
    wanted = server.client.get(f"{api}/favorites", params=[("kind", "prompt"), ("kind", "tags")]).json()
    assert sorted(f["text"] for f in wanted["items"]) == ["1girl, smile", "smile"] and wanted["total"] == 2
    paged = server.client.get(f"{api}/favorites", params={"limit": 2}).json()
    assert len(paged["items"]) == 2 and paged["next_cursor"]
    assert len(server.client.get(f"{api}/favorites", params={"limit": 2, "cursor": paged["next_cursor"]}).json()["items"]) == 1
    server.client.delete(f"{api}/favorites/{server.client.get(f'{api}/favorites', params={'kind': 'negative'}).json()['items'][0]['id']}")
    moved = server.client.patch(f"{api}/favorites/{loose['id']}", json={"collection_id": collection["id"]}).json()
    assert moved["collection_id"] == collection["id"]
    # name only: the collection stays
    assert server.client.patch(f"{api}/favorites/{loose['id']}", json={"name": "both"}).json()["collection_id"] == collection["id"]
    counts = {c["name"]: c["count"] for c in server.client.get(f"{api}/collections").json()}
    assert counts == {"": 0, "Faces": 2}
    exported = server.client.get(f"{api}/favorites-export").json()
    assert server.client.delete(f"{api}/collections/{collection['id']}").json()["ok"]
    assert server.client.get(f"{api}/favorites").json()["items"] == []
    assert server.client.post(f"{api}/favorites-import", json=exported).json() == {"added": 2}
    assert server.client.post(f"{api}/favorites", json={"text": ""}).status_code == 422


# endregion
