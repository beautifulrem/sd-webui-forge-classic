"""The gallery (filters, sort, stars, zip, delete, reveal) and PNG Info."""

import io
import os
import time
import zipfile
from types import SimpleNamespace

import piexif
import piexif.helper
import pytest
from PIL import Image, PngImagePlugin

from remi.library import GalleryIndex, category_of, model_of
from remi.server.routes import library as routes
from remi.server.testing import server_app
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui(tmp_path_factory):
    root = tmp_path_factory.mktemp("outputs")
    options = {
        "outdir_samples": "",
        "outdir_txt2img_samples": str(root / "txt2img"),
        "outdir_img2img_samples": str(root / "img2img"),
        "outdir_extras_samples": str(root / "extras"),
        "outdir_grids": "",
        "outdir_txt2img_grids": str(root / "grids"),
        "outdir_save": str(root / "saved"),
    }
    with webui_state(options) as state:
        state.root = root
        yield state


@pytest.fixture
def server(webui, tmp_path):
    with server_app(tmp_path) as server:
        yield server


def write_png(path, size=(64, 32), text=None, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    if text:
        info.add_text("parameters", text)
    for key, value in (extra or {}).items():
        info.add_itxt(key, value)
    Image.new("RGB", size, (30, 60, 90)).save(path, pnginfo=info)
    return str(path)


def infotext(prompt, model, seed=1):
    return f"{prompt}\nNegative prompt: ugly\nSteps: 20, Sampler: Euler, Seed: {seed}, Size: 64x32, Model: {model}"


@pytest.fixture
def library(server, webui, tmp_path):
    files = server.services.files
    base = time.time() - 1000
    made = {}
    for index, (name, folder, tab, text, size) in enumerate(
        [
            ("fox.png", "txt2img", "txt2img", infotext("a red fox", "anima-v1"), (64, 32)),
            ("whale.png", "txt2img", "txt2img", infotext("a blue whale", "sdxl"), (32, 64)),
            ("grid.png", "grids", "txt2img", infotext("a red fox grid", "anima-v1"), (128, 64)),
            ("old.png", "img2img", None, infotext("repainted fox", "anima-v1"), (40, 40)),
            ("upscaled.png", "extras", "extras", None, (256, 128)),
        ]
    ):
        path = write_png(webui.root / folder / f"{tmp_path.name}-{name}", size, text)
        record = files.register_output(path, tab=tab, infotext=text, created=base + index * 100)
        made[name] = record.id
    return made


def ids(response):
    return [item["id"] for item in response.json()["items"]]


def test_categories_and_models_are_derived(library, server):
    items = {item["name"].split("-", 1)[1]: item for item in server.client.get("/api/v2/gallery").json()["items"]}
    assert {name: item["category"] for name, item in items.items()} == {"fox.png": "txt2img", "whale.png": "txt2img", "grid.png": "grids", "old.png": "img2img", "upscaled.png": "extras"}
    assert items["fox.png"]["model"] == "anima-v1" and items["upscaled.png"]["model"] is None
    assert (items["whale.png"]["width"], items["whale.png"]["height"]) == (32, 64) and items["fox.png"]["size"] > 0


def test_filters_combine(library, server):
    get = lambda **params: ids(server.client.get("/api/v2/gallery", params=params))  # noqa: E731
    assert get(model="anima-v1") == [library["old.png"], library["grid.png"], library["fox.png"]]
    assert get(category="grids") == [library["grid.png"]]
    assert get(q="red fox") == [library["grid.png"], library["fox.png"]]
    assert get(q="anima repainted") == [library["old.png"]]
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 950)) + time.strftime("%z")
    since = since[:-2] + ":" + since[-2:]
    assert library["fox.png"] not in get(since=since) and library["whale.png"] in get(since=since)
    assert server.client.get("/api/v2/gallery", params={"since": "yesterday"}).status_code == 422


def test_sorting_and_paging(library, server):
    newest = ids(server.client.get("/api/v2/gallery"))
    assert newest == [library[name] for name in ("upscaled.png", "old.png", "grid.png", "whale.png", "fox.png")]
    assert ids(server.client.get("/api/v2/gallery", params={"sort": "oldest"})) == newest[::-1]
    names = [item["name"] for item in server.client.get("/api/v2/gallery", params={"sort": "name"}).json()["items"]]
    assert names == sorted(names)
    assert ids(server.client.get("/api/v2/gallery", params={"sort": "largest"}))[0] == library["upscaled.png"]

    for sort in ("newest", "oldest", "name", "largest"):
        seen, cursor = [], None
        while True:
            page = server.client.get("/api/v2/gallery", params={"sort": sort, "limit": 2, **({"cursor": cursor} if cursor else {})}).json()
            assert page["total"] == (5 if cursor is None else None)  # counted for the first page only
            seen += [item["id"] for item in page["items"]]
            cursor = page["next_cursor"]
            if not cursor:
                break
        assert sorted(seen) == sorted(newest) and len(seen) == 5, sort


def test_facets(library, server):
    facets = server.client.get("/api/v2/gallery/facets").json()
    assert facets["total"] == 5 and facets["starred"] == 0
    assert facets["models"][0] == {"name": "anima-v1", "count": 3}
    categories = {entry["name"]: entry["count"] for entry in facets["categories"]}
    assert categories == {"txt2img": 2, "img2img": 1, "extras": 1, "grids": 1, "saved": 0, "other": 0}
    assert facets["oldest"] < facets["newest"]


def test_stars(library, server):
    assert server.client.post("/api/v2/gallery/star", json={"ids": [library["fox.png"], library["whale.png"]]}).json() == {"count": 2}
    assert ids(server.client.get("/api/v2/gallery", params={"starred": True})) == [library["whale.png"], library["fox.png"]]
    server.client.post("/api/v2/gallery/star", json={"ids": [library["whale.png"]], "starred": False})
    assert server.client.get(f"/api/v2/gallery/{library['fox.png']}").json()["starred"] is True
    assert server.client.get("/api/v2/gallery/facets").json()["starred"] == 1


def test_delete_keeps_or_removes_the_file(library, server):
    files = server.services.files
    kept, removed = files.get(library["fox.png"]).path, files.get(library["whale.png"]).path
    assert server.client.post("/api/v2/gallery/delete", json={"ids": [library["fox.png"]]}).json() == {"count": 1}
    assert os.path.exists(kept) and library["fox.png"] not in ids(server.client.get("/api/v2/gallery"))
    assert server.client.post("/api/v2/gallery/delete", json={"ids": [library["whale.png"], "unknownid12345"], "delete_files": True}).json() == {"count": 1}
    assert not os.path.exists(removed)
    assert server.client.get("/api/v2/gallery/facets").json()["total"] == 3


def test_files_gone_from_disk_drop_out(library, server):
    os.remove(server.services.files.get(library["old.png"]).path)
    page = server.client.get("/api/v2/gallery").json()
    assert library["old.png"] not in [item["id"] for item in page["items"]] and page["total"] == 4
    assert server.client.get(f"/api/v2/gallery/{library['old.png']}").status_code == 404


def test_an_unmounted_drive_hides_its_images_until_it_is_back(library, server, webui, tmp_path):
    files = server.services.files
    folder = webui.root / "drive" / tmp_path.name
    path = write_png(folder / "away.png", text=infotext("far away", "anima-v1"))
    record = files.register_output(path, tab="txt2img", infotext=infotext("far away", "anima-v1"))
    server.client.post("/api/v2/gallery/star", json={"ids": [record.id]})
    files.thumbnail(record, 128)
    unmounted = folder.with_name(folder.name + "-unmounted")
    os.rename(folder, unmounted)  # the whole folder is gone, like a drive that is not mounted

    page = server.client.get("/api/v2/gallery").json()
    assert record.id not in [item["id"] for item in page["items"]] and len(page["items"]) == 5
    assert server.client.get(f"/api/v2/gallery/{record.id}").status_code == 404
    assert not files.is_removed(path) and list(files.thumbs_dir.glob(f"{record.id}-*.webp"))

    os.rename(unmounted, folder)
    back = server.client.get("/api/v2/gallery", params={"starred": True}).json()
    assert [item["id"] for item in back["items"]] == [record.id] and back["items"][0]["starred"] is True
    assert files.scan_outputs([str(folder)]) == 0  # still indexed: nothing to add


def test_a_deleted_file_leaves_the_index_but_is_found_again_if_it_comes_back(library, server, webui):
    files = server.services.files
    path = files.get(library["old.png"]).path
    data = open(path, "rb").read()
    os.remove(path)
    assert library["old.png"] not in ids(server.client.get("/api/v2/gallery"))
    assert not files.is_removed(path)
    with open(path, "wb") as file:
        file.write(data)
    assert files.scan_outputs([os.path.dirname(path)]) >= 1  # the shared folder also holds other tests' images
    assert any(item["name"] == os.path.basename(path) for item in server.client.get("/api/v2/gallery").json()["items"])


def test_a_page_reads_past_missing_files(server, webui, tmp_path):
    files = server.services.files
    folder = webui.root / "many" / tmp_path.name
    made = [files.register_output(write_png(folder / f"{index:02}.png"), created=1000.0 + index).id for index in range(6)]
    unmounted = str(tmp_path / "not-mounted")
    for record_id in made[1:5]:  # on a drive that is not there: kept in the index, left out of pages
        files._db.execute("UPDATE files SET path = ? || '/' || name WHERE id = ?", (unmounted, record_id))
    seen, cursor, pages = [], None, 0
    while True:
        page = server.client.get("/api/v2/gallery", params={"sort": "oldest", "limit": 1, **({"cursor": cursor} if cursor else {})}).json()
        pages += 1
        seen += [item["id"] for item in page["items"] if item["id"] in made]
        cursor = page["next_cursor"]
        if not cursor or pages > 20:
            break
    assert seen == [made[0], made[5]]
    assert files._db.execute(f"SELECT COUNT(*) FROM files WHERE id IN ({','.join('?' * 4)})", made[1:5]).fetchone()[0] == 4


def test_the_index_syncs_only_after_the_file_index_changed(server, tmp_path):
    calls = []
    index = GalleryIndex(server.services.files, folders=lambda: calls.append(1) or [])
    index.list()
    index.list()
    index.facets()
    assert len(calls) == 1
    server.services.files.register_output(write_png(tmp_path / "new.png"))
    assert index.list()[2] == 1 and len(calls) == 2
    index.close()


def test_zip_packs_the_originals(library, server):
    response = server.client.post("/api/v2/gallery/zip", json={"ids": [library["fox.png"], library["grid.png"], "unknownid12345"]})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["count"] == 2 and result["name"].endswith(".zip")
    download = server.client.get(f"/api/v2/files/{result['id']}", params={"download": 1})
    assert download.status_code == 200 and "attachment" in download.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        names = archive.namelist()
        assert len(names) == 2 and all(name.endswith((".png")) for name in names)
        assert archive.read(names[0])[:8] == b"\x89PNG\r\n\x1a\n"
    assert server.client.post("/api/v2/gallery/zip", json={"ids": ["unknownid12345"]}).status_code == 404


def test_zip_skips_files_deleted_meanwhile_and_leaves_no_temp_file(library, server, monkeypatch):
    files = server.services.files
    index = GalleryIndex(files)
    records = {file_id: files.get(file_id) for file_id in (library["fox.png"], library["whale.png"])}
    monkeypatch.setattr(files, "get", lambda file_id: records.get(file_id))  # looked up before the file went
    os.remove(records[library["whale.png"]].path)
    path, count = index.export_zip([library["fox.png"], library["whale.png"]])
    with zipfile.ZipFile(path) as archive:
        assert count == 1 and len(archive.namelist()) == 1
    os.remove(records[library["fox.png"]].path)
    with pytest.raises(ValueError):
        index.export_zip([library["fox.png"]])
    assert list(path.parent.glob("*.tmp")) == []
    index.close()


def test_old_exports_are_pruned_with_their_index_rows(library, server):
    response = server.client.post("/api/v2/gallery/zip", json={"ids": [library["fox.png"]]}).json()
    files = server.services.files
    record = files.get(response["id"])
    index = GalleryIndex(files)
    assert index.prune_exports() == 0 and files.get(response["id"]) is not None
    old = time.time() - 7200
    os.utime(record.path, (old, old))
    files._db.execute("UPDATE files SET created = ? WHERE id = ?", (old, record.id))
    assert index.prune_exports() == 1
    assert not os.path.exists(record.path)
    assert files._db.execute("SELECT COUNT(*) FROM files WHERE kind = 'export'").fetchone()[0] == 0
    index.close()


def test_reveal_is_guarded(library, server, monkeypatch):
    from modules import shared

    from remi import library as library_module

    opened = []
    monkeypatch.setattr(library_module, "reveal", opened.append)
    assert server.client.post(f"/api/v2/gallery/{library['fox.png']}/reveal").status_code == 200
    assert opened == [server.services.files.get(library["fox.png"]).path]
    assert server.client.post("/api/v2/gallery/unknownid12345/reveal").status_code == 404

    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", True)
    assert server.client.post(f"/api/v2/gallery/{library['fox.png']}/reveal").status_code == 403
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", False)
    monkeypatch.setattr(routes, "_local", lambda request: False)
    response = server.client.post(f"/api/v2/gallery/{library['fox.png']}/reveal")
    assert response.status_code == 403 and "WebUI machine" in response.json()["detail"]
    assert len(opened) == 1


def test_local_clients_are_loopback_only():
    request = lambda host: SimpleNamespace(client=SimpleNamespace(host=host))  # noqa: E731
    assert routes._local(request("127.0.0.1")) and routes._local(request("::1"))
    assert not routes._local(request("192.168.1.20")) and not routes._local(SimpleNamespace(client=None))
    proxied = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"), headers={"x-forwarded-for": "203.0.113.9"})
    assert not routes._local(proxied)


def test_model_and_category_helpers(tmp_path):
    assert model_of('x\nSteps: 1, Model: "a, b", Seed: 2') == "a, b"
    assert model_of("x\nSteps: 1") is None and model_of(None) is None
    folders = [("grids", str(tmp_path / "g")), ("txt2img", str(tmp_path / "t"))]
    assert category_of(str(tmp_path / "g" / "a.png"), "txt2img", folders) == "grids"
    assert category_of(str(tmp_path / "t" / "a.png"), None, folders) == "txt2img"
    assert category_of(str(tmp_path / "x.png"), "extras", folders) == "extras"
    assert category_of(str(tmp_path / "x.png"), None, folders) == "other"


def test_the_index_follows_outputs_registered_later(server, webui, tmp_path):
    index = GalleryIndex(server.services.files, folders=lambda: [])
    assert index.list()[2] == 0
    path = write_png(tmp_path / "later.png", text=infotext("late", "m1"))
    server.services.files.register_output(path, tab="img2img", infotext=infotext("late", "m1"))
    items, _cursor, total = index.list()
    assert total == 1 and items[0].model == "m1" and items[0].category == "img2img"
    index.close()


# region PNG Info


def upload(server, data, name="image.png"):
    response = server.client.post("/api/v2/uploads", files={"file": (name, data, "application/octet-stream")})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_png_info_lists_parameters_text_chunks_and_chunks(server, tmp_path):
    path = write_png(tmp_path / "info.png", text=infotext("a cat", "anima-v1", seed=42), extra={"workflow": '{"nodes": []}'})
    file_id = upload(server, open(path, "rb").read())
    response = server.client.post("/api/v2/pnginfo", json={"file_id": file_id})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["format"] == "PNG" and (data["width"], data["height"]) == (64, 32) and data["infotext"].startswith("a cat")
    assert data["params"]["Seed"] == "42" and data["params"]["Model"] == "anima-v1" and data["params"]["Prompt"] == "a cat"
    entries = {(entry["group"], entry["key"]): entry["value"] for entry in data["entries"]}
    assert entries[("Image", "Mode")] == "RGB" and entries[("Metadata", "workflow")] == '{"nodes": []}'
    types = [chunk["type"] for chunk in data["chunks"]]
    assert types[0] == "IHDR" and types[-1] == "IEND" and "IDAT" in types
    assert {chunk.get("key") for chunk in data["chunks"] if chunk["type"] in ("tEXt", "iTXt")} == {"parameters", "workflow"}


def test_png_info_reads_jpeg_exif(server):
    exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"Remi"}, "Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump("a dog\nSteps: 5, Seed: 3", encoding="unicode")}})
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(buffer, format="JPEG", exif=exif)
    data = server.client.post("/api/v2/pnginfo", json={"file_id": upload(server, buffer.getvalue(), "a.jpg")}).json()
    assert data["format"] == "JPEG" and data["infotext"].startswith("a dog") and data["chunks"] == []
    entries = {(entry["group"], entry["key"]): entry["value"] for entry in data["entries"]}
    assert entries[("EXIF", "Make")] == "Remi" and entries[("EXIF Exif", "UserComment")].startswith("a dog")


def test_png_info_of_an_image_without_metadata(server):
    buffer = io.BytesIO()
    Image.new("L", (5, 5)).save(buffer, format="PNG")
    data = server.client.post("/api/v2/pnginfo", json={"file_id": upload(server, buffer.getvalue())}).json()
    assert data["infotext"] is None and data["params"] == {} and data["mode"] == "L"
    assert server.client.post("/api/v2/pnginfo", json={"file_id": "unknownid12345"}).status_code == 404


def test_png_info_refuses_a_decompression_bomb(server, monkeypatch):
    buffer = io.BytesIO()
    Image.new("L", (64, 64)).save(buffer, format="PNG")
    file_id = upload(server, buffer.getvalue())
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # 64×64 is over twice the limit: an error, not a warning
    response = server.client.post("/api/v2/pnginfo", json={"file_id": file_id})
    assert response.status_code == 415


def test_png_info_bounds_the_text_entries(server, tmp_path):
    from remi import image_meta

    path = write_png(tmp_path / "many.png", extra={f"key{index}": "x" * 30_000 for index in range(260)})
    data = image_meta.read_metadata(path)
    metadata = [entry for entry in data["entries"] if entry["group"] == "Metadata"]
    assert len(metadata) <= image_meta.MAX_INFO_ENTRIES
    assert sum(len(entry["value"]) for entry in metadata) <= image_meta.MAX_TOTAL_CHARS


# endregion
