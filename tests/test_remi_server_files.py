"""Uploads, files and thumbnails by id, the outputs gallery and networks."""

import io
import os
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from remi.files import FileError, FileStore, clean_name, sniff, snap_width
from remi.networks import NetworkEntry, NetworkIndex
from remi.server.testing import server_app
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


@pytest.fixture
def server(webui, tmp_path):
    with server_app(tmp_path) as server:
        yield server


def png(size=(64, 48), color=(10, 20, 30), text=None):
    buffer = io.BytesIO()
    info = None
    if text:
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", text)
    Image.new("RGB", size, color).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def upload(client, data, name="image.png", mime="image/png"):
    return client.post("/api/v2/uploads", files={"file": (name, data, mime)})


# region Uploads


def test_an_image_upload_is_sniffed_sized_and_deduplicated(server):
    response = upload(server.client, png(), name="../../evil/cat.png", mime="text/plain")
    assert response.status_code == 201
    data = response.json()
    assert data["width"] == 64 and data["height"] == 48 and data["mime"] == "image/png" and data["name"] == "cat.png"
    assert upload(server.client, png(), name="again.png").json()["id"] == data["id"]
    record = server.services.files.get(data["id"])
    assert Path(record.path).is_relative_to(server.services.files.uploads_dir)
    served = server.client.get(f"/api/v2/files/{data['id']}")
    assert served.status_code == 200 and served.content == png() and served.headers["x-content-type-options"] == "nosniff"


def test_content_decides_the_type_not_the_name(server):
    assert upload(server.client, b"MZ\x90\x00 not an image", name="photo.png").status_code == 415
    assert upload(server.client, b"\x89PNG\r\n\x1a\n" + b"garbage" * 10, name="broken.png").status_code == 415
    assert upload(server.client, b"<script>alert(1)</script>", name="x.html", mime="text/html").status_code == 415
    text = upload(server.client, b"a, b\n1, 2\n", name="list.csv", mime="text/csv")
    assert text.status_code == 201 and text.json()["mime"] == "text/plain"
    assert server.client.get(f"/api/v2/files/{text.json()['id']}").headers["content-disposition"].startswith("attachment")
    assert upload(server.client, b"", name="empty.png").status_code == 400


def test_uploads_are_size_limited(server):
    server.services.files.max_upload_bytes = 1000
    response = upload(server.client, b"\x89PNG\r\n\x1a\n" + os.urandom(4000))
    assert response.status_code == 413
    assert upload(server.client, os.urandom(100_000)).status_code == 413  # refused by Content-Length
    assert not any(path.name.startswith(".incoming") for path in server.services.files.uploads_dir.iterdir())


def test_a_chunked_upload_without_content_length_is_cut_off(server):
    server.services.files.max_upload_bytes = 1000
    boundary = "remiboundary"
    head = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="big.png"\r\nContent-Type: image/png\r\n\r\n'.encode()

    def body():
        yield head
        for _ in range(20):
            yield os.urandom(64 * 1024)
        yield f"\r\n--{boundary}--\r\n".encode()

    response = server.client.post("/api/v2/uploads", content=body(), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert response.status_code == 413


def test_the_upload_stream_limit_stops_reading_early():
    import asyncio

    from remi.server.routes.files import BodyTooLarge, limited

    read = []

    async def stream():
        for index in range(100):
            read.append(index)
            yield b"x" * 1000

    async def consume():
        async for _chunk in limited(stream(), 5000):
            pass

    with pytest.raises(BodyTooLarge):
        asyncio.run(consume())
    assert len(read) == 6


def test_a_reupload_counts_as_new_for_pruning_and_concurrent_uploads_share_one_record(tmp_path):
    import threading
    import time

    store = FileStore(tmp_path)
    first = store.add_upload_bytes(png(color=(5, 5, 5)))
    store._db.execute("UPDATE files SET created = ? WHERE id = ?", (time.time() - 30 * 86400, first.id))
    again = store.add_upload_bytes(png(color=(5, 5, 5)))
    assert again.id == first.id and store.prune_uploads(86400) == 0 and store.get(first.id) is not None

    results = []
    data = png(color=(6, 6, 6))
    threads = [threading.Thread(target=lambda: results.append(store.add_upload_bytes(data).id)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(results)) == 1
    assert store._db.execute("SELECT COUNT(*) FROM files WHERE kind = 'upload'").fetchone()[0] == 2
    store.close()


def test_a_bad_form_is_rejected(server):
    assert server.client.post("/api/v2/uploads", data={"other": "x"}).status_code == 422
    assert server.client.post("/api/v2/uploads", content=b"nope", headers={"Content-Type": "application/json"}).status_code in (400, 422)


def test_sniffing_and_names():
    assert sniff(b"\xff\xd8\xff\xe0").mime == "image/jpeg"
    assert sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 ").mime == "image/webp"
    assert sniff(b"\x00\x00\x00\x18ftypisom").kind == "video"
    assert sniff(b"\x00\x00\x00\x18ftypavif").mime == "image/avif"
    assert sniff(b"\x1aE\xdf\xa3").mime == "video/webm"
    assert sniff((20).to_bytes(8, "little") + b'{"a":1}').ext == ".safetensors"
    assert sniff(b"hello", "notes.txt").kind == "text" and sniff(b"hello", "notes.exe") is None
    assert clean_name("C:\\x\\y\\pic.png") == "pic.png" and clean_name("") == "upload" and clean_name("a\x00b.png") == "ab.png"
    assert snap_width(None) == 256 and snap_width(200) == 256 and snap_width(5000) == 1024


# endregion

# region Files and thumbnails


def test_files_are_served_by_id_only(server):
    secret = server.services.root / "secret.txt"
    secret.write_text("secret")
    for path in ("/api/v2/files/..%2Fsecret.txt", "/api/v2/files/%2E%2E%2F%2E%2E%2Fetc%2Fpasswd", "/api/v2/files/short", "/api/v2/files/" + "a" * 70, "/api/v2/thumbs/..%2F..%2Fsecret.txt"):
        response = server.client.get(path)
        assert response.status_code == 404, path
        assert "secret" not in response.text


def test_a_file_whose_disk_copy_vanished_is_gone(server):
    file_id = upload(server.client, png()).json()["id"]
    os.remove(server.services.files.get(file_id).path)
    assert server.client.get(f"/api/v2/files/{file_id}").status_code == 404


def test_thumbnails_are_cached_webp_with_etags(server):
    file_id = upload(server.client, png(size=(1000, 500))).json()["id"]
    response = server.client.get(f"/api/v2/thumbs/{file_id}", params={"w": 200})
    assert response.status_code == 200 and response.headers["content-type"] == "image/webp"
    assert Image.open(io.BytesIO(response.content)).size == (256, 128)
    assert "immutable" in response.headers["cache-control"]
    etag = response.headers["etag"]
    assert server.client.get(f"/api/v2/thumbs/{file_id}", params={"w": 200}, headers={"If-None-Match": etag}).status_code == 304
    assert server.client.get(f"/api/v2/thumbs/{file_id}", params={"w": 512}).headers["etag"] != etag
    full = server.client.get(f"/api/v2/files/{file_id}")
    assert server.client.get(f"/api/v2/files/{file_id}", headers={"If-None-Match": full.headers["etag"]}).status_code == 304
    small = upload(server.client, png(size=(40, 20), color=(1, 2, 3))).json()["id"]
    assert Image.open(io.BytesIO(server.client.get(f"/api/v2/thumbs/{small}", params={"w": 1024}).content)).size == (40, 20)


def test_text_files_have_no_thumbnail(server):
    file_id = upload(server.client, b"x, y\n", name="a.csv").json()["id"]
    assert server.client.get(f"/api/v2/thumbs/{file_id}").status_code == 415


def test_the_upload_resolver_gives_pil_images_with_their_names(server):
    from remi.ui import ParamError, nodes

    file_id = upload(server.client, png(), name="source.png").json()["id"]
    image = server.services.files.resolve_upload(nodes.Image("init_img"), file_id)
    assert image.size == (64, 48) and image.filename == "source.png"
    assert server.services.files.resolve_upload(nodes.File("f"), file_id).endswith(".png")
    with pytest.raises(ParamError):
        server.services.files.resolve_upload(nodes.Image("x"), "unknownunknown")


def test_old_uploads_are_pruned_unless_a_queued_job_needs_them(tmp_path):
    store = FileStore(tmp_path)
    keep = store.add_upload_bytes(png(color=(1, 1, 1)))
    drop = store.add_upload_bytes(png(color=(2, 2, 2)))
    assert store.prune_uploads(-1, keep={keep.id}) == 1
    assert store.get(keep.id) is not None and store.get(drop.id) is None and not Path(drop.path).exists()
    store.close()


# endregion

# region Outputs gallery


def test_outputs_are_paged_and_searchable(server, tmp_path):
    files = server.services.files
    folder = tmp_path / "out"
    folder.mkdir()
    ids = []
    for index, prompt in enumerate(["red fox in snow", "blue whale", "red panda", "50% off_sale"]):
        path = folder / f"{index}.png"
        path.write_bytes(png(text=f"{prompt}\nSteps: 20"))
        os.utime(path, (1000 + index, 1000 + index))
        ids.append(files.register_output(path, tab="txt2img", infotext=f"{prompt}\nSteps: 20").id)

    page = server.client.get("/api/v2/outputs", params={"limit": 3}).json()
    assert [item["id"] for item in page["items"]] == ids[:0:-1] and page["total"] == 4
    rest = server.client.get("/api/v2/outputs", params={"cursor": page["next_cursor"]}).json()
    assert [item["id"] for item in rest["items"]] == [ids[0]] and rest["next_cursor"] is None

    assert [item["id"] for item in server.client.get("/api/v2/outputs", params={"q": "RED"}).json()["items"]] == [ids[2], ids[0]]
    assert [item["id"] for item in server.client.get("/api/v2/outputs", params={"q": "red snow"}).json()["items"]] == [ids[0]]
    assert server.client.get("/api/v2/outputs", params={"q": "%"}).json()["items"][0]["id"] == ids[3]
    assert server.client.get("/api/v2/outputs", params={"q": "_"}).json()["total"] == 1
    assert server.client.get("/api/v2/outputs", params={"cursor": "!!bad!!"}).status_code == 200

    assert server.client.delete(f"/api/v2/outputs/{ids[1]}").status_code == 204
    assert (folder / "1.png").exists() and server.client.get("/api/v2/outputs").json()["total"] == 3
    assert server.client.delete(f"/api/v2/outputs/{ids[2]}", params={"delete_file": True}).status_code == 204
    assert not (folder / "2.png").exists()
    # removed (but kept on disk) stays out of the gallery after a re-scan
    assert server.services.files.scan_outputs([str(folder)]) == 0
    assert server.client.get("/api/v2/outputs").json()["total"] == 2


def test_unsaved_outputs_are_temporary(tmp_path):
    import time

    store = FileStore(tmp_path)
    unsaved = store.save_output_image(Image.new("RGB", (8, 8)), "a cat\nSteps: 1")
    folder = tmp_path / "saved"
    folder.mkdir()
    (folder / "kept.png").write_bytes(png())
    saved = store.register_output(folder / "kept.png")
    assert store.prune_unsaved(3600) == 0
    store._db.execute("UPDATE files SET created = ?", (time.time() - 7200,))
    assert store.prune_unsaved(3600) == 1
    assert store.get(unsaved.id) is None and not Path(unsaved.path).exists()
    assert store.get(saved.id) is not None
    store.close()


def test_media_files_are_not_gzipped(server):
    file_id = upload(server.client, png(size=(512, 512))).json()["id"]
    served = server.client.get(f"/api/v2/files/{file_id}", headers={"Accept-Encoding": "gzip"})
    assert served.status_code == 200 and "content-encoding" not in served.headers
    listing = server.client.get("/api/v2/openapi-is-not-here", headers={"Accept-Encoding": "gzip"})
    assert listing.status_code == 404


def test_gallery_save_writes_copies_a_log_and_a_zip(server, webui, tmp_path, monkeypatch):
    from modules import shared

    folder = tmp_path / "gen"
    folder.mkdir()
    (folder / "a.png").write_bytes(png(text="a red fox\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 42, Size: 64x48"))
    record = server.services.files.register_output(folder / "a.png", tab="txt2img", infotext="a red fox\nSteps: 20, Sampler: Euler, CFG scale: 7, Seed: 42, Size: 64x48")
    save_dir = tmp_path / "saved"
    monkeypatch.setattr(shared.opts, "outdir_save", str(save_dir), raising=False)
    monkeypatch.setattr(shared.opts, "samples_format", "png", raising=False)
    response = server.client.post("/api/v2/outputs/save", json={"ids": [record.id], "zip": True})
    assert response.status_code == 200, response.text
    result = response.json()
    [name] = result["saved"]
    assert (save_dir / name).is_file()
    log = (save_dir / "log.csv").read_text().splitlines()
    assert log[0].startswith("prompt,seed") and log[1].startswith("a red fox,42")
    archive = server.client.get(f"/api/v2/files/{result['zip']}", params={"download": True})
    assert archive.status_code == 200 and archive.content[:2] == b"PK"
    assert server.client.post("/api/v2/outputs/save", json={"ids": ["missingmissing"]}).status_code == 404


def test_open_folder_respects_hide_ui_dir_config(server, webui, monkeypatch):
    from modules import shared
    from remi.server import gallery

    opened = []
    monkeypatch.setattr(gallery, "open_folder", opened.append)
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", True)
    assert server.client.post("/api/v2/outputs/open-folder", json={"folder": "txt2img"}).status_code == 403
    monkeypatch.setattr(shared.cmd_opts, "hide_ui_dir_config", False)
    assert server.client.post("/api/v2/outputs/open-folder", json={"folder": "txt2img"}).json()["ok"] is True
    assert opened == [gallery.output_folder("txt2img")]
    assert server.client.post("/api/v2/outputs/open-folder", json={"folder": "/etc"}).status_code == 422


def test_output_folders_are_indexed(server, tmp_path, monkeypatch):
    from remi.server import services

    folder = tmp_path / "older"
    (folder / "sub").mkdir(parents=True)
    (folder / "sub" / "a.png").write_bytes(png(text="old cat\nSteps: 3, Seed: 1, Size: 64x48"))
    (folder / "notes.txt").write_text("x")
    monkeypatch.setattr(services, "output_dirs", lambda: [str(folder), str(folder)])
    assert server.client.post("/api/v2/outputs/refresh").json() == {"added": 1}
    assert server.client.post("/api/v2/outputs/refresh").json() == {"added": 0}
    [item] = server.client.get("/api/v2/outputs", params={"q": "cat"}).json()["items"]
    assert item["name"] == "a.png" and (item["width"], item["height"]) == (64, 48)


# endregion

# region Networks


class FakeProvider:
    def __init__(self, root):
        self.root = root
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1

    def list(self):
        return [NetworkEntry("lora", path.stem, str(path), title=path.stem, root=str(self.root), hash=path.stem[:4], prompt=f"<lora:{path.stem}:1>") for path in sorted(self.root.rglob("*.safetensors"))]


def safetensors(path, metadata):
    import json

    header = json.dumps({"__metadata__": metadata}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(header).to_bytes(8, "little") + header)


@pytest.fixture
def networks(server, tmp_path):
    root = tmp_path / "Lora"
    for name in ("anime_style", "zeta_light", "alpha"):
        safetensors(root / ("sub" if name == "zeta_light" else "") / f"{name}.safetensors", {"ss_output_name": name, "ss_network_dim": "16"})
    (root / "alpha.preview.png").write_bytes(png(color=(9, 9, 9)))
    (root / "alpha.txt").write_text("the first letter")
    provider = FakeProvider(root)
    server.services.networks = NetworkIndex(server.services.files, lambda: {"lora": provider})
    return provider


def test_networks_are_searched_paged_and_never_expose_absolute_paths(server, networks):
    client = server.client
    assert [kind["id"] for kind in client.get("/api/v2/networks").json()] == ["lora"]
    page = client.get("/api/v2/networks/lora", params={"limit": 2}).json()
    assert [item["name"] for item in page["items"]] == ["alpha", "anime_style"] and page["total"] == 3
    rest = client.get("/api/v2/networks/lora", params={"cursor": page["next_cursor"]}).json()
    assert [item["name"] for item in rest["items"]] == ["zeta_light"] and rest["items"][0]["path"] == "sub/zeta_light.safetensors"
    assert [item["name"] for item in client.get("/api/v2/networks/lora", params={"q": "SUB light"}).json()["items"]] == ["zeta_light"]
    assert [item["name"] for item in client.get("/api/v2/networks/lora", params={"sort": "name", "order": "desc"}).json()["items"]][0] == "zeta_light"
    alpha = page["items"][0]
    assert alpha["description"] == "the first letter" and alpha["prompt"] == "<lora:alpha:1>"
    assert client.get(f"/api/v2/thumbs/{alpha['preview']}").status_code == 200
    assert str(networks.root) not in client.get("/api/v2/networks/lora").text
    assert client.get("/api/v2/networks/nope").status_code == 404


def test_network_metadata_and_user_metadata_editing(server, networks):
    client = server.client
    meta = client.get("/api/v2/networks/lora/alpha/meta").json()
    assert meta["metadata"] == {"ss_output_name": "alpha", "ss_network_dim": "16"} and meta["user_metadata"] == {}
    edited = client.patch("/api/v2/networks/lora/alpha/meta", json={"user_metadata": {"description": "edited", "activation text": "abc", "preferred weight": 0.7}}).json()
    assert edited["user_metadata"]["activation text"] == "abc" and edited["description"] == "edited"
    removed = client.patch("/api/v2/networks/lora/alpha/meta", json={"user_metadata": {"activation text": None}}).json()
    assert "activation text" not in removed["user_metadata"] and removed["user_metadata"]["description"] == "edited"
    assert (networks.root / "alpha.json").is_file()
    assert client.get("/api/v2/networks/lora/missing/meta").status_code == 404
    too_big = client.patch("/api/v2/networks/lora/alpha/meta", json={"user_metadata": {"notes": "x" * 70000}})
    assert too_big.status_code == 422


def test_network_previews_and_refresh(server, networks):
    client = server.client
    file_id = upload(client, png(color=(200, 0, 0))).json()["id"]
    item = client.put("/api/v2/networks/lora/anime_style/preview", json={"file_id": file_id}).json()
    assert item["preview"] and (networks.root / "anime_style.preview.png").is_file()
    # a new preview gets a new id (browsers cache files by id), revalidated
    other = upload(client, png(color=(0, 200, 0))).json()["id"]
    replaced = client.put("/api/v2/networks/lora/anime_style/preview", json={"file_id": other}).json()
    assert replaced["preview"] != item["preview"]
    assert client.get("/api/v2/networks/lora", params={"q": "anime"}).json()["items"][0]["preview"] == replaced["preview"]
    assert client.get(f"/api/v2/files/{replaced['preview']}").headers["cache-control"] == "private, no-cache"
    assert client.put("/api/v2/networks/lora/anime_style/preview", json={"file_id": "missingmissing"}).status_code == 404
    assert client.post("/api/v2/networks/lora/refresh").json()["ok"] is True and networks.refreshed == 1


def test_previews_are_converted_and_downscaled(tmp_path):
    from remi.networks import PREVIEW_SIZE, preview_image

    cmyk = preview_image(Image.new("CMYK", (3000, 1500)))
    assert cmyk.mode == "RGB" and cmyk.size == (PREVIEW_SIZE, PREVIEW_SIZE // 2)
    palette = Image.new("P", (10, 10))
    palette.info["transparency"] = 0
    assert preview_image(palette).mode == "RGBA"
    assert preview_image(Image.new("I;16", (10, 10))).mode == "RGB"


def test_network_items_are_cached_until_refreshed(server, networks, monkeypatch):
    from remi import networks as module

    reads = []
    original = module.read_user_metadata
    monkeypatch.setattr(module, "read_user_metadata", lambda entry: reads.append(entry.name) or original(entry))
    server.client.get("/api/v2/networks/lora")
    count = len(reads)
    server.client.get("/api/v2/networks/lora")
    assert len(reads) == count
    server.client.post("/api/v2/networks/lora/refresh")
    server.client.get("/api/v2/networks/lora")
    assert len(reads) > count


def test_file_store_rejects_unreadable_images(tmp_path):
    store = FileStore(tmp_path)
    with pytest.raises(FileError) as error:
        store.add_upload_bytes(b"GIF89a" + b"\x00" * 10)
    assert error.value.status == 415
    store.close()


# endregion
