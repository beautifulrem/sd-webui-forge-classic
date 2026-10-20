"""PNG Info's metadata reader (remi/image_meta.py) on its own: value display, PNG chunks, EXIF
(nested groups, UserComment encodings, limits) and the bounded entry list. The route is tested
in test_remi_server_library.py."""

import io
import random
import re
import struct
import time
import zlib

import piexif
import piexif.helper
import pytest
from PIL import Image, PngImagePlugin

from remi import image_meta
from remi.testing import webui_state


@pytest.fixture(scope="module", autouse=True)
def webui():
    with webui_state() as state:
        yield state


def chunk(kind: bytes, data: bytes = b"") -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# region values


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"plain text\nwith a newline\tand a tab", ("plain text\nwith a newline\tand a tab", "text")),
        (b"\xff\xfe\x00binary", ("<9 bytes>", "binary")),
        (b"control \x01 char", ("<14 bytes>", "binary")),
        (memoryview(b"view"), ("view", "text")),
        (300, ("300", "number")),
        (2.5, ("2.5", "number")),
        ((72, 72), ("72 × 72", "number")),
        ((0.5, 2), ("0.5 × 2", "number")),
        (True, ("True", "text")),
        (("a", 1), ("('a', 1)", "text")),
    ],
)
def test_values_are_shown_as_text_numbers_or_byte_counts(value, expected):
    assert image_meta._text(value) == expected


def test_long_values_are_cut():
    assert len(image_meta._text("x" * (image_meta.MAX_VALUE_CHARS + 5))[0]) == image_meta.MAX_VALUE_CHARS
    assert len(image_meta._text(b"y" * (image_meta.MAX_VALUE_CHARS + 5))[0]) == image_meta.MAX_VALUE_CHARS


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"UNICODE\x00" + "a cat\nSteps: 5".encode("utf-16-be"), "a cat\nSteps: 5"),
        (b"UNICODE\x00" + "plain".encode("utf-16-le"), "plain"),
        (b"UNICODE\x00" + "猫と犬".encode("utf-16-be"), "猫と犬"),
        (b"ASCII\x00\x00\x00a dog", "a dog"),
        (b"\x00" * 8 + b"undefined code", "undefined code"),
        (b"no prefix at all", "no prefix at all"),
    ],
)
def test_exif_user_comment_encodings(data, expected):
    assert image_meta._user_comment(data) == expected


def test_a_unicode_user_comment_that_decodes_to_nothing_readable_falls_back_to_utf8():
    # one odd byte: neither UTF-16 decodes it; the comment is shown as replaced UTF-8
    assert image_meta._user_comment(b"UNICODE\x00\xff") == "UNICODE\x00�"


# endregion

# region PNG chunks


def test_png_chunks_merge_consecutive_image_data_and_name_text_chunks(tmp_path):
    path = tmp_path / "a.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a cat")
    info.add_itxt("workflow", "{}", zip=True)
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    raw = path.read_bytes()
    # split the image data into three IDAT chunks
    start = raw.index(b"IDAT") - 4
    (length,) = struct.unpack(">I", raw[start : start + 4])
    data = raw[start + 8 : start + 8 + length]
    parts = [data[:3], data[3:6], data[6:]]
    path.write_bytes(raw[:start] + b"".join(chunk(b"IDAT", part) for part in parts) + raw[start + 12 + length :])

    chunks = image_meta.png_chunks(str(path))
    assert [c["type"] for c in chunks] == ["IHDR", "tEXt", "iTXt", "IDAT", "IEND"]
    assert {c.get("key") for c in chunks if c["type"] in ("tEXt", "iTXt")} == {"parameters", "workflow"}
    idat = chunks[3]
    assert idat["count"] == 3 and idat["length"] == length


def test_png_chunks_of_other_files_and_cut_or_unreadable_ones(tmp_path):
    jpeg = tmp_path / "a.jpg"
    Image.new("RGB", (4, 4)).save(jpeg)
    assert image_meta.png_chunks(str(jpeg)) == []
    assert image_meta.png_chunks(str(tmp_path / "missing.png")) == []

    cut = tmp_path / "cut.png"
    cut.write_bytes(PNG_SIGNATURE + chunk(b"IHDR", b"\x00" * 13) + b"\x00\x00")
    assert [c["type"] for c in image_meta.png_chunks(str(cut))] == ["IHDR"]


def test_png_chunks_are_bounded(tmp_path):
    many = tmp_path / "many.png"
    many.write_bytes(PNG_SIGNATURE + b"".join(chunk(b"tEXt" if i % 2 else b"zTXt", b"k%d\x00v" % i) for i in range(image_meta.MAX_CHUNKS + 50)))
    assert len(image_meta.png_chunks(str(many))) == image_meta.MAX_CHUNKS


def test_an_unreadable_png_keeps_the_chunks_read_so_far(tmp_path, monkeypatch):
    path = tmp_path / "a.png"
    path.write_bytes(PNG_SIGNATURE + chunk(b"IHDR", b"\x00" * 13) + chunk(b"IDAT", b"x") + chunk(b"IEND"))
    real_open = open

    class FailingFile:
        def __init__(self, file):
            self.file, self.reads = file, 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.file.close()

        def read(self, size):
            self.reads += 1
            if self.reads > 3:
                raise OSError("device gone")
            return self.file.read(size)

        def seek(self, *args):
            return self.file.seek(*args)

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: FailingFile(real_open(*args, **kwargs)))
    assert [c["type"] for c in image_meta.png_chunks(str(path))] == ["IHDR", "IDAT"]


# endregion

# region EXIF and the whole reader


def jpeg_with_exif(exif: dict) -> io.BytesIO:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(buffer, format="JPEG", exif=piexif.dump(exif))
    buffer.seek(0)
    return buffer


def test_exif_groups_gps_names_and_unknown_tags():
    exif = Image.Exif()
    exif[0x010F] = "Remi"  # Make
    exif[0xC5FE] = "custom"  # a tag PIL has no name for
    exif.get_ifd(0x8769)[0x9286] = b"ASCII\x00\x00\x00a dog"  # Exif IFD: UserComment
    exif.get_ifd(0x8825)[1] = "N"  # GPS IFD: GPSLatitudeRef
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16)).save(buffer, format="JPEG", exif=exif.tobytes())
    with Image.open(buffer) as image:
        entries = {(e["group"], e["key"]): e["value"] for e in image_meta._exif_entries(image)}
    assert entries[("EXIF", "Make")] == "Remi"
    assert entries[("EXIF", "0xc5fe")] == "custom"
    assert entries[("EXIF Exif", "UserComment")] == "a dog"
    assert entries[("EXIF GPS", "GPSLatitudeRef")] == "N"
    # the nested IFD pointers themselves are not listed
    assert not any(key in ("ExifOffset", "GPSInfo") for _group, key in entries)


def test_exif_is_bounded_and_unreadable_exif_is_skipped(monkeypatch):
    monkeypatch.setattr(image_meta, "MAX_EXIF_TAGS", 2)
    exif = {"0th": {piexif.ImageIFD.Make: b"a", piexif.ImageIFD.Model: b"b", piexif.ImageIFD.Software: b"c"}}
    with Image.open(jpeg_with_exif(exif)) as image:
        assert len(list(image_meta._exif_entries(image))) == 2

    class Broken:
        def getexif(self):
            raise ValueError("corrupt EXIF")

    assert list(image_meta._exif_entries(Broken())) == []

    class NoNested:
        def getexif(self):
            class Exif(dict):
                def get_ifd(self, ifd):
                    raise KeyError(ifd)

            return Exif({271: "Remi"})

    assert list(image_meta._exif_entries(NoNested())) == [{"group": "EXIF", "key": "Make", "value": "Remi", "kind": "text"}]


def test_read_metadata_of_an_animation_with_a_colour_profile(tmp_path):
    path = tmp_path / "anim.png"
    frames = [Image.new("RGB", (8, 8), (i * 40, 0, 0)) for i in range(3)]
    frames[0].save(path, save_all=True, append_images=frames[1:], icc_profile=b"\x00" * 128)
    data = image_meta.read_metadata(str(path), name="shown.png")
    entries = {(e["group"], e["key"]): e for e in data["entries"]}
    assert data["name"] == "shown.png" and data["format"] == "PNG"
    assert entries[("Image", "Frames")]["value"] == "3"
    assert entries[("Metadata", "ICC profile")] == {"group": "Metadata", "key": "ICC profile", "value": "<128 bytes>", "kind": "binary"}
    assert {"acTL", "fcTL", "fdAT"} <= {c["type"] for c in data["chunks"]}


def test_read_metadata_cuts_a_huge_infotext_and_survives_an_unparsable_one(tmp_path, monkeypatch):
    from modules import infotext_utils

    path = tmp_path / "big.png"
    info = PngImagePlugin.PngInfo()
    # one huge prompt line: cut, it is all that is left to parse (no parameters line)
    info.add_text("parameters", "a" * (image_meta.MAX_INFOTEXT_CHARS + 10) + "\nSteps: 20, Seed: 1")
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    started = time.monotonic()
    data = image_meta.read_metadata(str(path))
    assert time.monotonic() - started < 10
    assert len(data["infotext"]) == image_meta.MAX_INFOTEXT_CHARS
    assert data["params"]["Prompt"] == data["infotext"][: image_meta.MAX_VALUE_CHARS]

    def broken(text, skip):
        raise ValueError("unparsable")

    monkeypatch.setattr(infotext_utils, "parse_generation_parameters", broken)
    data = image_meta.read_metadata(str(path))
    assert data["params"] == {} and data["infotext"]


UPSTREAM_RE_PARAM = re.compile(r'\s*([\w\s\-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')


def test_the_infotext_parameter_pattern_matches_like_upstream_in_linear_time():
    """``re_param`` (modules/infotext_utils.py) finds what Forge's pattern finds without retrying every
    position of a long run of key characters (quadratic: minutes for one 200,000-character line)."""

    from modules import infotext_utils

    rng = random.Random(7)
    alphabet = ["a", "B", " ", "  ", ":", ",", ", ", '"', "\\", "-", "/", "(", ")", ".", "Steps", "7", "\t", "猫", "_", "\n"]
    for _ in range(20_000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert infotext_utils.re_param.findall(text) == UPSTREAM_RE_PARAM.findall(text), repr(text)
    line = 'Steps: 20, Sampler: Euler a, CFG scale: 5, Seed: 1, Size: 832x1216, Lora hashes: "a: 1, b: 2", Version: neo'
    assert infotext_utils.re_param.findall(line) == UPSTREAM_RE_PARAM.findall(line)
    assert len(infotext_utils.re_param.findall(line)) == 7

    for payload in ("a" * 200_000, " " * 200_000 + "a", "a, " + " " * 200_000, ("x" * 1000 + ",") * 200, 'k: "' * 50_000):
        started = time.monotonic()
        infotext_utils.re_param.findall(payload)
        assert time.monotonic() - started < 2, payload[:20]


def test_read_metadata_leaves_out_unset_and_per_pass_parameters(tmp_path, monkeypatch):
    from modules import infotext_utils

    path = tmp_path / "p.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a cat\nSteps: 20")
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    monkeypatch.setattr(infotext_utils, "parse_generation_parameters", lambda text, skip: {"Prompt": "a cat", "Size-1": "8", "Size-2": "8", "VAE": None, "Steps": 20})
    assert image_meta.read_metadata(str(path))["params"] == {"Prompt": "a cat", "Steps": "20"}


# endregion
