import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Optional, TypeVar

from PIL import Image, ImageDraw
from torch.hub import download_url_to_file

from modules.shared import cmd_opts

from ..utils import NUM, print

URL = TypeVar("URL", bound=str)

DOWNLOADS: Final[dict[str, URL]] = {
    # https://huggingface.co/Bingsu/adetailer
    "face_yolov8n.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt?download=true",
    "face_yolov8s.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt?download=true",
    "hand_yolov8n.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt?download=true",
    "hand_yolov8s.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt?download=true",
    "person_yolov8n-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt?download=true",
    "person_yolov8s-seg.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8s-seg.pt?download=true",
    # https://github.com/ultralytics/assets
    "yolov8x-worldv2.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-worldv2.pt",
    # https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector#models
    "mediapipe_face_short.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "mediapipe_face_full.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite",
    # https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker#models
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
}
"""The stock detectors: listed before they are downloaded, and each one is
downloaded into ``models/adetailer`` the first time a unit runs with it
(never at startup or on Refresh: no network without a user action)."""

_download_lock = threading.Lock()


def downloads_allowed() -> bool:
    """``--ad-no-huggingface`` turns the stock detector downloads off (read
    when asked, so a flag set after import counts)."""

    return not getattr(cmd_opts, "ad_no_huggingface", False)


@dataclass
class PredictOutput:
    bboxes: list[tuple[NUM]] = field(default_factory=tuple)
    masks: list[Image.Image] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    preview: Optional[Image.Image] = None


def _scan_models(path: Path) -> list[Path]:
    return [
        obj
        for obj in path.rglob("*")
        if (obj.is_file() and obj.suffix in (".pt", ".tflite", ".task"))
    ]


def download_model(name: str, path: os.PathLike) -> str:
    """``path`` of the detector ``name``, downloading a missing stock model
    first (a unit is about to run with it; one download at a time, written
    to a temporary file and moved into place by ``download_url_to_file``)."""

    path = os.fspath(path)
    if os.path.isfile(path):
        return path
    url = DOWNLOADS.get(name)
    if url is None or not downloads_allowed():
        hint = " (downloads are off: --ad-no-huggingface)" if url is not None else ""
        raise FileNotFoundError(f'[ADetailer]: Model "{name}" is not in the models folder{hint}')
    with _download_lock:
        if not os.path.isfile(path):
            print(f'Downloading "{name}"...')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            download_url_to_file(url=url, dst=path, progress=False)
    return path


def get_models(ad_dir: str, *extra_dirs: str, download: bool = False) -> dict[str, os.PathLike]:
    """Detector models by file name: the files in the folders, then (unless
    ``--ad-no-huggingface``) the stock models not downloaded yet, mapped to
    where :func:`download_model` puts them. Only scans: ``download=True``
    fetches every missing stock model now (an explicit request)."""

    if download and downloads_allowed():
        for name in DOWNLOADS:
            try:
                download_model(name, os.path.join(ad_dir, name))
            except Exception:
                print(f'Failed to download "{name}"')

    models: dict[str, os.PathLike] = {}
    model_paths: list[Path] = []

    for _dir in (ad_dir, *extra_dirs):
        if not _dir or not os.path.isdir(_dir):
            continue
        model_paths.extend(_scan_models(Path(_dir)))

    for path in model_paths:
        if path.name in models:
            continue
        models[path.name] = str(path)

    if downloads_allowed():
        for name in DOWNLOADS:
            models.setdefault(name, os.path.join(ad_dir, name))

    return models


# region BBox / Mask


def create_mask_from_bbox(
    bboxes: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> list[Image.Image]:
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill=255)
        masks.append(mask)
    return masks


def create_bbox_from_mask(
    masks: list[Image.Image], shape: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for mask in masks:
        mask = mask.resize(shape)
        bbox = mask.getbbox()
        if bbox is not None:
            bboxes.append(list(bbox))
    return bboxes
