import hashlib
import threading

from modules import anima_artifacts


def test_a_verified_file_is_not_blocked_by_another_download(tmp_path):
    ready = tmp_path / "gamma.npz"
    ready.write_bytes(b"cns")
    busy = tmp_path / "adapter.pt"

    lock = anima_artifacts._LOCKS_GUARD
    with lock:
        busy_lock = anima_artifacts._DOWNLOAD_LOCKS.setdefault(str(busy), threading.Lock())
    busy_lock.acquire()  # another thread is downloading adapter.pt
    try:
        result = []
        worker = threading.Thread(
            target=lambda: result.append(
                anima_artifacts.ensure_pinned_download(
                    path=str(ready), url="http://unused", sha256=hashlib.sha256(b"cns").hexdigest(), maximum_bytes=10
                )
            )
        )
        worker.start()
        worker.join(timeout=5)
        assert result == [str(ready)]
    finally:
        busy_lock.release()
