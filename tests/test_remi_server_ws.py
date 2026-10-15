"""The WebSocket hub, binary preview frames and the realtime threads."""

import asyncio
import io
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from remi.realtime import FORMAT_JPEG, FORMAT_WEBP, HEADER_SIZE, EventBus, JobRef, PreviewEncoder, PreviewFrame, ProgressMonitor, encode_preview, pack_frame, progress_snapshot, unpack_frame
from remi.server.testing import WS_URL, FakeState, StubExecutor, server_app
from remi.server.ws import MAX_EVENTS, Client
from remi.testing import webui_state


@pytest.fixture(scope="module")
def webui():
    with webui_state() as state:
        yield state


# region Frames


def test_frame_header_is_little_endian_type_format_seq_step():
    frame = pack_frame(FORMAT_WEBP, 0x01020304, 7, b"IMG")
    assert frame[:HEADER_SIZE] == bytes([1, 2, 4, 3, 2, 1, 7, 0, 0, 0]) and frame[HEADER_SIZE:] == b"IMG"
    assert unpack_frame(frame) == PreviewFrame(FORMAT_WEBP, 0x01020304, 7, b"IMG")
    with pytest.raises(ValueError):
        unpack_frame(b"\x02\x01" + bytes(8))
    with pytest.raises(ValueError):
        unpack_frame(b"\x01")


@pytest.mark.parametrize(("fmt", "magic"), [(FORMAT_JPEG, b"\xff\xd8"), (FORMAT_WEBP, b"RIFF")])
def test_previews_encode_as_jpeg_or_webp(fmt, magic):
    data = encode_preview(Image.new("RGBA", (40, 30), (255, 0, 0, 128)), fmt)
    assert data.startswith(magic)
    assert Image.open(io.BytesIO(data)).size == (40, 30)


# endregion

# region Realtime threads


def test_progress_snapshot_matches_internal_progress():
    state = FakeState()
    state.job_count, state.job_no, state.sampling_steps, state.sampling_step = 2, 1, 10, 5
    state.time_start, state.textinfo = 100.0, "Hires"
    snapshot = progress_snapshot(state, now=175.0)
    assert snapshot == {"step": 5, "steps": 10, "job_no": 1, "job_count": 2, "progress": 0.75, "eta": 25.0, "text": "Hires"}


def recorder(bus):
    seen = []
    bus.subscribe(SimpleNamespace(send_event=seen.append, send_frame=seen.append))
    return seen


def test_progress_monitor_publishes_changes_only():
    bus, state = EventBus(), FakeState()
    seen = recorder(bus)
    ref = JobRef("j1", 4, "txt2img")
    ticks = []
    monitor = ProgressMonitor(bus, lambda: ref, state=lambda: state, on_tick=lambda: ticks.append(1))
    state.job_count, state.sampling_steps, state.sampling_step, state.time_start = 1, 8, 1, time.time()
    last = monitor.tick(None)
    last = monitor.tick(last)
    state.sampling_step = 2
    monitor.tick(last)
    assert [(event["type"], event["step"], event["seq"]) for event in seen] == [("job.progress", 1, 4), ("job.progress", 2, 4)]
    assert len(ticks) == 3
    assert ProgressMonitor(bus, lambda: None, state=lambda: state).tick("x") is None
    # a job not begun yet: the state still shows the previous job's steps
    assert ProgressMonitor(bus, lambda: JobRef("j2", 5, "txt2img", begun=False), state=lambda: state).tick("x") == "x" and len(seen) == 2


def test_preview_encoder_sends_only_the_newest_frame_in_the_configured_format():
    bus, state = EventBus(), FakeState()
    seen = recorder(bus)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="png", live_preview_fps=6)
    encoder = PreviewEncoder(bus, lambda: JobRef("j", 9, "txt2img"), state=lambda: state, opts=lambda: opts)
    assert encoder.tick(None) is None  # no image yet
    state.current_image, state.id_live_preview, state.current_image_sampling_step = Image.new("RGB", (8, 8)), 1, 3
    last = encoder.tick(None)
    assert encoder.tick(last) == last  # same preview id: nothing new
    state.id_live_preview = 5  # several previews were produced meanwhile: only the newest is sent
    encoder.tick(last)
    frames = [item for item in seen if isinstance(item, PreviewFrame)]
    assert [(frame.fmt, frame.job_seq, frame.step) for frame in frames] == [(FORMAT_WEBP, 9, 3), (FORMAT_WEBP, 9, 3)]
    opts.live_previews_image_format = "jpeg"
    state.id_live_preview = 6
    encoder.tick(None)
    assert seen[-1].fmt == FORMAT_JPEG
    opts.live_previews_enable = False
    state.id_live_preview = 7
    assert encoder.tick(None) is None and len([item for item in seen if isinstance(item, PreviewFrame)]) == 3
    opts.live_preview_fps = 1000
    assert encoder.interval() == pytest.approx(1 / 30)


def test_a_new_job_never_streams_the_previous_jobs_last_preview():
    """Between a job becoming current and its ``begin()``, the state still holds the
    previous job's last preview: nothing goes out tagged with the new job until it begun."""

    bus, state = EventBus(), FakeState()
    seen = recorder(bus)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="png", live_preview_fps=6)
    current = {"ref": JobRef("a", 1, "txt2img")}
    encoder = PreviewEncoder(bus, lambda: current["ref"], state=lambda: state, opts=lambda: opts)
    state.current_image, state.id_live_preview = Image.new("RGB", (8, 8)), 4
    last = encoder.tick(None)
    current["ref"] = JobRef("b", 2, "img2img", begun=False)  # the next job, before its begin()
    last = encoder.tick(last)
    assert [frame.job_seq for frame in seen if isinstance(frame, PreviewFrame)] == [1]
    current["ref"] = JobRef("b", 2, "img2img")  # begun: the state is its own
    state.current_image, state.id_live_preview = Image.new("RGB", (8, 8)), 1
    encoder.tick(last)
    assert [frame.job_seq for frame in seen if isinstance(frame, PreviewFrame)] == [1, 2]


def _frame_times(bus):
    times = []
    bus.subscribe(SimpleNamespace(send_event=lambda event: None, send_frame=lambda frame: times.append((time.monotonic(), frame.step))))
    return times


def test_preview_encoder_wakes_on_a_new_preview_and_throttles_to_the_fps():
    bus, state = EventBus(), FakeState()
    times = _frame_times(bus)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="jpeg", live_preview_fps=2)  # 0.5 s interval
    encoder = PreviewEncoder(bus, lambda: JobRef("j", 1, "txt2img"), state=lambda: state, opts=lambda: opts)
    encoder.start()
    try:
        deadline = time.monotonic() + 2
        while not state.live_preview_listeners and time.monotonic() < deadline:
            time.sleep(0.005)
        assert state.live_preview_listeners == [encoder.notify]
        state.sampling_step = 1
        assigned = time.monotonic()
        state.assign_current_image(Image.new("RGB", (8, 8)))
        while not times and time.monotonic() < deadline:
            time.sleep(0.002)
        assert times and times[0][0] - assigned < 0.1  # not the 0.5 s polling interval
        # two more previews right away: only the newest goes out, one interval after the first
        state.sampling_step = 2
        state.assign_current_image(Image.new("RGB", (8, 8)))
        state.sampling_step = 3
        state.assign_current_image(Image.new("RGB", (8, 8)))
        while len(times) < 2 and time.monotonic() < deadline + 1:
            time.sleep(0.005)
        assert [step for _, step in times] == [1, 3]
        assert times[1][0] - times[0][0] >= 0.45
    finally:
        encoder.stop()
        encoder.join(2)
    assert state.live_preview_listeners == []


def test_preview_encoder_polls_a_state_without_listeners():
    bus = EventBus()
    times = _frame_times(bus)
    state = SimpleNamespace(current_image=None, id_live_preview=0, current_image_sampling_step=4, sampling_step=4)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="jpeg", live_preview_fps=20)
    encoder = PreviewEncoder(bus, lambda: JobRef("j", 1, "txt2img"), state=lambda: state, opts=lambda: opts)
    encoder.start()
    try:
        state.current_image, state.id_live_preview = Image.new("RGB", (8, 8)), 1
        deadline = time.monotonic() + 2
        while not times and time.monotonic() < deadline:
            time.sleep(0.005)
        assert [step for _, step in times] == [4]
    finally:
        encoder.stop()
        encoder.join(2)


def test_previews_are_not_encoded_when_nobody_watches():
    bus, state = EventBus(), FakeState()

    class Hub:
        watching = False
        frames = 0

        def send_event(self, event):
            pass

        def send_frame(self, frame):
            self.frames += 1

        def wants_frames(self):
            return self.watching

    hub = Hub()
    bus.subscribe(hub)
    opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="jpeg", live_preview_fps=6)
    encoder = PreviewEncoder(bus, lambda: JobRef("j", 1, "txt2img"), state=lambda: state, opts=lambda: opts)
    state.current_image, state.id_live_preview = Image.new("RGB", (8, 8)), 1
    last = encoder.tick(None)
    assert hub.frames == 0
    hub.watching = True
    encoder.tick(last)
    assert hub.frames == 1


# endregion

# region Hub


def test_a_slow_client_drops_the_oldest_events_and_keeps_one_preview():
    async def scenario():
        client = Client(SimpleNamespace(), max_events=3)
        for index in range(5):
            client.push_event({"type": "toast", "message": str(index)})
        for step in range(3):
            client.push_frame(PreviewFrame(FORMAT_WEBP, 1, step, b"x"))
        return client

    client = asyncio.run(scenario())
    assert [event["message"] for event in client.events] == ["2", "3", "4"] and client.dropped == 2
    assert client.preview.step == 2
    assert MAX_EVENTS >= 64


class FakeSocket:
    def __init__(self, fail_after=None):
        self.sent, self.closed, self.fail_after = [], None, fail_after

    async def send_text(self, text):
        if self.fail_after is not None and len(self.sent) >= self.fail_after:
            raise RuntimeError("socket gone")
        self.sent.append(text)

    async def send_bytes(self, data):
        self.sent.append(data)

    async def close(self, code=1000):
        self.closed = code


def test_a_client_that_fell_behind_gets_a_resync_and_a_fresh_snapshot():
    import json

    async def scenario():
        socket = FakeSocket()
        client = Client(socket, max_events=3, snapshot=lambda: [{"type": "queue.changed", "size": 0}])
        for index in range(5):
            client.push_event({"type": "toast", "message": str(index)})
        task = asyncio.create_task(client.send_loop())
        await asyncio.sleep(0.05)
        client.push_event({"type": "toast", "message": "later"})
        await asyncio.sleep(0.05)
        task.cancel()
        return [json.loads(text) for text in socket.sent]

    sent = asyncio.run(scenario())
    assert sent == [{"type": "resync", "dropped": 2}, {"type": "queue.changed", "size": 0}, {"type": "toast", "message": "later"}]


def test_a_failed_send_closes_the_socket():
    async def scenario():
        socket = FakeSocket(fail_after=1)
        client = Client(socket)
        client.push_event({"type": "a"})
        client.push_event({"type": "b"})
        await asyncio.wait_for(client.send_loop(), 1)  # returns instead of dying silently
        return socket

    socket = asyncio.run(scenario())
    assert socket.closed == 1011


def test_clients_only_get_preview_formats_they_announced():
    async def scenario():
        client = Client(SimpleNamespace())
        client.features = {"events.v1", "preview.binary", "preview.jpeg"}
        client.push_frame(PreviewFrame(FORMAT_WEBP, 1, 1, b"w"))
        first = client.preview
        client.push_frame(PreviewFrame(FORMAT_JPEG, 1, 2, b"j"))
        return first, client.preview

    first, second = asyncio.run(scenario())
    assert first is None and second.fmt == FORMAT_JPEG


def test_websocket_hello_events_and_binary_previews(webui, tmp_path):
    with server_app(tmp_path) as server, server.client.websocket_connect(WS_URL) as socket:
        hello = socket.receive_json()
        assert hello["type"] == "hello" and "preview.binary" in hello["features"]
        assert socket.receive_json() == {"type": "queue.changed", "size": 0, "paused": False, "running": None}
        socket.send_json({"type": "hello", "features": ["events.v1", "preview.binary", "preview.webp"]})
        socket.send_json({"type": "ping"})
        assert socket.receive_json() == {"type": "pong"}

        server.services.bus.publish("toast", level="info", message="hi")
        assert socket.receive_json() == {"type": "toast", "level": "info", "message": "hi"}
        server.services.bus.publish_frame(PreviewFrame(FORMAT_JPEG, 3, 1, b"jpeg"))  # not announced: skipped
        server.services.bus.publish_frame(PreviewFrame(FORMAT_WEBP, 3, 2, b"webp"))
        assert unpack_frame(socket.receive_bytes()) == PreviewFrame(FORMAT_WEBP, 3, 2, b"webp")


def test_a_job_streams_progress_and_previews_over_the_socket(webui, tmp_path, monkeypatch):
    state = FakeState()
    monkeypatch.setitem(webui.opts.data, "live_preview_fps", 30)
    monkeypatch.setitem(webui.opts.data, "live_previews_enable", True)
    executor = StubExecutor(state, steps=6, step_time=0.1)
    with server_app(tmp_path, state=state, executor=executor, realtime=True) as server, server.client.websocket_connect(WS_URL) as socket:
        socket.receive_json(), socket.receive_json()  # hello, queue
        job = server.client.post("/api/v2/jobs", json={"tab": "txt2img", "core": {"prompt": "x"}}).json()
        types, frames = [], []
        while "job.completed" not in types:
            message = socket.receive()
            if message.get("bytes") is not None:
                frames.append(unpack_frame(message["bytes"]))
            elif message.get("text"):
                import json

                event = json.loads(message["text"])
                types.append(event["type"])
                if event["type"] == "job.progress":
                    assert event["job_id"] == job["job_id"] and event["seq"] == job["seq"] and event["steps"] == 6
        assert types.index("job.queued") < types.index("job.started") < types.index("job.completed")
        assert "job.progress" in types and frames and all(frame.job_seq == job["seq"] for frame in frames)


# endregion
