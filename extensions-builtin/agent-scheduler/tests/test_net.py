import http.server
import importlib.util
import threading
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("agent_scheduler_net", Path(__file__).parents[1] / "agent_scheduler" / "net.py")
net = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(net)


@pytest.fixture()
def local_url():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


def test_peer_address_is_checked_on_the_socket(local_url, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")

    with net.guarded_session(net.callback_address_allowed) as session:
        assert session.get(local_url, timeout=5).text == "ok"
    with net.guarded_session(net.global_address_allowed) as session:
        with pytest.raises(net.AddressNotAllowed):
            session.get(local_url, timeout=5)


@pytest.mark.parametrize(
    "address",
    ["169.254.169.254", "::ffff:169.254.169.254", "fd00:ec2::254", "100.100.100.200", "0.0.0.0", "224.0.0.1", "fe80::1%eth0"],
)
def test_callback_policy_blocks_metadata_and_special_targets(address):
    assert not net.callback_address_allowed(address)


@pytest.mark.parametrize("address", ["127.0.0.1", "::1", "192.168.1.10", "10.0.0.2", "8.8.8.8"])
def test_callback_policy_allows_local_clients(address):
    assert net.callback_address_allowed(address)


def test_global_policy_unwraps_mapped_addresses():
    assert not net.global_address_allowed("::ffff:127.0.0.1")
    assert net.global_address_allowed("8.8.8.8")


def test_nat64_targets_are_unwrapped():
    assert not net.global_address_allowed("64:ff9b::a9fe:a9fe")
    assert not net.callback_address_allowed("64:ff9b::a9fe:a9fe")
    assert net.global_address_allowed("64:ff9b::808:808")


def test_proxied_requests_check_the_target(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    with net.guarded_session(net.callback_address_allowed) as session:
        with pytest.raises(net.AddressNotAllowed):
            session.post("http://169.254.169.254/latest/", timeout=5)
