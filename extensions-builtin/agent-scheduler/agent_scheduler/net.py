"""Outbound HTTP whose address policy is enforced on the connected socket.

Checking a URL's DNS answer before requesting it is not enough: the request
resolves the host again, and a rebinding DNS server can answer differently the
second time. These sessions check the peer address of every connection they
open instead. Through a configured proxy the proxy is the peer, so there the
target is checked by resolving it locally before sending: IP literals exactly,
host names as far as local DNS can see them (a name only the proxy resolves
is left to the proxy's own policy, as with any other client behind it).
"""

import ipaddress
import socket
from typing import Callable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from requests.utils import select_proxy
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

# Cloud metadata endpoints outside the link-local range.
_METADATA_ADDRESSES = frozenset(
    ipaddress.ip_address(address)
    for address in (
        "fd00:ec2::254",  # AWS IMDS over IPv6
        "100.100.100.200",  # Alibaba Cloud
    )
)


_NAT64 = ipaddress.ip_network("64:ff9b::/96")
# Local-use NAT64 (RFC 8215): the IPv4 part depends on the operator's prefix
# length, so these can reach anything and are never allowed.
_NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")


class AddressNotAllowed(ValueError):
    pass


def normalize_address(address) -> ipaddress._BaseAddress:
    address = ipaddress.ip_address(str(address).split("%")[0])
    if address.version == 6:
        embedded = address.ipv4_mapped or address.sixtofour
        if embedded is not None:
            return embedded
        if address in _NAT64:
            return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return address


def callback_address_allowed(address) -> bool:
    """Loopback / LAN callbacks are a supported use case (local clients);
    metadata, link-local, multicast and reserved targets never are."""
    address = normalize_address(address)
    if address.version == 6 and address in _NAT64_LOCAL:
        return False
    if address.is_loopback:
        return True  # ::1 is also inside the reserved ::/8 block
    return not (
        address in _METADATA_ADDRESSES
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def global_address_allowed(address) -> bool:
    address = normalize_address(address)
    if address.version == 6 and address in _NAT64_LOCAL:
        return False
    return address.is_global and address not in _METADATA_ADDRESSES


def _checked_connection(base, allowed: Callable):
    class CheckedConnection(base):
        def _new_conn(self):
            sock = super()._new_conn()
            peer = sock.getpeername()[0]
            if not allowed(peer):
                sock.close()
                raise AddressNotAllowed(f"connections to {peer} are not allowed")
            return sock

    return CheckedConnection


class _CheckedAdapter(HTTPAdapter):
    def __init__(self, allowed: Callable, **kwargs):
        self._allowed = allowed
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        allowed = self._allowed

        class CheckedHTTPPool(HTTPConnectionPool):
            ConnectionCls = _checked_connection(HTTPConnection, allowed)

        class CheckedHTTPSPool(HTTPSConnectionPool):
            ConnectionCls = _checked_connection(HTTPSConnection, allowed)

        self.poolmanager.pool_classes_by_scheme = {"http": CheckedHTTPPool, "https": CheckedHTTPSPool}

    def send(self, request, **kwargs):
        # Proxied requests use requests' own proxy pools (the peer is the
        # proxy): check the target before handing it over.
        if select_proxy(request.url, kwargs.get("proxies") or {}):
            host = urlparse(request.url).hostname or ""
            try:
                addresses = [info[4][0] for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)]
            except OSError:
                addresses = []  # only the proxy resolves it
            for address in addresses:
                if not self._allowed(address):
                    raise AddressNotAllowed(f"requests to {host} ({address}) are not allowed")
        return super().send(request, **kwargs)


def guarded_session(allowed: Callable) -> requests.Session:
    session = requests.Session()
    adapter = _CheckedAdapter(allowed)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
