from types import SimpleNamespace

import pytest

from backend.args import dynamic_args
from modules import anima_online_lora as leases


@pytest.fixture(autouse=True)
def _fresh_leases(monkeypatch):
    monkeypatch.setattr(leases, "_leases", set())
    monkeypatch.setattr(leases, "_user_setting", None)
    monkeypatch.setattr(dynamic_args, "online_lora", False)


def test_the_users_setting_comes_back_only_when_the_last_owner_releases():
    outer, nested = SimpleNamespace(), SimpleNamespace()
    leases.acquire_online_lora(outer, "scheduler")
    leases.acquire_online_lora(outer, "freefuse")
    leases.acquire_online_lora(nested, "scheduler")  # ADetailer's nested run
    assert dynamic_args.online_lora is True and leases.user_online_lora() is False

    leases.release_online_lora(nested, "scheduler")
    leases.release_online_lora(outer, "scheduler")
    assert dynamic_args.online_lora is True  # FreeFuse still needs it
    assert leases.online_lora_leased_by_other_than("scheduler")

    leases.release_online_lora(outer, "freefuse")
    assert dynamic_args.online_lora is False and not leases.online_lora_leased_by_other_than("scheduler")


def test_releasing_a_lease_never_held_changes_nothing():
    owner = SimpleNamespace()
    leases.release_online_lora(owner, "scheduler")
    assert dynamic_args.online_lora is False

    dynamic_args.online_lora = True  # the user's own setting
    leases.acquire_online_lora(owner, "freefuse")
    leases.release_online_lora(owner, "scheduler")
    leases.release_online_lora(SimpleNamespace(), "freefuse")
    assert leases.user_online_lora() is True
    leases.release_online_lora(owner, "freefuse")
    assert dynamic_args.online_lora is True
