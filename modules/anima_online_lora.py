"""Leases on Forge's global online-LoRA switch (``dynamic_args.online_lora``).

Scripts that need online LoRA patches for a run (the stage scheduler for its
per-step strengths, FreeFuse for its routing) each hold a lease for their
processing object: the first lease saves the user's setting and switches
online LoRA on, the last release restores that setting. So neither a second
script in the same run nor a nested run (ADetailer) restores the setting while
another owner still needs it.
"""

from __future__ import annotations

from backend.args import dynamic_args

# (owner, id(p)) of every held lease
_leases: set[tuple[str, int]] = set()
# The user's setting while any lease is held
_user_setting: bool | None = None


def acquire_online_lora(p, owner: str) -> None:
    global _user_setting

    if not _leases:
        _user_setting = bool(dynamic_args.online_lora)
    _leases.add((owner, id(p)))
    dynamic_args.online_lora = True


def release_online_lora(p, owner: str) -> None:
    """Drop ``owner``'s lease for ``p`` (none held: nothing happens)."""
    global _user_setting

    lease = (owner, id(p))
    if lease not in _leases:
        return
    _leases.discard(lease)
    if not _leases:
        dynamic_args.online_lora = bool(_user_setting)
        _user_setting = None


def apply_user_setting(value: bool) -> None:
    """Set the user's online-LoRA setting (Forge's settings refresh, e.g. mid-run for a
    Hires checkpoint switch). While leases are held it only replaces the setting their
    last release restores, so the running owners keep online patches."""
    global _user_setting

    if _leases:
        _user_setting = bool(value)
    else:
        dynamic_args.online_lora = value


def user_online_lora() -> bool:
    """The user's own online-LoRA setting, also while leases hold it on."""
    return bool(dynamic_args.online_lora) if _user_setting is None else _user_setting


def online_lora_leased_by_other_than(owner: str) -> bool:
    """Whether an owner besides ``owner`` needs online patches right now."""
    return any(name != owner for name, _ in _leases)
