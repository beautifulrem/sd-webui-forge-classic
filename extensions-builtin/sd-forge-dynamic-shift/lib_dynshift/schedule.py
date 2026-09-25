"""
Dynamic-shift sigma schedule math for flow-matching models (pure python, no torch).

Background:
    Forge Neo's flow models (Anima / PredictionDiscreteFlow) map a uniform
    timestep t in (0, 1] to a noise level via the "time SNR shift":

        sigma(t, s) = s * t / (1 + (s - 1) * t)

    The built-in "Shift" slider applies ONE constant s to the whole schedule.
    Here we interpolate s per-step between shift_start (step 0, high noise /
    composition phase) and shift_end (last step, low noise / detail phase),
    producing a schedule that can e.g. linger in the composition regime early
    while still resolving fine detail late.

    Because PredictionDiscreteFlow.timestep(sigma) == sigma * 1000, any
    monotonically decreasing sigma sequence is a valid schedule for the
    sampler; shift only ever changes step *spacing*.
"""

import math

CURVES = ["Linear", "Cosine", "Exponential"]

# Matches diffusers' FlowMatchEulerDiscreteScheduler convention:
# base timesteps run from 1.0 down to 1/num_train_timesteps, then a final 0.
_T_MIN = 1.0 / 1000.0


def time_snr_shift(shift: float, t: float) -> float:
    return shift * t / (1.0 + (shift - 1.0) * t)


def _interp_shift(s0: float, s1: float, u: float, curve: str) -> float:
    """Interpolate shift value at progress u in [0, 1] (0 = first step)."""
    if curve == "Cosine":
        w = (1.0 - math.cos(math.pi * u)) / 2.0
    elif curve == "Exponential":
        # geometric interpolation; both shifts are > 0 by UI constraint
        return s0 * (s1 / s0) ** u
    else:  # Linear
        w = u
    return s0 + (s1 - s0) * w


def compute_sigmas(n: int, shift_start: float, shift_end: float, curve: str = "Linear") -> list[float]:
    """
    Compute n+1 sigma values (n steps + trailing 0.0) for a flow-matching
    sampler, with the shift interpolated from shift_start to shift_end
    across the steps.

    Returns a strictly decreasing list of floats ending in 0.0.
    """
    n = max(int(n), 1)
    shift_start = max(float(shift_start), 0.01)
    shift_end = max(float(shift_end), 0.01)

    sigmas: list[float] = []
    for i in range(n):
        u = i / max(n - 1, 1)                    # 0 .. 1 across steps
        t = 1.0 + (_T_MIN - 1.0) * u             # 1.0 .. 1/1000, linear
        s = _interp_shift(shift_start, shift_end, u, curve)
        sigmas.append(time_snr_shift(s, t))

    # Enforce strict monotonic decrease. When shift_end > shift_start the
    # raw values can locally rise (sigma increases with shift); clamp them.
    for i in range(1, n):
        if sigmas[i] >= sigmas[i - 1]:
            sigmas[i] = sigmas[i - 1] * 0.9999

    sigmas.append(0.0)
    return sigmas


def preview_svg(n: int, shift_start: float, shift_end: float, curve: str,
                width: int = 360, height: int = 140) -> str:
    """Small inline SVG plotting the dynamic schedule vs. both constant-shift
    references, for the UI accordion. Pure cosmetics."""
    dyn = compute_sigmas(n, shift_start, shift_end, curve)[:-1]
    ref0 = compute_sigmas(n, shift_start, shift_start, "Linear")[:-1]
    ref1 = compute_sigmas(n, shift_end, shift_end, "Linear")[:-1]

    pad = 8

    def path(vals, color, dash=""):
        pts = []
        for i, v in enumerate(vals):
            x = pad + (width - 2 * pad) * (i / max(len(vals) - 1, 1))
            y = pad + (height - 2 * pad) * (1.0 - v)
            pts.append(f"{x:.1f},{y:.1f}")
        d = "M" + " L".join(pts)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash_attr}/>'

    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="background:transparent">'
        f'{path(ref0, "#888888", dash="4 4")}'
        f'{path(ref1, "#888888", dash="1 3")}'
        f'{path(dyn, "#ff7b00")}'
        f'<text x="{pad}" y="{height - 2}" font-size="10" fill="#888">'
        f'orange = dynamic &#183; dashed = constant {shift_start:g} / dotted = constant {shift_end:g}'
        f"</text></svg>"
    )
