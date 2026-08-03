"""
Post Processing Suite - core effects library.

Every effect operates on a float32 numpy array of shape (H, W, 3) in linear-ish
sRGB space, values in [0, 1]. Functions are pure (return a new array) and must
preserve shape. Only numpy + PIL are required; FFT descreen uses numpy.fft.

Keeping the math here (UI-free) makes it trivial to unit-test in isolation.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

EPS = 1e-6


# ----------------------------------------------------------------------------
# conversions
# ----------------------------------------------------------------------------
def pil_to_arr(img: "Image.Image") -> tuple[np.ndarray, "Image.Image | None"]:
    """Return (float rgb array in [0,1], alpha PIL or None)."""
    alpha = None
    if img.mode == "RGBA":
        alpha = img.getchannel("A")
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr, alpha


def arr_to_pil(arr: np.ndarray, alpha: "Image.Image | None" = None) -> "Image.Image":
    out = np.clip(arr, 0.0, 1.0)
    out = (out * 255.0 + 0.5).astype(np.uint8)
    img = Image.fromarray(out, mode="RGB")
    if alpha is not None:
        img = img.convert("RGBA")
        img.putalpha(alpha)
    return img


def _lum(arr: np.ndarray) -> np.ndarray:
    """Rec.709 luma, shape (H, W, 1)."""
    return (arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722)[..., None]


def _gaussian(arr: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur via PIL. Accepts (H,W), (H,W,1) or (H,W,3); preserves shape."""
    if radius <= 0:
        return arr
    ndim = arr.ndim
    a = arr[..., None] if ndim == 2 else arr
    ch = a.shape[-1]
    if ch == 1:
        u8 = (np.clip(a[..., 0], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        pil = Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(radius)))
        out = (np.asarray(pil, dtype=np.float32) / 255.0)[..., None]
    else:
        pil = arr_to_pil(a).filter(ImageFilter.GaussianBlur(radius=float(radius)))
        out = np.asarray(pil, dtype=np.float32) / 255.0
    return out[..., 0] if ndim == 2 else out


def _rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.max(arr, axis=-1)
    mn = np.min(arr, axis=-1)
    df = mx - mn
    h = np.zeros_like(mx)
    mask = df > EPS
    rc = np.where(mask, (mx - r) / (df + EPS), 0)
    gc = np.where(mask, (mx - g) / (df + EPS), 0)
    bc = np.where(mask, (mx - b) / (df + EPS), 0)
    h = np.where(mx == r, bc - gc, h)
    h = np.where(mx == g, 2.0 + rc - bc, h)
    h = np.where(mx == b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    s = np.where(mx > EPS, df / (mx + EPS), 0)
    v = mx
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i.astype(int) % 6
    conds = [i == k for k in range(6)]
    r = np.select(conds, [v, q, p, p, t, v])
    g = np.select(conds, [t, v, v, q, p, p])
    b = np.select(conds, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


# ----------------------------------------------------------------------------
# TONE
# ----------------------------------------------------------------------------
def tone(arr, exposure=0.0, brightness=0.0, contrast=1.0, gamma=1.0,
         black=0.0, white=1.0, highlights=0.0, shadows=0.0):
    out = arr.copy()
    if exposure:
        out = out * (2.0 ** exposure)
    if brightness:
        out = out + brightness
    if white != 1.0 or black != 0.0:
        out = (out - black) / max(white - black, EPS)
    if contrast != 1.0:
        out = (out - 0.5) * contrast + 0.5
    out = np.clip(out, 0.0, 1.0)
    if gamma != 1.0:
        out = np.power(out, 1.0 / max(gamma, EPS))
    if shadows or highlights:
        l = _lum(out)
        # smooth masks: shadows weight high in darks, highlights high in brights
        s_mask = np.clip(1.0 - l * 2.0, 0.0, 1.0)
        h_mask = np.clip((l - 0.5) * 2.0, 0.0, 1.0)
        out = out + shadows * s_mask + highlights * h_mask
    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------------
# COLOR
# ----------------------------------------------------------------------------
def color(arr, saturation=1.0, vibrance=0.0, hue=0.0, temperature=0.0, tint=0.0):
    out = arr.copy()
    if temperature or tint:
        out = out + np.array([temperature, tint * 0.5 - 0.0, -temperature], dtype=np.float32)
        out[..., 1] = out[..., 1] + tint  # green/magenta on green channel
        out = np.clip(out, 0.0, 1.0)
    if hue:
        hsv = _rgb_to_hsv(out)
        hsv[..., 0] = (hsv[..., 0] + hue / 360.0) % 1.0
        out = _hsv_to_rgb(hsv)
    if vibrance:
        hsv = _rgb_to_hsv(out)
        hsv[..., 1] = np.clip(hsv[..., 1] + vibrance * (1.0 - hsv[..., 1]), 0.0, 1.0)
        out = _hsv_to_rgb(hsv)
    if saturation != 1.0:
        l = _lum(out)
        out = l + (out - l) * saturation
    return np.clip(out, 0.0, 1.0)


def color_balance(arr, sh=(0, 0, 0), mid=(0, 0, 0), hi=(0, 0, 0)):
    l = _lum(arr)
    s_mask = np.clip(1.0 - l * 2.0, 0.0, 1.0)
    h_mask = np.clip((l - 0.5) * 2.0, 0.0, 1.0)
    m_mask = np.clip(1.0 - np.abs(l - 0.5) * 2.0, 0.0, 1.0)
    out = arr.copy()
    for mask, vec in ((s_mask, sh), (m_mask, mid), (h_mask, hi)):
        out = out + mask * np.array(vec, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def split_tone(arr, shadow_color=(0.2, 0.3, 0.6), highlight_color=(0.9, 0.7, 0.3),
               balance=0.0, strength=0.0):
    if strength <= 0:
        return arr
    l = _lum(arr)
    pivot = 0.5 + balance * 0.5
    sh_w = np.clip((pivot - l) / max(pivot, EPS), 0.0, 1.0)
    hi_w = np.clip((l - pivot) / max(1.0 - pivot, EPS), 0.0, 1.0)
    sc = np.array(shadow_color, dtype=np.float32)
    hc = np.array(highlight_color, dtype=np.float32)
    tone_layer = sh_w * sc + hi_w * hc + (1.0 - sh_w - hi_w) * l
    return np.clip(arr * (1.0 - strength) + tone_layer * strength, 0.0, 1.0)


# ----------------------------------------------------------------------------
# OPTICAL
# ----------------------------------------------------------------------------
def chromatic_aberration(arr, amount=2.0, radial=True):
    if amount == 0:
        return arr
    h, w = arr.shape[:2]
    out = arr.copy()
    if radial:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        dx = (xx - cx) / max(cx, 1.0)
        dy = (yy - cy) / max(cy, 1.0)
        scale = amount / max(w, h)
        for ch, sgn in ((0, 1.0), (2, -1.0)):
            mapx = np.clip(xx + dx * amount * sgn, 0, w - 1)
            mapy = np.clip(yy + dy * amount * sgn, 0, h - 1)
            out[..., ch] = _bilinear(arr[..., ch], mapx, mapy)
    else:
        shift = int(round(amount))
        out[..., 0] = np.roll(arr[..., 0], shift, axis=1)
        out[..., 2] = np.roll(arr[..., 2], -shift, axis=1)
    return np.clip(out, 0.0, 1.0)


def _bilinear(channel, mapx, mapy):
    h, w = channel.shape
    x0 = np.floor(mapx).astype(int); x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(mapy).astype(int); y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1); y0 = np.clip(y0, 0, h - 1)
    wx = mapx - x0; wy = mapy - y0
    top = channel[y0, x0] * (1 - wx) + channel[y0, x1] * wx
    bot = channel[y1, x0] * (1 - wx) + channel[y1, x1] * wx
    return top * (1 - wy) + bot * wy


def lens_distortion(arr, k=0.0):
    if k == 0:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xx - cx) / max(cx, 1.0)
    ny = (yy - cy) / max(cy, 1.0)
    r2 = nx * nx + ny * ny
    f = 1.0 + k * r2
    mapx = np.clip(cx + nx * f * cx, 0, w - 1)
    mapy = np.clip(cy + ny * f * cy, 0, h - 1)
    out = np.empty_like(arr)
    for ch in range(3):
        out[..., ch] = _bilinear(arr[..., ch], mapx, mapy)
    return np.clip(out, 0.0, 1.0)


def bloom(arr, threshold=0.7, radius=8.0, intensity=0.6):
    if intensity <= 0:
        return arr
    l = _lum(arr)
    mask = np.clip((l - threshold) / max(1.0 - threshold, EPS), 0.0, 1.0)
    bright = arr * mask
    blurred = _gaussian(bright, radius)
    return np.clip(arr + blurred * intensity, 0.0, 1.0)


def vignette(arr, amount=0.5, radius=0.8, feather=0.5, roundness=1.0):
    if amount == 0:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    nx = (xx - cx) / max(cx, 1.0)
    ny = (yy - cy) / max(cy, 1.0)
    # roundness: 1.0 -> circle, lower -> follows aspect ratio
    ar = w / max(h, 1)
    nx = nx * (ar ** (1.0 - roundness))
    dist = np.sqrt(nx * nx + ny * ny)
    inner = radius
    outer = radius + feather + EPS
    mask = np.clip((dist - inner) / (outer - inner), 0.0, 1.0)
    mask = mask * mask * (3 - 2 * mask)  # smoothstep
    factor = (1.0 - amount * mask)[..., None]
    return np.clip(arr * factor, 0.0, 1.0)


# ----------------------------------------------------------------------------
# BLUR / SHARPEN
# ----------------------------------------------------------------------------
def blur(arr, radius=2.0, kind="gaussian"):
    if radius <= 0:
        return arr
    if kind == "box":
        pil = arr_to_pil(arr).filter(ImageFilter.BoxBlur(radius=float(radius)))
        return np.asarray(pil, dtype=np.float32) / 255.0
    return _gaussian(arr, radius)


def sharpen(arr, amount=1.0, radius=2.0, threshold=0.0):
    if amount <= 0:
        return arr
    pil = arr_to_pil(arr).filter(
        ImageFilter.UnsharpMask(radius=float(radius),
                                percent=int(np.clip(amount * 100, 0, 500)),
                                threshold=int(np.clip(threshold * 255, 0, 255)))
    )
    return np.asarray(pil, dtype=np.float32) / 255.0


# ----------------------------------------------------------------------------
# FILM / TEXTURE
# ----------------------------------------------------------------------------
def film_grain(arr, intensity=0.08, size=1.0, colored=False, shadow_weight=0.5, seed=0):
    if intensity <= 0:
        return arr
    rng = np.random.default_rng(seed if seed else None)
    h, w = arr.shape[:2]
    if colored:
        noise = rng.standard_normal((h, w, 3)).astype(np.float32)
    else:
        noise = rng.standard_normal((h, w, 1)).astype(np.float32)
        noise = np.repeat(noise, 3, axis=2)
    if size > 1.0:
        noise = _gaussian(noise * 0.5 + 0.5, (size - 1.0) * 1.5)
        noise = (noise - 0.5) * 2.0
    l = _lum(arr)
    weight = (1.0 - shadow_weight) + shadow_weight * (1.0 - l)
    return np.clip(arr + noise * intensity * weight, 0.0, 1.0)


def add_noise(arr, intensity=0.05, kind="gaussian", seed=0):
    if intensity <= 0:
        return arr
    rng = np.random.default_rng(seed if seed else None)
    if kind == "uniform":
        n = (rng.random(arr.shape).astype(np.float32) - 0.5) * 2.0
    else:
        n = rng.standard_normal(arr.shape).astype(np.float32)
    return np.clip(arr + n * intensity, 0.0, 1.0)


def halation(arr, threshold=0.6, radius=6.0, intensity=0.5, tint=(1.0, 0.2, 0.1)):
    if intensity <= 0:
        return arr
    l = _lum(arr)
    mask = np.clip((l - threshold) / max(1.0 - threshold, EPS), 0.0, 1.0)
    glow = _gaussian(mask, radius)
    t = np.array(tint, dtype=np.float32)
    return np.clip(arr + glow * intensity * t, 0.0, 1.0)


def scanlines(arr, intensity=0.3, thickness=1, gap=1):
    if intensity <= 0:
        return arr
    h = arr.shape[0]
    period = max(thickness + gap, 1)
    rows = (np.arange(h) % period) < thickness
    factor = np.where(rows, 1.0 - intensity, 1.0).astype(np.float32)[:, None, None]
    return np.clip(arr * factor, 0.0, 1.0)


# ----------------------------------------------------------------------------
# STYLIZE
# ----------------------------------------------------------------------------
def sepia(arr, strength=1.0):
    m = np.array([[0.393, 0.769, 0.189],
                  [0.349, 0.686, 0.168],
                  [0.272, 0.534, 0.131]], dtype=np.float32)
    sep = np.clip(arr @ m.T, 0.0, 1.0)
    return np.clip(arr * (1 - strength) + sep * strength, 0.0, 1.0)


def duotone(arr, dark=(0.05, 0.05, 0.2), light=(0.95, 0.85, 0.6), strength=1.0):
    l = _lum(arr)
    d = np.array(dark, dtype=np.float32)
    li = np.array(light, dtype=np.float32)
    duo = d + (li - d) * l
    return np.clip(arr * (1 - strength) + duo * strength, 0.0, 1.0)


def posterize(arr, levels=6):
    levels = max(int(levels), 2)
    return np.round(arr * (levels - 1)) / (levels - 1)


def pixelate(arr, block=8):
    block = max(int(block), 1)
    if block <= 1:
        return arr
    h, w = arr.shape[:2]
    small = arr_to_pil(arr).resize((max(w // block, 1), max(h // block, 1)), Image.BILINEAR)
    big = small.resize((w, h), Image.NEAREST)
    return np.asarray(big, dtype=np.float32) / 255.0


# ----------------------------------------------------------------------------
# EDGE / SKETCH / COMIC
# ----------------------------------------------------------------------------
def _sobel_mag(gray):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    gp = np.pad(gray, 1, mode="edge")
    gx = (
        kx[0, 0] * gp[:-2, :-2] + kx[0, 2] * gp[:-2, 2:]
        + kx[1, 0] * gp[1:-1, :-2] + kx[1, 2] * gp[1:-1, 2:]
        + kx[2, 0] * gp[2:, :-2] + kx[2, 2] * gp[2:, 2:]
    )
    gy = (
        ky[0, 0] * gp[:-2, :-2] + ky[0, 1] * gp[:-2, 1:-1] + ky[0, 2] * gp[:-2, 2:]
        + ky[2, 0] * gp[2:, :-2] + ky[2, 1] * gp[2:, 1:-1] + ky[2, 2] * gp[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def edge_detect(arr, strength=1.0, invert=False):
    gray = _lum(arr)[..., 0]
    mag = _sobel_mag(gray)
    mag = np.clip(mag * strength, 0.0, 1.0)
    if invert:
        mag = 1.0 - mag
    return np.repeat(mag[..., None], 3, axis=2)


def sketch(arr, strength=1.0, radius=6.0):
    gray = _lum(arr)[..., 0]
    inv = 1.0 - gray
    inv_blur = _gaussian(np.repeat(inv[..., None], 3, axis=2), radius)[..., 0]
    dodge = gray / np.clip(1.0 - inv_blur, EPS, 1.0)
    dodge = np.clip(dodge, 0.0, 1.0)
    out = np.repeat(dodge[..., None], 3, axis=2)
    return np.clip(arr * (1 - strength) + out * strength, 0.0, 1.0)


def comic(arr, levels=5, edge_strength=1.0, edge_threshold=0.25):
    base = posterize(color(arr, saturation=1.3), levels)
    gray = _lum(arr)[..., 0]
    mag = _sobel_mag(gray)
    edges = (mag > edge_threshold).astype(np.float32) * edge_strength
    edges = np.clip(edges, 0.0, 1.0)[..., None]
    return np.clip(base * (1.0 - edges), 0.0, 1.0)


# ----------------------------------------------------------------------------
# OVERLAYS - light leak / lens flare
# ----------------------------------------------------------------------------
def light_leak(arr, color_rgb=(1.0, 0.4, 0.1), angle=45.0, position=0.8,
               width=0.5, intensity=0.5, blend="screen"):
    if intensity <= 0:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = xx / max(w - 1, 1); ny = yy / max(h - 1, 1)
    a = np.radians(angle)
    proj = nx * np.cos(a) + ny * np.sin(a)
    d = np.abs(proj - position)
    grad = np.clip(1.0 - d / max(width, EPS), 0.0, 1.0)
    grad = grad * grad
    leak = grad[..., None] * np.array(color_rgb, dtype=np.float32) * intensity
    if blend == "add":
        return np.clip(arr + leak, 0.0, 1.0)
    return np.clip(1.0 - (1.0 - arr) * (1.0 - leak), 0.0, 1.0)  # screen


def lens_flare(arr, x=0.7, y=0.3, intensity=0.7, color_rgb=(1.0, 0.9, 0.7), ghosts=4):
    if intensity <= 0:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = x * (w - 1), y * (h - 1)
    out = arr.copy()
    col = np.array(color_rgb, dtype=np.float32)
    # main glow
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(w, h)
    glow = np.exp(-(d ** 2) / 0.01) * intensity
    out = np.clip(out + glow[..., None] * col, 0.0, 1.0)
    # ghosts along the line toward image center
    icx, icy = (w - 1) / 2.0, (h - 1) / 2.0
    for i in range(1, int(ghosts) + 1):
        t = i / (ghosts + 1.0)
        gx = cx + (icx - cx) * 2.0 * t
        gy = cy + (icy - cy) * 2.0 * t
        gd = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2) / max(w, h)
        gl = np.exp(-(gd ** 2) / 0.004) * intensity * 0.25
        out = np.clip(out + gl[..., None] * col, 0.0, 1.0)
    return out


# ----------------------------------------------------------------------------
# DESCREEN - FFT notch filter (kills halftone / moire / grid patterns)
# ----------------------------------------------------------------------------
def descreen(arr, strength=0.9, protect=0.04, threshold=3.0, notch_radius=3,
             mode="luma"):
    """
    Remove periodic high-frequency patterns (halftone dots / moire) via FFT.

    protect:   low-frequency radius (fraction of half-spectrum) left untouched,
               so overall image structure survives.
    threshold: how many std-devs above local mean a frequency peak must be to
               count as an artifact and get notched.
    notch_radius: pixel radius of each notch in the spectrum.
    mode:      'luma' notches the shared luminance spectrum (color-safe, fast),
               'rgb' notches each channel independently (stronger).
    """
    if strength <= 0:
        return arr

    def process_plane(plane):
        H, W = plane.shape
        F = np.fft.fftshift(np.fft.fft2(plane))
        mag = np.abs(F)
        logmag = np.log1p(mag)
        cy, cx = H // 2, W // 2
        # build protect mask (keep low freqs)
        yy, xx = np.mgrid[0:H, 0:W]
        rr = np.sqrt(((xx - cx) / (W / 2.0)) ** 2 + ((yy - cy) / (H / 2.0)) ** 2)
        protect_mask = rr < protect
        # local background via blur of the log-magnitude
        bg = _gaussian(np.repeat(logmag[..., None], 3, 2), 6.0)[..., 0]
        resid = logmag - bg
        std = resid[~protect_mask].std() + EPS
        peaks = (resid > threshold * std) & (~protect_mask)
        if not peaks.any():
            return plane
        # dilate peaks into circular notches
        notch = np.zeros((H, W), dtype=np.float32)
        ys, xs = np.where(peaks)
        nr = int(notch_radius)
        for py, px in zip(ys, xs):
            y0, y1 = max(py - nr, 0), min(py + nr + 1, H)
            x0, x1 = max(px - nr, 0), min(px + nr + 1, W)
            notch[y0:y1, x0:x1] = 1.0
        notch = _gaussian(np.repeat(notch[..., None], 3, 2), nr)[..., 0]
        notch = np.clip(notch, 0.0, 1.0) * strength
        F_filtered = F * (1.0 - notch)
        out = np.fft.ifft2(np.fft.ifftshift(F_filtered)).real
        return out.astype(np.float32)

    if mode == "rgb":
        out = np.stack([process_plane(arr[..., c]) for c in range(3)], axis=-1)
    else:
        l = _lum(arr)[..., 0]
        l_clean = process_plane(l)
        delta = (l_clean - l)[..., None]
        out = arr + delta
    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------------
# LUT (.cube) loading + trilinear apply
# ----------------------------------------------------------------------------
def parse_cube(path: str):
    size = None
    data = []
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            up = line.upper()
            if up.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
            elif up.startswith("LUT_1D_SIZE"):
                raise ValueError("1D .cube LUTs are not supported (need a 3D LUT).")
            elif up.startswith("DOMAIN_MIN"):
                domain_min = np.array(list(map(float, line.split()[1:4])), dtype=np.float32)
            elif up.startswith("DOMAIN_MAX"):
                domain_max = np.array(list(map(float, line.split()[1:4])), dtype=np.float32)
            elif up.startswith(("TITLE", "LUT_3D_INPUT_RANGE")):
                continue
            else:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        data.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
    if size is None or len(data) != size ** 3:
        raise ValueError(f"Malformed .cube (size={size}, rows={len(data)}).")
    lut = np.array(data, dtype=np.float32).reshape(size, size, size, 3)
    # .cube fastest-changing index is red; reshape gives [b][g][r]; transpose to [r][g][b]
    lut = lut.reshape(size, size, size, 3).transpose(2, 1, 0, 3)
    return lut, domain_min, domain_max


def apply_lut(arr, lut, domain_min=(0, 0, 0), domain_max=(1, 1, 1), strength=1.0):
    if strength <= 0 or lut is None:
        return arr
    size = lut.shape[0]
    dmin = np.array(domain_min, dtype=np.float32)
    dmax = np.array(domain_max, dtype=np.float32)
    norm = np.clip((arr - dmin) / np.maximum(dmax - dmin, EPS), 0.0, 1.0)
    coords = norm * (size - 1)
    i0 = np.floor(coords).astype(int)
    i1 = np.clip(i0 + 1, 0, size - 1)
    f = coords - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]

    def L(ri, gi, bi):
        return lut[ri, gi, bi]

    c00 = L(r0, g0, b0) * (1 - fr) + L(r1, g0, b0) * fr
    c01 = L(r0, g0, b1) * (1 - fr) + L(r1, g0, b1) * fr
    c10 = L(r0, g1, b0) * (1 - fr) + L(r1, g1, b0) * fr
    c11 = L(r0, g1, b1) * (1 - fr) + L(r1, g1, b1) * fr
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    out = c0 * (1 - fb) + c1 * fb
    return np.clip(arr * (1 - strength) + out * strength, 0.0, 1.0)


# ----------------------------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------------------------
def jpeg_crush(arr, quality=60):
    import io
    pil = arr_to_pil(arr)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(np.clip(quality, 1, 100)))
    buf.seek(0)
    re = Image.open(buf).convert("RGB")
    return np.asarray(re, dtype=np.float32) / 255.0
