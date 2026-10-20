"""Interleaved overhead with a non-browser WebSocket client (receives and discards); a diagnostic of
the 2026-10-20 verify stage (docs/research/ui-performance.md). NO_CLIENT=1 toggles the mute file
without connecting. Usage: ws_interleave.py <size> <jobs> <window s>, against serve_traced.py on 7861."""
import json, os, sys, threading, time, urllib.request, statistics as st
import websockets.sync.client as wsc

BASE = "http://127.0.0.1:7861"; MUTE = "/Volumes/Remi/Anima/remi-perf/fx/mute"; TRACE = "/Volumes/Remi/Anima/remi-perf/fx/trace.jsonl"
SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 512; JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else 2; WIN = float(sys.argv[3]) if len(sys.argv) > 3 else 15
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def req(m, p, b=None):
    r = urllib.request.Request(BASE + p, data=None if b is None else json.dumps(b).encode(), method=m, headers={"Content-Type": "application/json"})
    return json.loads(op.open(r, timeout=60).read())
frames = [0]
def recv():
    with wsc.connect("ws://127.0.0.1:7861/api/v2/ws", origin="http://127.0.0.1:7861", max_size=None) as ws:
        for msg in ws:
            if isinstance(msg, bytes): frames[0] += 1
if os.environ.get("NO_CLIENT") != "1":
    threading.Thread(target=recv, daemon=True).start()
time.sleep(2)
def trace(since):
    out = []
    for line in open(TRACE):
        e = json.loads(line)
        if e["e"] == "step" and e["t"] >= since: out.append(e)
    return out
allpairs = []
for j in range(JOBS):
    if os.path.exists(MUTE): os.remove(MUTE)
    job = req("POST", "/api/v2/jobs", {"tab": "txt2img", "core": {"prompt": "masterpiece, best quality, 1girl, silver hair", "negative_prompt": "worst quality", "width": SIZE, "height": SIZE, "cfg_scale": 1}, "scripts": {"sampler": {"sampler_name": "Euler", "scheduler": "Simple", "steps": 150}, "seed": {"seed": 12345}}})["job_id"]
    t0 = time.time()
    while not trace(t0): time.sleep(0.5)
    order = ["on", "off", "off", "on"]; windows = []; i = 0
    while req("GET", f"/api/v2/jobs/{job}")["status"] in ("queued", "running"):
        c = order[i % 4]; i += 1
        if c == "off": open(MUTE, "w").close()
        elif os.path.exists(MUTE): os.remove(MUTE)
        s = time.time()
        while time.time() - s < WIN:
            time.sleep(0.2)
        windows.append([c, s, time.time(), []])
    if os.path.exists(MUTE): os.remove(MUTE)
    tr = trace(t0)
    for a, b in zip(tr, tr[1:]):
        if b["step"] != a["step"] + 1: continue
        for w in windows:
            if a["t"] >= w[1] and b["t"] <= w[2]: w[3].append((b["t"] - a["t"]) * 1000)
    pairs = []
    for k in range(0, len(windows) - 1, 2):
        x, y = windows[k], windows[k + 1]
        if not x[3] or not y[3]: continue
        on, off = (x, y) if x[0] == "on" else (y, x)
        pairs.append((st.mean(on[3]) / st.mean(off[3]) - 1) * 100)
    allpairs += pairs
    print(f"job {j+1}: {len(pairs)} pairs mean {st.mean(pairs):.2f} median {st.median(pairs):.2f}; frames so far {frames[0]}", flush=True)
n = len(allpairs); m = st.mean(allpairs); se = st.stdev(allpairs) / n ** 0.5
print(f"pooled: n {n} mean {m:.2f} % se {se:.2f} upper95 {m + 1.68 * se:.2f} median {st.median(allpairs):.2f}")
json.dump({"pairs": allpairs}, open(f"/Volumes/Remi/Anima/remi-verify/ws_interleave_{SIZE}_{os.environ.get('TAG', 'x')}.json", "w"))
