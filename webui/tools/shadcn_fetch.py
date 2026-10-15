"""Fetch shadcn/ui (base-nova) registry items with curl (Node cannot use the local proxy)."""

import json
import os
import re
import subprocess
import sys

STYLE = "base-nova"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seen, npm = set(), set()


def get(name):
    url = f"https://ui.shadcn.com/r/styles/{STYLE}/{name}.json"
    out = subprocess.run(["curl", "-sf", "--max-time", "30", url], capture_output=True, text=True)
    if out.returncode:
        print("missing", name)
        return None
    return json.loads(out.stdout)


def target(path):
    path = path.replace(f"registry/{STYLE}/", "")
    if path.startswith("ui/"):
        return f"src/components/ui/{path[3:]}"
    if path.startswith("lib/"):
        return f"src/lib/{path[4:]}"
    if path.startswith("hooks/"):
        return f"src/hooks/{path[6:]}"
    return f"src/{path}"


def add(name):
    name = name.rsplit("/", 1)[-1].removesuffix(".json")
    if name in seen:
        return
    seen.add(name)
    item = get(name)
    if not item:
        return
    npm.update(item.get("dependencies", []))
    for f in item.get("files", []):
        dst = os.path.join(ROOT, f.get("target") or target(f["path"]))
        content = re.sub(r"@/registry/[\w-]+/(ui|lib|hooks)/", lambda m: f"@/{'components/ui' if m.group(1) == 'ui' else m.group(1)}/", f["content"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as out:
            out.write(content)
    for dep in item.get("registryDependencies", []):
        add(dep)


for n in sys.argv[1:]:
    add(n)
print("NPM:", " ".join(sorted(d for d in npm if d not in ("cn",))))
