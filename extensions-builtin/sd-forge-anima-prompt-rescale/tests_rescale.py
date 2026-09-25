"""Unit tests for the prompt-schedule rescaler (verified Forge neo semantics)."""
import sys
sys.path.insert(0, "/tmp/anima_work/build/sd-forge-anima-prompt-rescale")
from lib_anima_rescale.rescale import (
    executed_steps, remap_when, find_schedule_whens, rescale_prompt_schedule,
)
fails = 0
def check(name, got, want):
    global fails
    ok = got == want
    if not ok: fails += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n        got : {got!r}\n        want: {want!r}")

print("== executed_steps (matches Forge setup_img2img_steps) ==")
check("35 @ 0.5 -> 17", executed_steps(35, 0.5), 17)
check("35 @ 1.0 -> 34 (0.999 cap)", executed_steps(35, 1.0), 34)
check("35 @ 0.1 -> 3", executed_steps(35, 0.1), 3)
check("min 1", executed_steps(35, 0.001), 1)

print("\n== remap_when ==")
check("15/35 @ d0.5 t35 -> 7", remap_when(15/35, 35, 0.5), 7)
check("15/35 @ d1 t35 -> 15", remap_when(15/35, 35, 1.0), 15)
check("15/35 @ d0.1 t35 -> 1", remap_when(15/35, 35, 0.1), 1)
check("0.9 @ d0.5 t35 -> 15", remap_when(0.9, 35, 0.5), 15)

print("\n== rescale_prompt_schedule (absolute) ==")
check("headline 15 -> 7", rescale_prompt_schedule("[@guweiz:@wlop:15]", 35, 35, 0.5), "[@guweiz:@wlop:7]")
check("d1 same steps 15->15", rescale_prompt_schedule("[a:b:15]", 35, 35, 1.0), "[a:b:15]")
check("fraction author 0.4286 -> 7", rescale_prompt_schedule("[a:b:0.4286]", 35, 35, 0.5), "[a:b:7]")
check("steps 35->50 @ d1", rescale_prompt_schedule("[a:b:15]", 35, 50, 1.0), "[a:b:21]")
check("neg 10 -> 5", rescale_prompt_schedule("[blurry:sharp:10]", 35, 35, 0.5), "[blurry:sharp:5]")

print("\n== nested (grounded) ==")
def gt(when, S, T, d):
    f = when if when < 1 else when / S
    te = max(1, int(min(d, 0.999) * T))
    return max(1, min(te, round(f * te)))
inner = gt(0.3, 35, 35, 0.5); outer = gt(20, 35, 35, 0.5)
print(f"   inner 0.3 -> {inner}, outer 20 -> {outer}")
check("nested rescale", rescale_prompt_schedule("[a:[b:c:0.3]:20]", 35, 35, 0.5), f"[a:[b:c:{inner}]:{outer}]")

print("\n== fraction output ==")
check("fraction 15 -> 0.2", rescale_prompt_schedule("[a:b:15]", 35, 35, 0.5, as_fraction=True), "[a:b:0.2]")

print("\n== robustness ==")
check("alternation ignored", find_schedule_whens("[a|b|c]"), [])
check("attention weight ignored", find_schedule_whens("(cat:1.3)"), [])
check("scheduled w/ weighted operand", find_schedule_whens("[(cat:1.3):dog:10]"), [(15, 17, 10.0)])
check("no schedule unchanged", rescale_prompt_schedule("a plain prompt", 35, 35, 0.5), "a plain prompt")
check("alternation untouched", rescale_prompt_schedule("[a|b|c]", 35, 35, 0.5), "[a|b|c]")
check("weights untouched", rescale_prompt_schedule("(masterpiece:1.4)", 35, 35, 0.5), "(masterpiece:1.4)")
check("mixed", rescale_prompt_schedule("(best:1.2), [a:b:15], [x|y]", 35, 35, 0.5), "(best:1.2), [a:b:7], [x|y]")
check("malformed safe", rescale_prompt_schedule("[broken:::]", 35, 35, 0.5), "[broken:::]")

print("\nDONE. failures =", fails)
sys.exit(1 if fails else 0)
