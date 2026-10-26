# Independent scoring history

Each round, independent reviewer agents scored the rewrite against
[scorecard.md](scorecard.md), one reviewer per dimension, reproducing every
claim themselves (unverifiable counts as a failure). Live dimensions used the
real Anima Turbo model and a browser. Acceptance target: every dimension at
least 90.

| Dimension | R1 | R2 | R3 | R4 | R5 (final) |
|---|---|---|---|---|---|
| 1 Gradio removal and architecture | 67 | 89 | 91 | 89 | **99** (rescored after the `__refresh__` layering fix) |
| 2 Built-in extension adaptation | 85 | 97 | 96 | 94 | **93** |
| 3 Generation correctness | 82 | 93 | 98 | 98 | **90** |
| 4 Feature completeness (SPA) | 62 | 89 | 87 | 92 | **94** |
| 5 Performance | 75 | 66 | 94 | 86 | **92** |
| 6 Visual design and UX / 7 Accessibility and i18n | — | 80 | 92 | 91 | **92** (lower of the two) |
| 8 Tests and quality gates | 77 | 91 | 97 | 96 | **94** |
| 9 Security | 88 | 81 | 94 | 94 | **97** |
| 10 Documentation and upstream absorption | 73 | 85 | 93 | 96 | **91** (rescored after doc fixes) |

Round 5 was scored at commit `63614d3f`; dimensions 1 and 10 were rescored at
`7d00df65` after the layering and documentation fixes.

## Remaining known deductions (round 5)

- Generation correctness: the LoRA-dependent Anima features need a LoRA in
  `models/Lora` for a reviewer to observe them (`tools/anima_test_loras.py`
  writes test LoRAs; `tools/smoke_real.py --features` uses them); the
  refiner's CFG and the prompt-schedule rescaler's switch are not written to
  the infotext, so pasting does not restore them.
- Gradio removal: the dev-only legacy UI recorder (`tools/record_legacy_ui.py`,
  `tools/legacy_ui_recordings.py`) still builds Gradio controls in a separate
  Gradio-era interpreter to record porting fixtures.
- See each round's reviewer reports for the full lists.
