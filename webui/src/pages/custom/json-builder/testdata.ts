/** Shared test data: a draft and its prompt as upstream compiles it (tests only, not bundled). */

// Items through a helper: a literal label property in pages/ reads as a chrome string to translate (locales.test.ts).
const item = (label: string, caption: string, bbox: number[]) => ({ label, caption, bbox })

/** A draft as the builder writes it ("Save JSON"). */
export const DRAFT = {
  width: 1024,
  height: 768,
  bbox_coord_space: "abs",
  bbox_format: "xyxy",
  summary: { meta: ["photo", "soft light"], canvas: "single", caption: "A cat (orange) on a sofa" },
  items: [
    item("character", "an orange cat", [100.4, 200, 500.2, 700]),
    { ...item("text", "a neon sign", [600, 50, 1000, 150]), text: "OPEN (24h)" },
  ],
}

/** What upstream's compile.js returns for {@link DRAFT} (recorded with Forge Neo c6c2b43d). */
export const COMPILED =
  '{"summary":{"meta":["photo","soft light"],"canvas":"single","caption":"A cat \\(orange\\) on a sofa"},' +
  '"items":[{"label":"character","caption":"an orange cat","bbox":[-184,-412,316,-11]},' +
  '{"label":"text","caption":"a neon sign","bbox":[-334,88,-234,488],"text":"OPEN \\(24h\\)"}],' +
  '"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}'
