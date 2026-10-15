/** The text a loaded prompt file puts in the box: its lines, each stripped (scripts/prompts_from_file.py did this on upload). */
export function fileLines(text: string): string {
  // U+FFFD: bytes that were not UTF-8 (Python decoded with errors="ignore")
  return text
    .replaceAll("\uFFFD", "")
    .split("\n")
    .map((line) => line.trim())
    .join("\n")
}
