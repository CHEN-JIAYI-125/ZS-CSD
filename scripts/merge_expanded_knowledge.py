#!/usr/bin/env python3
"""Merge _expanded_chunk*.json into target_knowledge_template.json, then remove temp files."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TEMPLATE_PATH = DATA / "target_knowledge_template.json"

TEMP_GLOBS = (
    "_chunk*.json",
    "_expanded_chunk*.json",
    "_targets_need_expand.json",
    "_need_rewrite.txt",
    "_claim_templated.txt",
)


def main() -> None:
    with TEMPLATE_PATH.open(encoding="utf-8") as f:
        template = json.load(f)

    expanded = {}
    for path in sorted(DATA.glob("_expanded_chunk*.json")):
        chunk = json.load(path.open(encoding="utf-8"))
        expanded.update(chunk)

    merged = 0
    skipped = []
    for target, fields in expanded.items():
        if target not in template:
            skipped.append(f"unknown target: {target}")
            continue
        entry = template[target]
        for key in ("description", "favor_reason", "against_reason"):
            if key in fields and str(fields[key]).strip():
                entry[key] = fields[key].strip()
        entry.pop("neutral_hint", None)
        merged += 1

    with TEMPLATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
        f.write("\n")

    removed = []
    for pattern in TEMP_GLOBS:
        for path in DATA.glob(pattern):
            path.unlink()
            removed.append(path.name)

    pyc = DATA / "__pycache__" / "_gen_expanded_chunk3.cpython-313.pyc"
    if pyc.exists():
        pyc.unlink()
        removed.append(str(pyc.relative_to(ROOT)))

    print(f"Merged expanded entries: {merged}")
    print(f"Template total: {len(template)}")
    print(f"Removed temp files: {len(removed)}")
    if skipped:
        print("Skipped:")
        for line in skipped:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
