#!/usr/bin/env python3
# add_given_best.py  – append “Best Values” | “Given Best” column pair

from pathlib import Path

SRC  = Path("../alpha=0.75/results.csv")            # <- original file
DEST = Path("../alpha=0.75/updated_results.csv")    # <- output file

# Published best-cut values
best_values = {
    "g1": 12078, "g2": 12084, "g3": 12077,
    "g11":  627, "g12":  621, "g13":  645,
    "g14": 3187, "g15": 3169, "g16": 3172,
    "g22": 14123,"g23": 14129,"g24": 14131,
    "g32": 1560, "g33": 1537, "g34": 1541,
    "g35": 8000, "g36": 7996, "g37": 8009,
    "g43": 7027, "g44": 7022, "g45": 7020,
    "g48": 6000, "g49": 6000, "g50": 5988,
}

# ---------------------------------------------------------------------------
# 1.  Read whole CSV verbatim
# ---------------------------------------------------------------------------
lines = SRC.read_text(encoding="utf-8").splitlines()

if len(lines) < 2:
    raise RuntimeError("CSV must have at least two rows (description + header).")

# ---------------------------------------------------------------------------
# 2.  Extend row 0 and row 1
# ---------------------------------------------------------------------------
lines[0] = f"{lines[0]},Best Values"
lines[1] = f"{lines[1]},Given Best"

# ---------------------------------------------------------------------------
# 3.  Process each data row (starting from line index 2)
# ---------------------------------------------------------------------------
for i in range(2, len(lines)):
    cells = lines[i].split(",")

    graph_id = cells[0].strip()      # first column holds g1, g2, …

    value = best_values.get(graph_id, "n/a")
    cells.append(str(value))

    lines[i] = ",".join(cells)

# ---------------------------------------------------------------------------
# 4.  Write the new file
# ---------------------------------------------------------------------------
DEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"✓  Updated CSV written to {DEST.resolve()}")
