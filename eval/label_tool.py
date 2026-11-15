"""Label one sheet in the terminal, one task at a time.

    python3 eval/label_tool.py path/to/A.csv

Keys:  y = yes   n = no   u = unsure (left blank)   c = add a comment
       b = back one task   q = save and quit
Every answer is saved at once, and the tool starts again at the first unlabelled task.
A copy of the sheet as it was is kept in <sheet>.bak (not a .csv, so it is never read as labels).
"""

import csv
import os
import shutil
import sys
import textwrap

KEYS = "[y]es  [n]o  [u]nsure  [c]omment  [b]ack  [q]uit"


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def save(path, fields, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def show(i, rows):
    row = rows[i]
    done = sum(1 for r in rows if r["label"].strip())
    os.system("clear")
    print(f"Task {i + 1} of {len(rows)}   ({done} labelled)   kind: {row['kind']}\n")
    print("QUESTION")
    print(textwrap.fill(row["question"], 90))
    print("\nCONTEXT")
    for line in row["context"].splitlines():
        print(textwrap.fill(line, 90) if line else "")
    if row["label"].strip() or row["note"].strip():
        print(f"\n(current label: {row['label'] or 'blank'}; note: {row['note'] or '-'})")
    print("\n" + KEYS)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    fields, rows = load(path)
    if not os.path.exists(path + ".bak"):
        shutil.copy(path, path + ".bak")
    i = next((k for k, r in enumerate(rows) if not r["label"].strip()), len(rows))
    while True:
        if i >= len(rows):
            blank = sum(1 for r in rows if not r["label"].strip())
            print(f"\nEnd of the sheet. {len(rows) - blank} labelled, {blank} left blank. Saved to {path}.")
            print("Press b to go back, or q to quit.")
        else:
            show(i, rows)
        key = input("> ").strip().lower()
        if key == "q":
            save(path, fields, rows)
            print(f"Saved to {path}.")
            return
        if key == "b":
            i = max(0, i - 1)
            continue
        if i >= len(rows):
            continue
        if key in ("y", "n"):
            rows[i]["label"] = "yes" if key == "y" else "no"
        elif key == "u":
            rows[i]["label"] = ""
        elif key == "c":
            rows[i]["note"] = input("comment: ").strip()
            save(path, fields, rows)
            continue
        else:
            continue
        save(path, fields, rows)
        i += 1


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Every answer so far is saved.")
