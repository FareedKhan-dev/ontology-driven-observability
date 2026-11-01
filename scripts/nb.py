"""Small dev tool for the notebook source (not part of the notebook itself).

    uv run python scripts/nb.py run --sections 0,4 -p PROFILE SMOKE
    uv run python scripts/nb.py run --upto 6
    uv run python scripts/nb.py build            # jupytext: .py -> .ipynb
    uv run python scripts/nb.py execute -p PROFILE SMOKE
    uv run python scripts/nb.py lint
    uv run python scripts/nb.py stats
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebook" / "ontology_observability.py"
IPYNB = SRC.with_suffix(".ipynb")

HEADING = re.compile(r"^##\s+(\S+)\s+·")
AI_PHRASES = [
    "delve", "leverage", "seamless", "tapestry", "in conclusion", "it's important to note",
    "it is important to note", "robust solution", "cutting-edge", "game-changer", "unlock",
    "harness the power", "in today's", "let's dive", "dive into", "embark", "journey of",
    "realm", "pivotal", "furthermore,", "moreover,", "in summary,", "as an ai",
]


def parse_cells(path: Path = SRC) -> list[dict]:
    """Split a jupytext percent file into cells with type, tags, source and section."""
    lines = path.read_text(encoding="utf-8").splitlines()
    cells, cur, section = [], None, None
    i = 0
    if lines and lines[0].startswith("# ---"):
        i = 1
        while i < len(lines) and not lines[i].startswith("# ---"):
            i += 1
        i += 1
    for line in lines[i:]:
        if line.startswith("# %%"):
            if cur:
                cells.append(cur)
            kind = "markdown" if "[markdown]" in line else "code"
            tags = re.findall(r'"([^"]+)"', line.split("tags=", 1)[1]) if "tags=" in line else []
            cur = {"type": kind, "tags": tags, "lines": []}
            continue
        if cur is None:
            continue
        cur["lines"].append(line)
    if cur:
        cells.append(cur)
    for c in cells:
        if c["type"] == "markdown":
            c["source"] = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in c["lines"]).strip()
            m = next((HEADING.match(l) for l in c["source"].splitlines() if HEADING.match(l)), None)
            if m:
                section = m.group(1)
        else:
            c["source"] = "\n".join(c["lines"]).strip()
        c["section"] = section
    return cells


def _value(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def run(sections: list[str] | None, upto: str | None, params: dict) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    cells = parse_cells()
    order = []
    for c in cells:
        if c["section"] not in order:
            order.append(c["section"])
    if upto is not None:
        keep = set(order[: order.index(upto) + 1])
    else:
        keep = {None, "0", *(sections or [])}
    ns: dict = {"__name__": "__main__"}
    os.chdir(ROOT / "notebook")
    t_all = time.perf_counter()
    for n, c in enumerate(cells):
        if c["type"] != "code" or c["section"] not in keep or not c["source"]:
            continue
        first = c["source"].splitlines()[0][:90]
        print(f"\n── cell {n} (section {c['section']}) {first}", flush=True)
        t0 = time.perf_counter()
        exec(compile(c["source"], f"<cell {n}>", "exec"), ns)
        if "parameters" in c["tags"]:
            for k, v in params.items():
                ns[k] = v
                print(f"  [override] {k} = {v!r}")
        print(f"  [{time.perf_counter() - t0:.1f}s]", flush=True)
    print(f"\n[done] {time.perf_counter() - t_all:.1f}s")


def build() -> None:
    subprocess.run([sys.executable, "-m", "jupytext", "--to", "ipynb", str(SRC), "-o", str(IPYNB)], check=True)


def execute(params: dict) -> None:
    build()
    cmd = [sys.executable, "-m", "papermill", str(IPYNB), str(IPYNB), "--cwd", str(ROOT / "notebook"),
           "-k", "python3", "--log-output"]
    for k, v in params.items():
        cmd += ["-p", k, str(v)]
    # Python orders sets of strings differently in every process unless the hash seed is fixed; the kernel
    # inherits this, so two runs of the notebook iterate every set the same way
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONHASHSEED": "0"})


def lint() -> int:
    cells = parse_cells()
    problems = 0
    for n, c in enumerate(cells):
        text = c["source"]
        if c["type"] == "code":
            # only comments and string literals carry prose in code cells
            prose = "\n".join(re.findall(r"#.*$", text, flags=re.M) + re.findall(r"\"[^\"\n]*\"|'[^'\n]*'", text))
        else:
            prose = text
        if "—" in prose:
            problems += 1
            print(f"  cell {n} ({c['type']}): em dash")
        low = prose.lower()
        for p in AI_PHRASES:
            if p in low:
                problems += 1
                print(f"  cell {n} ({c['type']}): phrase {p!r}")
    print(f"  lint: {problems} problem(s)")
    return problems


def stats() -> None:
    cells = parse_cells()
    md = [c for c in cells if c["type"] == "markdown"]
    code = [c for c in cells if c["type"] == "code"]
    words = sum(len(re.findall(r"\b\w+\b", c["source"])) for c in md)
    lines = sum(len([l for l in c["source"].splitlines() if l.strip()]) for c in code)
    figs = sum(len(re.findall(r"\bfinish\(|\bdraw_flow\(", c["source"])) for c in code)
    printing = sum(bool(re.search(r"\bprint\(|\bfinish\(|\bdraw_flow\(|display\(", c["source"])) for c in code)
    print(f"  cells {len(cells)} (markdown {len(md)}, code {len(code)})")
    print(f"  markdown words {words:,}   code lines {lines:,}")
    print(f"  figure calls {figs}   code cells that print {printing}/{len(code)} "
          f"({printing / max(len(code), 1):.0%})")
    sections = []
    for c in cells:
        if c["section"] not in sections:
            sections.append(c["section"])
    print(f"  sections {[s for s in sections if s is not None]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "build", "execute", "lint", "stats"])
    ap.add_argument("--sections", default=None)
    ap.add_argument("--upto", default=None)
    ap.add_argument("-p", nargs=2, action="append", default=[], metavar=("NAME", "VALUE"))
    a = ap.parse_args()
    params = {k: _value(v) for k, v in a.p}
    if a.cmd == "run":
        secs = [s.strip() for s in a.sections.split(",")] if a.sections else None
        run(secs, a.upto, params)
    elif a.cmd == "build":
        build()
    elif a.cmd == "execute":
        execute(params)
    elif a.cmd == "lint":
        sys.exit(1 if lint() else 0)
    else:
        stats()


if __name__ == "__main__":
    main()
