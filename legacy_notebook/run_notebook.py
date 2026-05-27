#!/usr/bin/env python3
"""Execute a Jupyter notebook as a plain script - no Jupyter server, no UI.

Every code cell is run in order inside one shared namespace, streaming output
to the console AND a log file in real time. It hardens the naive
``exec(cell.source)`` approach against the things that break it in practice:

* IPython magics are handled - ``%pip``/``!pip`` are skipped (or installed with
  --install), ``%matplotlib`` and other line/cell/shell magics are stripped so
  they don't raise SyntaxError.
* matplotlib runs headless (Agg backend) and ``plt.show()`` is patched to save
  each figure as a PNG instead of blocking on a GUI window.
* the working directory is switched to the notebook's folder, so its relative
  imports (``from kg_data import ...``) and data paths (``data/dummy_kg``)
  resolve exactly as they do inside Jupyter.
* stdout / stderr / log are forced to UTF-8 so the notebook's box-drawing and
  math characters (─ ═ ≈) don't crash the Windows cp1252 console.

Usage - run with the PROJECT's interpreter, not the Windows Store python stub::

    C:\\Users\\thari\\miniconda3\\python.exe run_notebook.py
    C:\\Users\\thari\\miniconda3\\python.exe run_notebook.py src/knowledge_graph_clone.ipynb
    ...  --install            # actually run %pip / !pip lines (default: skip)
    ...  --continue-on-error  # keep going after a failing cell (default: stop)
    ...  --outdir runs/foo    # where figures + run.log are written

Exit code is non-zero if any cell raised.
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Windows + conda: torch and MKL each bundle their own OpenMP runtime
# (libiomp5md.dll). Loading both hard-aborts the process with "OMP Error #15"
# (exit code 3) before Python can even raise. Jupyter/VSCode typically set this
# for you; we must set it ourselves, BEFORE numpy/torch/matplotlib load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Default notebook, resolved relative to THIS file so it works from any cwd.
DEFAULT_NOTEBOOK = Path(__file__).resolve().parent / "src" / "knowledge_graph_clone.ipynb"

# Cell magics whose body is plain Python - we drop the magic line and run the
# rest. Anything else (%%bash, %%html, ...) means "skip the whole cell".
PASSTHROUGH_CELL_MAGICS = {"%%time", "%%timeit", "%%capture", "%%prun"}


# --------------------------------------------------------------------------- #
# Notebook loading
# --------------------------------------------------------------------------- #
def load_code_cells(notebook_path: Path):
    """Return [(notebook_cell_index, source_str), ...] for code cells only.

    Uses nbformat if available (handles version quirks); falls back to reading
    the .ipynb as plain JSON so the script has zero hard dependencies.
    """
    try:
        import nbformat

        nb = nbformat.read(str(notebook_path), as_version=4)
        cells = nb.cells
        get_type = lambda c: c.cell_type
        get_src = lambda c: c.source
    except Exception:
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = nb["cells"]
        get_type = lambda c: c["cell_type"]
        get_src = lambda c: "".join(c["source"])

    return [(i, get_src(c)) for i, c in enumerate(cells) if get_type(c) == "code"]


# --------------------------------------------------------------------------- #
# Magic / shell handling
# --------------------------------------------------------------------------- #
def _rewrite_line(line: str, install: bool):
    """Rewrite one source line. Returns (new_line, note_or_None).

    Magic/shell lines become a `pass  # ...` comment so line numbers stay aligned
    for tracebacks. pip lines optionally become a real install call.
    """
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    pip = re.match(r"^[%!]\s*pip\b(.*)$", stripped)
    if pip:
        args = pip.group(1).strip()
        if install:
            return f"{indent}__runner_pip_install({args!r})", f"pip install: {args}"
        return f"{indent}pass  # [runner] skipped (use --install): {stripped}", \
            f"skipped pip: {stripped}"

    if stripped.startswith("%"):
        return f"{indent}pass  # [runner] skipped magic: {stripped}", f"skipped magic: {stripped}"
    if stripped.startswith("!"):
        return f"{indent}pass  # [runner] skipped shell: {stripped}", f"skipped shell: {stripped}"
    return line, None


def preprocess_source(source: str, install: bool):
    """Strip/translate magics in a code cell. Returns (clean_source, notes)."""
    lines = source.splitlines()
    notes = []

    # Cell magic (%%foo) only matters as the first non-blank line.
    first_idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first_idx is not None and lines[first_idx].strip().startswith("%%"):
        magic = lines[first_idx].strip().split()[0]
        if magic in PASSTHROUGH_CELL_MAGICS:
            notes.append(f"stripped cell magic {magic}")
            lines.pop(first_idx)
        else:
            return "", [f"skipped whole cell (cell magic {magic})"]

    out = []
    for line in lines:
        new_line, note = _rewrite_line(line, install)
        out.append(new_line)
        if note:
            notes.append(note)
    return "\n".join(out), notes


def __runner_pip_install(args: str):
    """Injected into the cell namespace so translated %pip lines can run."""
    cmd = [sys.executable, "-m", "pip"] + args.split()
    print(f"[runner] running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


# --------------------------------------------------------------------------- #
# Headless matplotlib
# --------------------------------------------------------------------------- #
def setup_headless_matplotlib(outdir: Path, log):
    """Force the Agg backend and make plt.show() save figures to PNG files."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # matplotlib not installed - cells may still run
        log(f"[runner] matplotlib unavailable ({exc}); plots will be skipped if reached")
        return

    counter = {"n": 0}

    def _save_show(*_args, **_kwargs):
        for num in plt.get_fignums():
            counter["n"] += 1
            path = outdir / f"figure_{counter['n']:02d}.png"
            try:
                plt.figure(num).savefig(path, dpi=120, bbox_inches="tight")
                log(f"[runner] saved figure -> {path}")
            except Exception as exc:
                log(f"[runner] could not save figure {num}: {exc}")
        plt.close("all")

    plt.show = _save_show


# --------------------------------------------------------------------------- #
# Output tee (real-time console + log file)
# --------------------------------------------------------------------------- #
class _Tee(io.TextIOBase):
    """Write to several text streams at once, flushing each write."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_notebook(notebook_path: Path, outdir: Path, install: bool, continue_on_error: bool):
    notebook_path = notebook_path.resolve()
    notebook_dir = notebook_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    # UTF-8 everywhere so box-drawing/math chars never crash the console.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    real_stdout, real_stderr = sys.stdout, sys.stderr
    log_path = outdir / "run.log"
    log_fh = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, file=real_stdout, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    code_cells = load_code_cells(notebook_path)
    log(f"Notebook : {notebook_path}")
    log(f"Workdir  : {notebook_dir}")
    log(f"Python   : {sys.executable}")
    log(f"Outdir   : {outdir}")
    log(f"Started  : {datetime.now():%Y-%m-%d %H:%M:%S}  ({len(code_cells)} code cells)")

    # Execute as if launched from the notebook's folder, with it on sys.path
    # (this is what makes `from kg_data import ...` and `datasets/...` resolve).
    os.chdir(notebook_dir)
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))

    setup_headless_matplotlib(outdir, log)

    namespace = {
        "__name__": "__main__",
        "__file__": str(notebook_path),
        "__runner_pip_install": __runner_pip_install,
    }

    failures = 0
    start_all = datetime.now()

    for ordinal, (cell_index, raw_source) in enumerate(code_cells, start=1):
        clean_source, notes = preprocess_source(raw_source, install)

        log("\n" + "=" * 72)
        log(f"CELL {ordinal}/{len(code_cells)}  (notebook index {cell_index})")
        log("=" * 72)
        for note in notes:
            log(f"[runner] {note}")
        log("--- source " + "-" * 61)
        log(raw_source.rstrip())
        log("--- output " + "-" * 61)

        if not clean_source.strip():
            log("[runner] (nothing to execute)")
            continue

        # Tee real stdout/stderr -> console + log for the cell's own prints.
        sys.stdout = _Tee(real_stdout, log_fh)
        sys.stderr = _Tee(real_stderr, log_fh)
        cell_start = datetime.now()
        try:
            compiled = compile(clean_source, f"<cell {ordinal}>", "exec")
            exec(compiled, namespace)
            ok = True
        except Exception:
            ok = False
            tb = traceback.format_exc()
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr
        elapsed = (datetime.now() - cell_start).total_seconds()

        if ok:
            log(f"[runner] OK in {elapsed:.2f}s")
        else:
            failures += 1
            log(tb)
            log(f"[runner] FAILED in {elapsed:.2f}s")
            if not continue_on_error:
                log("\n[runner] stopping (use --continue-on-error to keep going)")
                break

    total = (datetime.now() - start_all).total_seconds()
    log("\n" + "=" * 72)
    log(f"Finished : {datetime.now():%Y-%m-%d %H:%M:%S}  in {total:.1f}s, {failures} failure(s)")
    log(f"Log saved: {log_path}")
    log_fh.close()
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a Jupyter notebook without Jupyter.")
    parser.add_argument("notebook", nargs="?", default=str(DEFAULT_NOTEBOOK),
                        help=f"path to the .ipynb (default: {DEFAULT_NOTEBOOK})")
    parser.add_argument("--outdir", default=None,
                        help="output dir for figures + run.log (default: ./notebook_run_<timestamp>)")
    parser.add_argument("--install", action="store_true",
                        help="actually execute %%pip / !pip lines (default: skip them)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="keep running after a failing cell (default: stop)")
    args = parser.parse_args(argv)

    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        print(f"error: notebook not found: {notebook_path}", file=sys.stderr)
        return 2

    if args.outdir:
        outdir = Path(args.outdir).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = (Path.cwd() / f"notebook_run_{stamp}").resolve()

    return run_notebook(notebook_path, outdir, args.install, args.continue_on_error)


if __name__ == "__main__":
    sys.exit(main())
