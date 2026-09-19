"""Presentation adapter for existing synchronous pipeline functions.

The backend and its model commands remain unchanged. This generator observes
existing progress callbacks/logs and yields stage-local UI updates.
"""
from __future__ import annotations

import html
import inspect
import queue
import re
import threading
import time
from contextvars import copy_context
from pathlib import Path

import gradio as gr

CSS = """
.stage-meter {height:8px;overflow:hidden;background:#333344;border-radius:6px;margin:12px 0}
.stage-meter span {display:block;height:100%;background:#7567ff;border-radius:6px}
.stage-meter.indeterminate span {width:32%;animation:stage-work 1.4s ease-in-out infinite}
@keyframes stage-work {0% {transform:translateX(-110%)} 100% {transform:translateX(420%)}}
"""

# A disconnected generator must not allow a second GPU stage to overlap the
# still-running synchronous backend worker.
JOB_LOCK = threading.Lock()


class Observation:
    def __init__(self):
        self.lock = threading.Lock()
        self.description = "Starting…"

    def __call__(self, value, desc=None, **kwargs):
        with self.lock:
            if desc:
                self.description = str(desc)


def progress_panel(label, description, elapsed, fraction=None):
    label, description = html.escape(label), html.escape(description)
    if fraction is None:
        meter = '<div class="stage-meter indeterminate" role="progressbar" aria-label="Working; percentage unavailable"><span></span></div>'
        detail = "Working · this phase does not report a percentage"
    else:
        percent = max(0, min(100, round(100 * fraction)))
        meter = f'<div class="stage-meter" role="progressbar" aria-valuenow="{percent}" aria-valuemin="0" aria-valuemax="100"><span style="width:{percent}%"></span></div>'
        detail = f"{percent}% of reported frames"
    return f"### {label}\n\n**Running · {int(elapsed)}s elapsed**\n\n{description}\n\n{meter}\n\n{detail}"


def _read_log(path, offset):
    if path is None or not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(offset, path.stat().st_size - 24000))
        return stream.read().decode("utf-8", errors="replace")


def present_job(fn, args, *, label, count, status_index, board_index, log_index):
    observed = Observation()
    results = queue.Queue()
    state = args[0] if args and isinstance(args[0], dict) else {}
    log_path = Path(state["job"]) / "pipeline.log" if state.get("job") else None
    offset = log_path.stat().st_size if log_path and log_path.is_file() else 0
    start = time.monotonic()

    def work():
        try:
            with JOB_LOCK:
                result = fn(*args, progress=observed)
                if inspect.isgenerator(result):
                    for item in result:
                        results.put(("result", item))
                else:
                    results.put(("result", result))
        except Exception as error:
            results.put(("error", error))
        finally:
            results.put(("done", None))

    context = copy_context()
    worker = threading.Thread(target=lambda: context.run(work), daemon=True)
    worker.start()
    completed = False
    while True:
        try:
            kind, value = results.get(timeout=.5)
            if kind == "done":
                break
            if kind == "result":
                completed = True
                yield value
            else:
                completed = True
                outputs = [gr.skip() for _ in range(count)]
                outputs[status_index] = f"**Stopped:** {html.escape(str(value))}"
                outputs[board_index] = f"### {html.escape(label)}\n\n❌ **Stopped**\n\n{html.escape(str(value))}"
                yield tuple(outputs)
        except queue.Empty:
            if completed:
                continue
            with observed.lock:
                description = observed.description
            log = _read_log(log_path, offset)
            fraction = None
            if fn.__name__ == "run_sam2":
                match = re.search(r"(\d+)/(\d+)$", description)
                if match and int(match[2]):
                    fraction = int(match[1]) / int(match[2])
            elif fn.__name__ == "extract_subject_splat":
                matches = re.findall(r"(?m)^(\d+)/(\d+)\s+\S+\s+fg>5:", log)
                if matches:
                    done, total = map(int, matches[-1])
                    if total:
                        fraction = done / total
                        description = f"Checking mask evidence: {done}/{total} camera views"
            outputs = [gr.skip() for _ in range(count)]
            outputs[board_index] = progress_panel(label, description, time.monotonic()-start, fraction)
            outputs[status_index] = "Processing. Updates appear here while this stage runs."
            if log:
                outputs[log_index] = log
            yield tuple(outputs)
