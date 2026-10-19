"""Manual browser fixture: exercise real Gradio event transport without a model job."""
import sys
import time
from pathlib import Path
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
from ProgressUI import CSS, present_job


def run_sam2(progress):
    progress(.05, desc="UI fixture: waiting for telemetry")
    time.sleep(4)
    for count in (1, 2, 3):
        progress(count / 4, desc=f"UI fixture frames: {count}/4")
        time.sleep(4)
    return "Test completed", "### Test stage\n\nComplete", "No model or files changed."


def run_fixture():
    yield from present_job(run_sam2, [], label="UI test — no model running",
                           count=3, status_index=0, board_index=1, log_index=2)


with gr.Blocks() as demo:
    gr.Markdown("# Progress display test (no models)")
    with gr.Tab("Test stage"):
        button = gr.Button("Run display test")
        status, board, log = gr.Markdown("Ready"), gr.Markdown("Waiting"), gr.Textbox()
    with gr.Tab("Unrelated panel"):
        gr.Markdown("This panel must stay idle.")
    button.click(run_fixture, outputs=[status, board, log], show_progress="minimal", show_progress_on=board)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7871, inbrowser=False, css=CSS)
