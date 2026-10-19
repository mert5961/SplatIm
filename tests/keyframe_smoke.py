"""Real SAM smoke check, or isolated full-UI fixture; never edits the source job.

python tests/keyframe_smoke.py <completed-job> [--serve]
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))
import VideoSubjectSplat as pipeline
from SubjectKeyframes import save_keyframe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    source = json.loads((args.job / "job.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="splatim-keyframes-") as folder:
        job = Path(folder)
        frames = job / "sam2_frames"
        frames.mkdir()
        for index, original in enumerate(np.linspace(0, len(source["frames"])-1, 8).round().astype(int)):
            with Image.open(source["frames"][original]) as image:
                image.thumbnail((320, 320))
                size = image.size
                image.convert("RGB").save(frames / f"{index:05d}.jpg")
            if 1 <= index <= 6:
                with Image.open(args.job / "sam2_raw" / f"{original:05d}.png") as mask:
                    mask = np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST)) > 127
                save_keyframe(job, index, mask, "mask")
        state = dict(job=str(job), frames=[str(p) for p in sorted(frames.glob("*.jpg"))])
        (job / "job.json").write_text(json.dumps(state), encoding="utf-8")
        editor, note, context, summary = pipeline.load_target_frame(state, 1)
        if args.serve:
            import gradio as gr
            import SharpSplatUI as ui
            with ui.demo:
                ui.demo.load(lambda: (state, gr.update(minimum=0, maximum=7, value=1),
                                     editor, note, context, summary),
                             outputs=[ui.video_subject_state, ui.target_frame, ui.target_editor,
                                      ui.target_note, ui.target_context, ui.keyframe_status])
            ui.demo.launch(server_name="127.0.0.1", server_port=7872, inbrowser=False,
                           css=ui.PROGRESS_CSS, allowed_paths=[str(job)])
        else:
            result = pipeline.run_sam2(state, 1, editor, context,
                                      progress=lambda value, desc=None: print(desc, flush=True))
            print(result[3], flush=True)
            print((job / "sam2-keyframes-used.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
