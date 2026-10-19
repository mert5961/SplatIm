"""Render recovery diagnostics on existing registered views (not a held-out test)."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from plyfile import PlyData
from scipy.ndimage import binary_erosion, binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "outputs"))
from SubjectRecovery import score_uncertain, export_recovery
from splat_alpha_render import load_ply, load_cameras, gauss_cov3d
from splat_rgb_render import render_rgb_alpha


def compare(job, minimum_score=.9, stems=None, height=960, quick=False):
    job = Path(job).resolve()
    scores = score_uncertain(job)
    result = export_recovery(job, minimum_score, 5000, scores)
    out = Path(result["final"]).parent / "comparison"
    out.mkdir(exist_ok=True)
    cams, views = load_cameras(job / "sparse_txt")
    lookup = {view["stem"]: view for view in views}
    stems = stems or ["00000", "00020", "00040", "00060", "00080", "00100"]
    paths = {
        "Original": job / "subject-extraction/keep.ply",
        "Uncertain": job / "subject-extraction/uncertain.ply",
        "Recovered": Path(result["final"]),
        "Full scene": job / "scene.ply",
    }
    if quick:
        paths = {name: paths[name] for name in ("Original", "Recovered")}
    models = {}
    for name, path in paths.items():
        xyz, scales, rotations, opacity = load_ply(path)
        data = PlyData.read(path)["vertex"].data
        color = np.clip(.5 + .28209479177387814 * np.column_stack([data[f"f_dc_{i}"] for i in range(3)]), 0, 1)
        models[name] = (xyz, gauss_cov3d(scales, rotations), opacity, color)
    sheet = Image.new("RGB", ((len(paths)+1) * 240, len(stems) * 440), (23, 26, 33))
    draw = ImageDraw.Draw(sheet)
    metrics = []
    for row, stem in enumerate(stems):
        view = lookup[stem]
        cam = dict(cams[view["cam_id"]])
        scale = height / cam["h"]
        w, h = round(cam["w"] * scale), height
        for k in ("fx", "fy", "cx", "cy"):
            cam[k] *= scale
        cam.update(w=w, h=h)
        gt_path = next((job / "dataset/images").glob(stem + ".*"))
        gt = Image.open(gt_path).convert("RGB").resize((w, h))
        matte = np.asarray(Image.open(job / "dataset/masks" / f"{stem}.png").convert("L").resize((w, h))) / 255
        inside = binary_erosion(matte > .8, iterations=max(1, round(5*height/384)))
        outside = ~binary_dilation(matte > .2, iterations=max(1, round(3*height/384)))
        thumb = gt.copy()
        thumb.thumbnail((216, 384))
        sheet.paste(thumb, (12, row * 440 + 36))
        draw.text((12, row * 440 + 10), f"Input {stem}", fill="white")
        yy, xx = np.indices((h, w))
        checker = np.where(((xx // 16 + yy // 16) % 2)[..., None], .16, .27)
        for col, (name, model) in enumerate(models.items(), 1):
            alpha, rgb = render_rgb_alpha(*model, cam, view["q"], view["t"], w, h)
            composite = np.clip(rgb * alpha[..., None] + checker * (1-alpha[..., None]), 0, 1)
            image = Image.fromarray((composite * 255).astype(np.uint8))
            image.save(out / f"{stem}-{name.replace(' ', '-')}.png")
            np.savez_compressed(out / f"{stem}-{name.replace(' ', '-')}.npz", alpha=alpha, rgb=rgb)
            image.thumbnail((216, 384))
            sheet.paste(image, (col * 240 + 12, row * 440 + 36))
            draw.text((col * 240 + 12, row * 440 + 10), name, fill="white")
            entry = dict(view=stem, model=name, interior_alpha=float(alpha[inside].mean()),
                         interior_below_half=float((alpha[inside] < .5).mean()),
                         outside_alpha=float(alpha[outside].mean()),
                         silhouette_iou=float(((alpha>.5)&(matte>.5)).sum()/max(1,((alpha>.5)|(matte>.5)).sum())))
            metrics.append(entry)
            draw.text((col * 240 + 12, row * 440 + 424), f"body alpha {entry['interior_alpha']:.3f}", fill="white")
            print(json.dumps(entry), flush=True)
        sheet.save(out / "comparison.png")
    aggregate = {name: {key: float(np.mean([m[key] for m in metrics if m["model"] == name]))
                       for key in ("interior_alpha", "interior_below_half", "outside_alpha", "silhouette_iou")}
                 for name in models}
    (out / "metrics.json").write_text(json.dumps(dict(aggregate=aggregate, views=metrics,
        height=height, note="Sampled registered training views; DC color renderer, no mask clipping. Not a held-out quality benchmark."), indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    print(out / "comparison.png", flush=True)


def compare_viewer_camera(job, minimum_score=.9):
    """Inspect the actual viewer's default camera; there is no target image here."""
    from scipy.spatial.transform import Rotation
    job = Path(job).resolve()
    output = job / "subject-recovery" / f"score-{minimum_score:.3f}-max-5000" / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    eye, target, up = np.array([-1., -4., 6.]), np.array([0., 4., 0.]), np.array([0., -1., -.6])
    w, h = 540, 610
    focal = h/(2*np.tan(np.deg2rad(50)/2))
    cam = dict(w=w, h=h, fx=focal, fy=focal, cx=w/2, cy=h/2)
    sheet = Image.new("RGB", (3*w, 2*(h+40)), (23, 26, 33))
    draw = ImageDraw.Draw(sheet)
    candidates = [("Original", job / "subject-extraction/keep.ply"),
                  ("Recovered", next(output.parent.glob("*-recovered-subject.ply"))),
                  ("Uncertain", job / "subject-extraction/uncertain.ply")]
    yy, xx = np.indices((h, w))
    checker = np.where(((xx//16+yy//16)%2)[..., None], .16, .27)
    for row, angle in enumerate((0, np.pi)):
        orbit = Rotation.from_rotvec(up / np.linalg.norm(up) * angle).as_matrix()
        eye_at = target + orbit @ (eye-target)
        forward = target-eye_at
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        matrix = np.stack([right, np.cross(forward, right), forward])
        quat = Rotation.from_matrix(matrix).as_quat()
        quat = np.array([quat[3], *quat[:3]])
        for column, (name, path) in enumerate(candidates):
            xyz, scales, rotations, opacity = load_ply(path)
            data = PlyData.read(path)["vertex"].data
            rgb = np.clip(.5 + .28209479177387814 * np.column_stack([data[f"f_dc_{i}"] for i in range(3)]),0,1)
            alpha, image = render_rgb_alpha(xyz, gauss_cov3d(scales, rotations), opacity, rgb,
                                           cam, quat, -matrix@eye_at, w, h)
            composite = image*alpha[...,None]+checker*(1-alpha[...,None])
            sheet.paste(Image.fromarray((np.clip(composite,0,1)*255).astype(np.uint8)), (column*w,row*(h+40)+40))
            angle_name = "Default" if row == 0 else "Opposite orbit (no reference image)"
            draw.text((column*w+12,row*(h+40)+12), f"{angle_name}: {name}", fill="white")
    sheet.save(output / "viewer-default.png")
    print(output / "viewer-default.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("--score", type=float, default=.9)
    parser.add_argument("--stems", nargs="*")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.viewer:
        compare_viewer_camera(args.job, args.score)
    else:
        compare(args.job, args.score, args.stems, args.height, args.quick)
