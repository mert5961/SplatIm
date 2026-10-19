"""Optional scored recovery from an existing H2 extraction; original files stay intact."""
from __future__ import annotations

import json
import hashlib
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def score_uncertain(job: Path) -> dict:
    """Score uncertain splats using geometry, color and multi-view mask footprints.

    Scores are heuristic support scores, not calibrated probabilities. Only the
    uncertain partition can be recovered; accepted splats are always preserved.
    """
    import sys
    if str(ROOT / "work") not in sys.path:
        sys.path.insert(0, str(ROOT / "work"))
    from splat_alpha_render import load_cameras, quat_to_mat, gauss_cov3d

    scene = job / "scene.ply"
    data = PlyData.read(scene)["vertex"].data
    with np.load(job / "subject-extraction" / "evidence.npz") as evidence:
        fg, bg = evidence["fg"].copy(), evidence["bg"].copy()
    if len(data) != len(fg) or len(fg) != len(bg):
        raise ValueError("Extraction evidence does not match this scene.")
    if not np.isfinite(fg).all() or not np.isfinite(bg).all():
        raise ValueError("Extraction evidence contains non-finite values.")
    fraction = fg / np.maximum(fg + bg, 1e-6)
    keep = (fg >= 5) & (fraction >= .5)
    uncertain = ~(keep | (fg < 2))
    baseline = PlyData.read(job / "subject-extraction" / "keep.ply")["vertex"].data
    if baseline.dtype != data.dtype or baseline.tobytes() != data[keep].tobytes():
        raise ValueError("The saved keep.ply no longer matches its scene/evidence; extract again first.")
    indices = np.flatnonzero(uncertain)
    anchors = np.flatnonzero(keep & (fraction >= .8))
    if len(anchors) < 16:
        anchors = np.flatnonzero(keep)
    if len(anchors) < 2 or not len(indices):
        raise ValueError("This extraction needs accepted and uncertain splats for recovery.")
    xyz = np.column_stack([data[k] for k in ("x", "y", "z")])
    scales = np.exp(np.column_stack([data[f"scale_{i}"] for i in range(3)]))
    rgb = np.clip(.5 + .28209479177387814 * np.column_stack(
        [data[f"f_dc_{i}"] for i in range(3)]), 0, 1)
    if not np.isfinite(xyz).all() or not np.isfinite(scales).all():
        raise ValueError("Scene geometry contains non-finite values.")
    distance, neighbors = cKDTree(xyz[anchors]).query(xyz[indices], k=min(8, len(anchors)))
    nearby = anchors[neighbors]
    local_radius = np.maximum(np.median(scales[nearby].max(2), axis=1), 1e-6)
    proximity = np.exp(-.5 * (distance[:, 0] / (2 * local_radius)) ** 2)
    color_distance = np.linalg.norm(rgb[indices, None] - rgb[nearby], axis=2)
    color_support = np.exp(-.5 * (np.min(color_distance, axis=1) / .18) ** 2)
    # Limit the added splats' long axes to the nearby reliable geometry. The
    # accepted partition keeps its exact original attributes.
    cap = 1.5 * np.quantile(scales[nearby].max(2), .75, axis=1)
    candidate_scales = np.minimum(scales[indices], cap[:, None])
    rotation = np.column_stack([data[f"rot_{i}"] for i in range(4)])[indices]
    rotation /= np.maximum(np.linalg.norm(rotation, axis=1, keepdims=True), 1e-8)
    cov = gauss_cov3d(candidate_scales, rotation)
    cams, views = load_cameras(job / "sparse_txt")
    # Gaussian quadrature integrates the projected footprint instead of trusting
    # just its center pixel (which can sit outside a valid body silhouette).
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    offsets = np.array([(x, y) for x in nodes for y in nodes]) * np.sqrt(2)
    weights2 = np.outer(weights, weights).reshape(-1) / np.pi
    support = np.zeros(len(indices))
    observed = np.zeros(len(indices))
    for view in views:
        cam = cams[view["cam_id"]]
        rotation_cam = quat_to_mat(view["q"][None])[0]
        pos = xyz[indices] @ rotation_cam.T + view["t"]
        depth = pos[:, 2]
        z = np.maximum(depth, .2)
        u, v = cam["fx"] * pos[:, 0] / z + cam["cx"], cam["fy"] * pos[:, 1] / z + cam["cy"]
        jac = np.zeros((len(indices), 2, 3))
        jac[:, 0, 0], jac[:, 1, 1] = cam["fx"] / z, cam["fy"] / z
        jac[:, 0, 2], jac[:, 1, 2] = -cam["fx"] * pos[:, 0] / z**2, -cam["fy"] * pos[:, 1] / z**2
        cam_cov = rotation_cam[None] @ cov @ rotation_cam.T[None]
        projected = jac @ cam_cov @ jac.transpose(0, 2, 1)
        projected = (projected + projected.transpose(0, 2, 1)) * .5
        values, vectors = np.linalg.eigh(projected)
        chol = vectors * np.sqrt(np.maximum(values, 0) + .3)[:, None, :]
        sample = np.stack([u, v], axis=1)[:, None] + offsets[None] @ chol.transpose(0, 2, 1)
        sx, sy = np.rint(sample[:, :, 0]).astype(np.int64), np.rint(sample[:, :, 1]).astype(np.int64)
        valid = (depth[:, None] > .2) & (sx >= 0) & (sy >= 0) & (sx < cam["w"]) & (sy < cam["h"])
        with Image.open(job / "dataset" / "masks" / f"{view['stem']}.png") as mask:
            matte = np.asarray(mask.convert("L").resize((cam["w"], cam["h"])), dtype=float) / 255
        sampled = matte[np.clip(sy, 0, cam["h"]-1), np.clip(sx, 0, cam["w"]-1)]
        support += (sampled * valid * weights2).sum(1)
        observed += (valid * weights2).sum(1)
    mask_support = support / np.maximum(observed, 1e-6)
    evidence_support = np.minimum(fg[indices] / 5, 1)
    score = .55 * mask_support + .20 * proximity + .15 * color_support + .10 * evidence_support
    eligible = (mask_support >= .6) & (proximity >= .25) & (observed >= min(5, len(views)))
    score[~eligible] = 0
    return dict(data=data, keep=keep, indices=indices, score=score, mask_support=mask_support,
                proximity=proximity, color_support=color_support, fg=fg[indices],
                fraction=fraction[indices], scales=candidate_scales,
                observed=observed, eligible=eligible, scene_sha256=hashlib.sha256(scene.read_bytes()).hexdigest())


def export_recovery(job: Path, minimum_score: float = .90, max_added: int = 5000,
                    scored: dict | None = None) -> dict:
    if not 0 <= float(minimum_score) <= 1 or int(max_added) < 0:
        raise ValueError("Score must be 0–1 and maximum added splats must be non-negative.")
    scored = score_uncertain(job) if scored is None else scored
    candidate = np.flatnonzero(scored["eligible"] & (scored["score"] >= minimum_score))
    ranked = candidate[np.argsort(-scored["score"][candidate], kind="stable")]
    selected = ranked[:int(max_added)]
    added = scored["data"][scored["indices"][selected]].copy()
    for i in range(3):
        added[f"scale_{i}"] = np.log(scored["scales"][selected, i])
    accepted = scored["data"][scored["keep"]]
    merged = np.concatenate([accepted, added])
    tag = f"score-{minimum_score:.3f}-max-{int(max_added)}"
    output = job / "subject-recovery" / tag
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((job / "job.json").read_text(encoding="utf-8"))
    name = Path(state["source"]).stem
    final = output / f"{name}-recovered-subject.ply"
    recovered = output / "recovered-only.ply"
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(final)
    PlyData([PlyElement.describe(added, "vertex")], text=False).write(recovered)
    np.savez_compressed(output / "scores.npz", **{key: scored[key] for key in
        ("indices", "score", "mask_support", "proximity", "color_support", "fg", "fraction", "observed", "eligible")},
        recovered_indices=scored["indices"][selected])
    preview = []
    added_set = set(selected.tolist())
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scene_index", "support_score", "mask_support", "near_body", "color_match", "added"])
        for slot in np.argsort(-scored["score"], kind="stable"):
            row = [int(scored["indices"][slot]), round(float(scored["score"][slot]), 4),
                   round(float(scored["mask_support"][slot]), 4), round(float(scored["proximity"][slot]), 4),
                   round(float(scored["color_support"][slot]), 4), int(slot) in added_set]
            writer.writerow(row)
            if len(preview) < 50:
                preview.append(row)
    report = dict(original_count=int(len(accepted)), uncertain_count=len(scored["indices"]),
                  recovered_count=len(added), final_count=len(merged), minimum_score=float(minimum_score),
                  maximum_added=int(max_added), scene_sha256=scored["scene_sha256"],
                  final=str(final), recovered=str(recovered), report=str(output / "report.json"),
                  scores=str(output / "scores.csv"), evidence=str(output / "scores.npz"),
                  scoring_version=1,
                  note="Heuristic support score, not a probability. Original kept splats are unchanged; only added splats have local size caps.")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return dict(report, preview=preview)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    parser.add_argument("--score", type=float, default=.9)
    parser.add_argument("--max-added", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps({k: v for k, v in export_recovery(args.job, args.score, args.max_added).items()
                      if k != "preview"}, indent=2))
